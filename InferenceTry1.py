#!/usr/bin/env python3
"""
infer_medsam_to_geojson.py

Run tiled inference with a fine-tuned MedSAM model over a WSI, save a stitched
binary mask (.npy), and export polygons as a GeoJSON ready for QuPath.

Requires:
  - openslide-python
  - torch, torchvision
  - segment-anything (the same you used for training)
  - skimage OR opencv-python (for polygonization; skimage preferred)

Author: you :)
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Optional
import glob
import numpy as np
import torch
import torch.nn.functional as F_torch
from PIL import Image

import openslide
from segment_anything import sam_model_registry
from torchvision.transforms import functional as tvF
from tqdm import tqdm
from GetTissueMask import generate_tissue_mask
from ConvertSegmentationArray import export_geojson
# ---- Optional polygonization backends ----
_SKIMAGE_OK = False
_CV2_OK = False
try:
    from skimage import measure
    _SKIMAGE_OK = True
except Exception:
    pass

try:
    import cv2
    _CV2_OK = True
except Exception:
    pass


# ----------------------- Utilities -----------------------

def get_level_for_target_mag(slide: openslide.OpenSlide, target_mag: float = 20.0) -> int:
    """
    Heuristic selection of OpenSlide level based on desired objective magnification.
    Uses slide properties if available; otherwise pick highest level where min(dim) >= 1024.
    """
    try:
        base_mag = float(slide.properties.get("aperio.AppMag", 40.0))
        best, best_diff = 0, float("inf")
        for l, ds in enumerate(slide.level_downsamples):
            est_mag = base_mag / float(ds)
            diff = abs(est_mag - target_mag)
            if diff < best_diff:
                best, best_diff = l, diff
        return best
    except Exception:
        best = slide.level_count - 1
        for l in range(slide.level_count):
            w, h = slide.level_dimensions[l]
            if min(w, h) >= 1024:
                best = l
        return best


def load_model(weights_path: str, model_type: str, device: torch.device):
    sam = sam_model_registry[model_type]()
    state_dict = torch.load(weights_path, map_location=device)
    sam.load_state_dict(state_dict, strict=True)
    sam.to(device)
    sam.eval()
    return sam


def read_tile(slide, level: int, x: int, y: int, tile_size: int) -> np.ndarray:
    """Read RGB tile at (x,y) (top-left) for given level as HxWx3 uint8."""
    region = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
    return np.array(region, dtype=np.uint8)


def run_sam_on_tile(model, device, img_np: np.ndarray, full_tile_box: bool = True) -> np.ndarray:
    """
    Run MedSAM on a single tile:
      - Create tensor (1,3,H,W)
      - Use a box prompt. If full_tile_box=True, use the whole tile as the box.
      - Return logits (H,W) before sigmoid.
    """
    img_t = tvF.to_tensor(img_np).unsqueeze(0).to(device)  # [1,3,H,W], float32 0..1
    with torch.no_grad():
        image_embeddings = model.image_encoder(img_t)

        H, W = img_np.shape[:2]
        if full_tile_box:
            # [x_min, y_min, x_max, y_max]
            box = torch.tensor([[0.0, 0.0, float(W), float(H)]], dtype=torch.float32, device=device)
        else:
            # fallback: small center box
            cxy = (W // 2, H // 2)
            box = torch.tensor([[cxy[0]-5, cxy[1]-5, cxy[0]+5, cxy[1]+5]], dtype=torch.float32, device=device)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box.unsqueeze(0),  # [B, 1, 4] or [1,1,4]
            masks=None
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale logits to original tile size
        up = model.postprocess_masks(
            low_res_masks,
            (H, W),
            (H, W)
        )  # [1,1,H,W]
        logits = up[0, 0]  # [H,W]
        return logits.detach().cpu().numpy()


def stitch_tiles(
    slide, level: int, model, device, tile_size: int, stride: int
) -> np.ndarray:
    """
    Slide over the chosen level with given tile_size/stride, run SAM per tile,
    aggregate stitched logits by taking the max over overlaps.
    Returns a 2D float32 array of logits (H,W).
    """
    W, H = slide.level_dimensions[level]
    logits_full = np.full((H, W), -np.inf, dtype=np.float32)

    xs = list(range(0, max(1, W - tile_size + 1), stride))
    ys = list(range(0, max(1, H - tile_size + 1), stride))
    if xs[-1] != W - tile_size:
        xs.append(max(0, W - tile_size))
    if ys[-1] != H - tile_size:
        ys.append(max(0, H - tile_size))

    for y in tqdm(ys):
        for x in tqdm(xs):
            tile = read_tile(slide, level, x, y, tile_size)
            logits = run_sam_on_tile(model, device, tile, full_tile_box=True)  # HxW
            # Max-aggregate into canvas
            patch = logits_full[y:y+tile_size, x:x+tile_size]
            np.maximum(patch, logits, out=patch)

    return logits_full


def binarize_logits(logits: np.ndarray, threshold: float) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(np.uint8)


def polygons_from_mask(mask: np.ndarray, min_area: int = 0) -> List[np.ndarray]:
    """
    Extract polygon exterior coordinates in (x,y) pixel space from a binary mask (H,W).
    Returns list of Nx2 arrays. min_area filters small regions (in pixels^2).
    """
    polys_xy = []


    if _CV2_OK:
        # OpenCV fallback: find external contours
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < max(1, float(min_area)):
                continue
            cnt = cnt.reshape(-1, 2)  # (N,2) x,y
            polys_xy.append(cnt.astype(np.float64))
        return polys_xy

    # Minimal fallback: raster scan bbox (coarse, not recommended)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    rect = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]], dtype=np.float64)
    return [rect]





# ----------------------- Main -----------------------

slides=glob.glob('/Volumes/INTENSO/Projekt Zystennieren/Histobilder Nieren PKD/Niere li/Niere li 1/*.svs')
weights='finetuned_medsam_cyst_segmentation.pth'
model_type='vit_b'
tile_size=1024
stride=1024
level=0
target_magnification=40
threshold=0.5
min_area=1000
class_name='cyst'
device="mps"
device = torch.device(device)
for wsi in slides:
    slide = openslide.OpenSlide(wsi)

    if level is not None:
        level = level
    else:
        if target_magnification is None:
            level = 2  # safe default mirroring your training stub
        else:
            level = get_level_for_target_mag(slide, target_magnification)
    downsample = float(slide.level_downsamples[level])  # level -> level-0
    W, H = slide.level_dimensions[level]
    model = load_model(weights, model_type, device)
    print('model_loaded')
    logits = stitch_tiles(
        slide=slide,
        level=level,
        model=model,
        device=device,
        tile_size=tile_size,
        stride=stride
    )

    mask = binarize_logits(logits, threshold=threshold)  # uint8 {0,1}
    tissue_mask=generate_tissue_mask(slide)
    out_path=wsi.replace('/Volumes/INTENSO/Projekt Zystennieren/Histobilder Nieren PKD','/Users/maximilianfischer/PycharmProjects/MedSam/Inference').replace('.svs','.geojson')
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    export_geojson(mask,tissue_mask,out_path)



