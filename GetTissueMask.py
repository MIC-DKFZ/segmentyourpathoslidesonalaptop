#!/usr/bin/env python3
"""
Create a tissue mask for a Whole Slide Image (WSI) at level 0 resolution.

Output: a NumPy boolean array (True = tissue) with the same width/height as level 0.

Requirements:
  - openslide-python
  - numpy
  - scikit-image (skimage)
  - pillow

Example:
  python make_tissue_mask.py /path/to/slide.svs --out mask_level0.npy

Notes:
  - Generating a full-resolution mask can be very large (e.g., 100k x 100k -> 10^10 booleans ~ 10 GB if stored densely as bytes).
    Use the --out mask.npy to save to disk as a .npy (booleans). Consider using downsampled masks for most tasks.
"""

import argparse
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('TkAGG')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import openslide
except ImportError as e:
    print("Error: openslide-python is required. pip install openslide-python", file=sys.stderr)
    raise

try:
    from skimage import color, filters, morphology, util
except ImportError:
    print("Error: scikit-image is required. pip install scikit-image", file=sys.stderr)
    raise


def pick_working_level(slide, max_dim: int) -> int:
    """Pick the coarsest level whose larger dimension is <= max_dim (for quick tissue detection)."""
    w0, h0 = slide.level_dimensions[0]
    # If level 0 already small enough, use it
    if max(w0, h0) <= max_dim:
        return 0
    # Otherwise, pick the lowest level that meets the constraint
    for lvl, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) <= max_dim:
            return lvl
    # If none are small enough, pick the last level (lowest resolution)
    return slide.level_count - 1


def read_level_as_rgb(slide, level: int) -> np.ndarray:
    """Read an entire level as an RGB numpy array (H, W, 3), dtype=uint8."""
    w, h = slide.level_dimensions[level]
    # read_region expects level relative (w,h)
    img = slide.read_region((0, 0), level, (w, h))  # RGBA Pill Image
    img = img.convert("RGB")
    return np.asarray(img)


def make_tissue_mask_lowres(rgb: np.ndarray,
                            method: str = "otsu",
                            min_obj_area: int = 500,
                            hole_area: int = 500) -> np.ndarray:

    rgb_f = util.img_as_float32(rgb)

    if method.lower() == "hsv":
        hsv = color.rgb2hsv(rgb_f)
        s = hsv[..., 1]
        v = hsv[..., 2]
        # Tissue often has higher saturation and lower value than background.
        # Combine simple normalized criterion then Otsu on that score.
        score = s - v * 0.2
        thresh = filters.threshold_otsu(score)
        mask = score > thresh
    else:  # 'otsu'
        gray = color.rgb2gray(rgb_f)
        # Background is usually bright; tissue darker -> gray < thresh
        thresh = filters.threshold_otsu(gray)
        mask = gray < thresh

    # Morphological cleanup
    if min_obj_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_obj_area)
    if hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=hole_area)

    # Optional open/close to smooth boundaries
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask


def upsample_mask_to_level0(mask_low: np.ndarray, level_size: tuple[int, int]) -> np.ndarray:
    """Upsample a low-res boolean mask to level 0 size using nearest-neighbor.

    Returns a boolean array of shape (H0, W0).
    """
    H0, W0 = level_size[1], level_size[0]
    pil_mask = Image.fromarray(mask_low.astype(np.uint8) * 255, mode="L")
    # Use NEAREST to preserve binary mask without smoothing
    pil_up = pil_mask.resize((W0, H0), resample=Image.NEAREST)
    mask0 = np.asarray(pil_up) > 127
    return mask0


def save_mask(mask: np.ndarray, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".npy"}:
        np.save(out_path, mask)
    elif out_path.suffix.lower() in {".png", ".tif", ".tiff"}:
        img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        img.save(out_path)
    else:
        # default to .npy if no/unknown extension
        np.save(out_path.with_suffix('.npy'), mask)



def generate_tissue_mask(inference_slide, max_dim=3000, method='otsu', min_obj_area=500,hole_area=20000):
    slide=inference_slide
    max_dimensions=max_dim
    threshold_method=method
    minimum_object_area=min_obj_area
    hole_area_size=hole_area



    W0, H0 = slide.level_dimensions[0]
    print(f"Level 0 size: {W0} x {H0}")

    lvl = pick_working_level(slide, max_dimensions)
    Wl, Hl = slide.level_dimensions[lvl]
    ds = slide.level_downsamples[lvl]
    print(f"Using level {lvl} ({Wl}x{Hl}), downsample ~ {ds:.2f}x from level 0")

    rgb_low = read_level_as_rgb(slide, lvl)
    mask_low = make_tissue_mask_lowres(rgb_low, method=threshold_method,min_obj_area=minimum_object_area,hole_area=hole_area_size)

    print("Upsampling mask to level 0... (this may be large)")
    mask0 = upsample_mask_to_level0(mask_low, (W0, H0))

    print(f"Mask shape: {mask0.shape}, dtype: {mask0.dtype}, tissue pixels: {int(mask0.sum())}")

    return mask0



