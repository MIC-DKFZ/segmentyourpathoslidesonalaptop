import torch
import numpy as np
import os
import cv2  # Used for finding contours to generate polygons
import json
import uuid  # ADDED: For generating unique feature IDs
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib
matplotlib.use('TkAGG')
import matplotlib.pyplot as plt
from PIL import Image
import time
#array=np.load('segmentation_mask.npy')

stop
# --- Configuration (Adjust as needed) ---
# INPUTS
MODEL_TYPE = "vit_b"
FINETUNED_CHECKPOINT_PATH = "finetuned_medsam_cyst_segmentation.pth"  # Output from the training script
TARGET_WSI_PATH = "/Users/maximilianfischer/PycharmProjects/MedSam/WSIs/target_slide_01.svs"  # Path to the WSI you want to segment

# Tiling & WSI Parameters
TILE_SIZE = 1024  # Must match the size used during training
MAGNIFICATION_LEVEL = 40  # WSI level to extract tiles from (e.g., 20x or level 2)

# OUTPUTS
OUTPUT_NPY_MASK = "segmentation_mask.npy"
OUTPUT_GEOJSON = "qupath_segmentation.geojson"

# DEVICE
DEVICE = torch.device("mps")  # Or "cuda" or "cpu"

# --- 0. Mocking openslide and setup (as in the original code) ---
try:
    import openslide

    print("OpenSlide library detected.")


    def get_openslide_level(wsi_slide, target_mag):
        """
        Determines the best openslide level index based on the requested magnification.
        This is a common but specific implementation detail for WSI processing.
        We'll use a fixed level for simplicity, assuming MAGNIFICATION_LEVEL=20 corresponds to level 2.
        """
        # A real implementation would involve checking slide properties, but here we assume level 2
        # is the 20x view (or similar level index for speed).
        if target_mag == 40 and len(wsi_slide.level_downsamples) > 0:
            return 0
        elif target_mag == 20 and len(wsi_slide.level_downsamples) > 1:
            return 1
        elif target_mag == 10 and len(wsi_slide.level_downsamples) > 2:
            return 2


except ImportError:
    print("WARNING: openslide not found. Inference will not run correctly without real WSI handling.")
    openslide = None


# --- 1. Model Initialization ---

def load_finetuned_predictor(checkpoint_path, model_type, device):
    """Loads the finetuned SAM model and returns a SamPredictor instance."""
    print(f"Loading finetuned SAM model from: {checkpoint_path}")

    # 1. Initialize the model architecture
    try:
        sam: Sam = sam_model_registry[model_type]()
    except KeyError:
        print(f"Error: Model type {model_type} not found in SAM registry.")
        return None

    # 2. Load the finetuned state dictionary
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        sam.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Ensure finetuning ran successfully.")
        return None

    sam.to(device)
    # The predictor is required for easy inference and image pre-processing
    predictor = SamPredictor(sam)
    print("Finetuned SamPredictor loaded successfully.")
    return predictor


# --- 2. Inference and Stitching ---

def run_wsi_inference(predictor: SamPredictor, wsi_path: str, tile_size: int):
    """
    Performs sliding window inference on a WSI and returns the stitched full mask.
    """
    if openslide is None:
        print("Cannot run inference: openslide is not available.")
        return None, 1.0

    print(f"Starting inference on WSI: {wsi_path}")
    slide = openslide.OpenSlide(wsi_path)

    # 1. Determine Level and Downsampling Factor
    level_idx = get_openslide_level(slide, MAGNIFICATION_LEVEL)
    level_downsample = slide.level_downsamples[level_idx]
    level_width, level_height = slide.level_dimensions[level_idx]

    print(f"Using Level {level_idx} (Approx. {MAGNIFICATION_LEVEL}x). Dimensions: {level_width}x{level_height}")

    # Initialize the full mask array at the processing level's resolution
    full_mask = np.zeros((level_height, level_width), dtype=np.uint8)

    # 2. Define the sliding window grid
    x_coords = range(0, level_width, tile_size)
    y_coords = range(0, level_height, tile_size)

    # 3. Iterate through tiles and run inference
    total_tiles = len(x_coords) * len(y_coords)
    pbar = tqdm(total=total_tiles, desc="Segmenting WSI Tiles")

    # The SAM model requires the original Level 0 coordinates for read_region
    # Location = (x_level_0, y_level_0)

    for y_level in y_coords:
        for x_level in x_coords:
            # Calculate the top-left Level 0 coordinate for openslide.read_region
            x_level_0 = int(x_level * level_downsample)
            y_level_0 = int(y_level * level_downsample)

            # Adjust read size if we are at the edge of the slide
            current_tile_width = min(tile_size, level_width - x_level)
            current_tile_height = min(tile_size, level_height - y_level)

            # Skip very small edge tiles if they are too small (e.g., less than 10% of tile_size)
            if current_tile_width < tile_size // 10 or current_tile_height < tile_size // 10:
                pbar.update(1)
                continue

            # Read the tile from the WSI
            tile_img = slide.read_region(
                location=(x_level_0, y_level_0),
                level=level_idx,
                size=(current_tile_width, current_tile_height)
            ).convert("RGB")

            tile_img_np = np.array(tile_img)

            # --- Inference Step ---
            predictor.set_image(tile_img_np)

            # Prompt Strategy: Use a large bounding box covering the entire tile.
            # This prompts SAM to segment any 'cyst' it finds within this region.
            box_prompt = np.array([0, 0, current_tile_width, current_tile_height])

            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt[None, :],
                multimask_output=False,  # We want the single best prediction
            )

            # The mask shape is (1, H, W)
            predicted_mask = masks[0]

            save_mask=np.asarray(predicted_mask)#.cpu()
            save_mask=Image.fromarray(save_mask)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            save_mask.save(os.path.join('Output',timestr+'_mask.png'))
            tile_img.save(os.path.join('Output',timestr+'_tile.png'))


            # --- Stitching ---
            # Resize the mask back to the original read size if SAM performed internal resizing
            # (SAM output is 256x256 logits, then upscaled to the input size which is the tile size)

            # Ensure the mask is a binary uint8 array matching the tile size
            stitched_mask = (predicted_mask > 0).astype(np.uint8)

            # Place the mask into the full mask array
            full_mask[
            y_level:y_level + current_tile_height,
            x_level:x_level + current_tile_width
            ] = stitched_mask[:current_tile_height, :current_tile_width]

            pbar.update(1)

    slide.close()
    print("\nInference complete. Full mask generated.")
    return full_mask, level_downsample


# --- 3. Mask to GeoJSON Conversion for QuPath ---

def mask_to_geojson(mask_np: np.ndarray, downsample_factor: float, output_path: str):
    """
    Converts a binary segmentation mask (at the processing level) to GeoJSON format.
    The coordinates are scaled back to Level 0 (WSI full resolution).
    MODIFIED: Added UUID and removed integer casting for coordinates.
    """
    if mask_np is None:
        print("Cannot convert to GeoJSON: mask_np is None.")
        return

    print(f"Converting mask to GeoJSON (scaling factor: {downsample_factor})...")

    # Find contours (polygons) in the binary mask
    # cv2.RETR_EXTERNAL retrieves only the outer contours
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(
        mask_np.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features = []
    """
    # Iterate through all detected contours
    for contour in tqdm(contours, desc="Processing Contours"):
        # Contours are (N, 1, 2) where N is the number of points. Reshape to (N, 2)
        contour = contour.squeeze()

        # We need at least 3 points to define a polygon
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue

        # Scale coordinates from the processing level back to WSI Level 0
        # CRITICAL: Removed .astype(np.int64) to maintain float precision like QuPath exports
        scaled_coords = (contour * downsample_factor)

        # GeoJSON coordinates are (x, y) - OpenSlide coords are (x, y)
        coordinates = scaled_coords.tolist()

        # QuPath GeoJSON requires the path to be closed (first and last coordinate must match)
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        # Create the GeoJSON Feature object
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),  # ADDED: Unique ID for QuPath compatibility
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]  # GeoJSON polygons are lists of rings
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": "Cyst"  # Label for QuPath
                }
            }
        }
        features.append(feature)
    """
    #####Modified new section
    mask_u8 = (mask_np > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours, bail early
    if not contours:
        print("No contours found; nothing to write.")
        return

    # Normalize hierarchy
    h = None
    if hierarchy is not None and hierarchy.size > 0 and hierarchy.shape[1] > 0:
        h = hierarchy[0]  # shape (N, 4)

    features = []

    for i, contour in enumerate(contours):
        # If we have hierarchy, skip non-outer contours (parent != -1)
        if h is not None and h[i, 3] != -1:
            continue

        # Simplify and filter tiny outer rings
        contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)
        if cv2.contourArea(contour) < 50:
            continue

        outer = (contour.squeeze().astype(np.float64) * downsample_factor).tolist()
        if outer[0] != outer[-1]:
            outer.append(outer[0])

        # Collect holes for this outer ring
        holes = []
        if h is not None:
            child_idx = h[i, 2]
            while child_idx != -1:
                hole = contours[child_idx]
                hole = cv2.approxPolyDP(hole, epsilon=1.0, closed=True)
                if cv2.contourArea(hole) >= 50:
                    ring = (hole.squeeze().astype(np.float64) * downsample_factor).tolist()
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
                    holes.append(ring)
                child_idx = h[child_idx, 0]

        coords = [outer] + holes
        features.append({
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {"objectType": "annotation", "classification": {"name": "Cyst"}}
        })
        ##########end of new section

    # Final GeoJSON FeatureCollection structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Save the GeoJSON file
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)

    print(f"Successfully generated QuPath GeoJSON file: {output_path}")


# --- 4. Main Execution ---

if __name__ == "__main__":
    if openslide is None:
        print("ERROR: openslide is required for WSI inference. Please install it to run this script.")
    if TARGET_WSI_PATH == "/Users/maximilianfischer/PycharmProjects/MedSam/WSIs/target_slide_01.svs":
        print("NOTE: Please update the TARGET_WSI_PATH variable to point to your actual .svs file.")

    # A. Load Model
    predictor = load_finetuned_predictor(FINETUNED_CHECKPOINT_PATH, MODEL_TYPE, DEVICE)

    if predictor:
        # B. Run Inference and Stitching
        full_mask_npy, downsample_factor = run_wsi_inference(predictor, TARGET_WSI_PATH, TILE_SIZE)

        if full_mask_npy is not None:
            # C. Save the Intermediate NumPy Mask
            np.save(OUTPUT_NPY_MASK, full_mask_npy)
            print(f"Intermediate segmentation mask saved to: {OUTPUT_NPY_MASK}")

            # D. Convert NumPy Mask to GeoJSON
            mask_to_geojson(full_mask_npy, downsample_factor, OUTPUT_GEOJSON)

            print("\nPipeline complete. The GeoJSON file is ready to be imported into QuPath.")