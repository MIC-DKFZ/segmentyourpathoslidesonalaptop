import argparse
import json
import numpy as np
from skimage.draw import polygon
from shapely.geometry import shape
import openslide
import os

def geojson_to_segmentation_mask(geojson_path, wsi_path, level, output_path):
    """
    Converts a GeoJSON file into a binary NumPy segmentation mask by rasterizing
    the polygons onto the dimensions of a specified WSI level.
    """
    # 1. Get the dimensions (H, W) from the WSI at the specified level
    try:
        slide = openslide.OpenSlide(wsi_path)
        # slide.level_dimensions returns a tuple of (width, height) for all levels
        wsi_level_dims = slide.level_dimensions[level]
        image_width, image_height = wsi_level_dims
        print(f"Target mask dimensions at Level {level}: ({image_height}, {image_width})")
    except Exception as e:
        print(f"Error reading WSI or level dimensions: {e}")
        return

    # 2. Initialize the blank segmentation mask
    # The mask array must be (Height, Width)
    segmentation_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 3. Calculate the downsampling factor
    # GeoJSON coordinates are typically in the Level 0 (full resolution) reference frame.
    # We need to scale the coordinates down to the target 'level' size.
    downsample_factor = slide.level_downsamples[level]
    print(f"Scaling factor (Level 0 to Level {level}): {downsample_factor}")

    # 4. Load the GeoJSON data
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return

    features = data.get("features", [])
    if not features:
        print("Warning: GeoJSON file contains no features. Saving empty mask.")
        np.save(output_path, segmentation_mask)
        return

    # 5. Iterate through features and rasterize
    print(f"Rasterizing {len(features)} features...")
    
    for feature in features:
        geom = shape(feature.get("geometry"))
        
        if geom.geom_type == 'Polygon':
            # Get exterior coordinates (x, y) in Level 0
            exterior_coords = np.array(geom.exterior.coords)
            
            # Scale coordinates to the target level
            scaled_coords = exterior_coords / downsample_factor

            # Separate x (column) and y (row) coordinates
            # GeoJSON/WSI coordinates are (x, y) -> (Column, Row)
            r = scaled_coords[:, 1]  # Y-coordinates -> Row
            c = scaled_coords[:, 0]  # X-coordinates -> Column

            # Rasterize the polygon using skimage.draw
            # We must use integer coordinates for pixel drawing
            rr, cc = polygon(r.astype(int), c.astype(int), shape=segmentation_mask.shape)
            
            # Set the pixels within the polygon to 255 (or 1)
            segmentation_mask[rr, cc] = 255 
            
            # NOTE: If you have holes/interiors, you will need to iterate 
            # through geom.interiors and set the pixels to 0 (as shown in the previous response).

    # 6. Save the resulting NumPy array
    np.save(output_path, segmentation_mask)
    print(f"✅ Successfully saved segmentation mask to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoJSON polygon annotations (Level 0 coordinates) into a Level-specific NumPy segmentation mask for a Whole-Slide Image (WSI)."
    )
    # Required arguments
    parser.add_argument('--wsi', type=str, required=True, help='Path to the input WSI file (.svs).')
    parser.add_argument('--geojson', type=str, required=True, help='Path to the input GeoJSON annotation file.')
    parser.add_argument('--out', type=str, required=True, help='Path to save the output NumPy segmentation mask (.npy).')
    parser.add_argument('--level', type=int, default=0, help='The WSI pyramid level to rasterize the mask to (e.g., 0 for full resolution). Default is 0.')

    args = parser.parse_args()

    geojson_to_segmentation_mask(
        geojson_path=args.geojson,
        wsi_path=args.wsi,
        level=args.level,
        output_path=args.out
    )

if __name__ == "__main__":
    main()