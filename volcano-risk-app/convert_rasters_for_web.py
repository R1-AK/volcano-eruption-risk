"""
Convert GeoTIFF rasters to PNG for web display.
Reads from validation_outputs/ and saves to ../volcano-risk-web/public/rasters/
"""

import os
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Paths
SOURCE_DIR = "validation_outputs"
TARGET_DIR = "../volcano-risk-web/public/rasters"

# LULC colormap (TerraMind classes)
LULC_COLORS = {
    0: (0, 0, 0),  # No data - black
    1: (0, 0, 255),  # Water - blue
    2: (34, 139, 34),  # Trees - forest green
    3: (144, 238, 144),  # Flooded veg - light green
    4: (255, 215, 0),  # Crops - gold
    5: (255, 0, 0),  # Built - red
    6: (210, 180, 140),  # Bare - tan
    7: (255, 255, 255),  # Snow/ice - white
    8: (128, 128, 128),  # Clouds - gray
    9: (154, 205, 50),  # Rangeland - yellow-green
}


def normalize_dem_to_rgb(dem_array):
    """Convert DEM to RGB using terrain colormap."""
    # Remove NaN values
    valid_mask = ~np.isnan(dem_array)

    if not valid_mask.any():
        return np.zeros((dem_array.shape[0], dem_array.shape[1], 3), dtype=np.uint8)

    # Normalize to 0-1
    mn = np.nanmin(dem_array)
    mx = np.nanmax(dem_array)

    if mx - mn < 1:  # Avoid division by zero
        normalized = np.zeros_like(dem_array)
    else:
        normalized = (dem_array - mn) / (mx - mn)

    # Apply terrain colormap
    cmap = plt.cm.terrain
    rgb = cmap(normalized)[:, :, :3]  # Drop alpha channel
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    return rgb_uint8


def lulc_to_rgb(lulc_array):
    """Convert LULC class indices to RGB using predefined colors."""
    h, w = lulc_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in LULC_COLORS.items():
        mask = lulc_array == class_id
        rgb[mask] = color

    return rgb


def convert_volcano_rasters(volcano_folder):
    """Convert DEM and LULC for a single volcano."""
    safe_name = os.path.basename(volcano_folder)

    # Paths to input TIFFs
    dem_tif = os.path.join(volcano_folder, "dem_copernicus_real.tif")
    lulc_tif = os.path.join(volcano_folder, "lulc_terramind_ai.tif")

    # Output paths
    dem_png = os.path.join(TARGET_DIR, f"{safe_name}_DEM.png")
    lulc_png = os.path.join(TARGET_DIR, f"{safe_name}_LULC.png")

    success_count = 0

    # Convert DEM
    if os.path.exists(dem_tif):
        try:
            with rasterio.open(dem_tif) as src:
                dem_array = src.read(1)

            rgb = normalize_dem_to_rgb(dem_array)
            Image.fromarray(rgb).save(dem_png)
            print(f"  ✓ DEM: {dem_png}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ DEM failed: {e}")

    # Convert LULC
    if os.path.exists(lulc_tif):
        try:
            with rasterio.open(lulc_tif) as src:
                lulc_array = src.read(1).astype(np.uint8)

            rgb = lulc_to_rgb(lulc_array)
            Image.fromarray(rgb).save(lulc_png)
            print(f"  ✓ LULC: {lulc_png}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ LULC failed: {e}")

    return success_count


def main():
    """Process all volcanoes in validation_outputs."""
    print("=" * 70)
    print("CONVERTING GEOTIFF RASTERS TO PNG FOR WEB DISPLAY")
    print("=" * 70)

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"\nTarget directory: {TARGET_DIR}")

    # Check if source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"\nERROR: {SOURCE_DIR} not found!")
        print("Please run export_dem_lulc.py first to generate rasters.")
        return

    # Get all volcano folders
    volcano_folders = [
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, f))
    ]

    if not volcano_folders:
        print(f"\nNo volcano folders found in {SOURCE_DIR}")
        return

    print(f"\nFound {len(volcano_folders)} volcano folders")
    print(f"Processing...\n")

    total_converted = 0

    for i, folder in enumerate(volcano_folders, 1):
        volcano_name = os.path.basename(folder).replace("_", " ")
        print(f"[{i}/{len(volcano_folders)}] {volcano_name}")

        count = convert_volcano_rasters(folder)
        total_converted += count

    print(f"\n{'=' * 70}")
    print("CONVERSION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Processed: {len(volcano_folders)} volcanoes")
    print(f"Converted: {total_converted} rasters")
    print(f"\nPNG files saved to: {TARGET_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Start your Next.js dev server: npm run dev")
    print(f"  2. Open http://localhost:3000")
    print(f"  3. Click a volcano marker")
    print(f"  4. Toggle DEM/LULC layers on/off")


if __name__ == "__main__":
    main()
