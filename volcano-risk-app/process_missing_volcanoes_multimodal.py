"""
Process remaining volcanoes without Sentinel-2 data using TerraMind's multi-modal capabilities.

SIMPLIFIED Strategy (NO S2):
1. Try Sentinel-1 RTC (radar) → generate LULC + download DEM
2. If no S1, use DEM only → generate LULC

This script SKIPS Sentinel-2 entirely as it causes poor quality for these 183 volcanoes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import torch

import rioxarray
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from scipy.ndimage import zoom

from pystac_client import Client
import planetary_computer

# TerraMind imports
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.tasks.tiled_inference import tiled_inference


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "terramind_model": "terramind_v1_base_generate",
    "generation_timesteps": 10,
    "lulc_crop_size": 256,
    "lulc_stride": 192,
    "lulc_batch_size": 4,
    "region_size": 1000,
    "buffer_km": 10.0,
}

CSV_PATH = "data/volcano_catalog_clean.csv"
OUTPUT_BASE = "validation_outputs_multimodal"
YEAR = 2024

# Global model handles
_S2_GENERATOR = None
_S1_GENERATOR = None
_DEM_GENERATOR = None


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def init_s2_generator():
    """Initialize Sentinel-2 → LULC/DEM generator."""
    global _S2_GENERATOR
    if _S2_GENERATOR is None:
        print("Loading S2 → LULC/DEM generator...")
        _S2_GENERATOR = FULL_MODEL_REGISTRY.build(
            CONFIG["terramind_model"],
            modalities=["S2L2A"],
            output_modalities=["LULC", "DEM"],
            pretrained=True,
            standardize=True,
            timesteps=CONFIG["generation_timesteps"],
        )
        _S2_GENERATOR.eval()
        print("  ✓ S2 generator loaded")
    return _S2_GENERATOR


def init_s1_generator():
    """Initialize Sentinel-1 RTC → LULC/DEM generator."""
    global _S1_GENERATOR
    if _S1_GENERATOR is None:
        print("Loading S1 RTC → LULC/DEM generator...")
        try:
            _S1_GENERATOR = FULL_MODEL_REGISTRY.build(
                CONFIG["terramind_model"],
                modalities=["S1RTC"],  # Sentinel-1 RTC modality
                output_modalities=["LULC", "DEM"],
                pretrained=True,
                standardize=True,
                timesteps=CONFIG["generation_timesteps"],
            )
            _S1_GENERATOR.eval()
            print("  ✓ S1 generator loaded")
        except Exception as e:
            print(f"  ⚠️ S1 generator not available: {e}")
            _S1_GENERATOR = None
    return _S1_GENERATOR


def init_dem_generator():
    """Initialize DEM → LULC generator."""
    global _DEM_GENERATOR
    if _DEM_GENERATOR is None:
        print("Loading DEM → LULC generator...")
        try:
            _DEM_GENERATOR = FULL_MODEL_REGISTRY.build(
                CONFIG["terramind_model"],
                modalities=["DEM"],
                output_modalities=["LULC"],
                pretrained=True,
                standardize=True,
                timesteps=CONFIG["generation_timesteps"],
            )
            _DEM_GENERATOR.eval()
            print("  ✓ DEM generator loaded")
        except Exception as e:
            print(f"  ⚠️ DEM generator not available: {e}")
            _DEM_GENERATOR = None
    return _DEM_GENERATOR


# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================

def download_sentinel2_region(
    lat: float,
    lon: float,
    year: int,
    region_size: int = 1000,
    buffer_km: float = 10.0,
) -> Tuple[Optional[np.ndarray], Optional[rasterio.transform.Affine]]:
    """Download Sentinel-2 region. Returns None if not available."""
    print(f"  Searching Sentinel-2 for lat={lat}, lon={lon}...")

    delta_deg = buffer_km / 111.0
    bbox = [lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg]

    api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(api_url, modifier=planetary_computer.sign_inplace)

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        max_items=50,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": 20}},
    )

    items = list(search.items())

    if not items:
        print(f"    No Sentinel-2 data found")
        return None, None

    print(f"    Found {len(items)} items, trying best...")

    sorted_items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))

    band_names = ["B01", "B02", "B03", "B04", "B05", "B06",
                  "B07", "B08", "B8A", "B09", "B11", "B12"]

    for item in sorted_items[:3]:
        try:
            assets = item.assets
            bands_list = []
            transform = None

            for band in band_names:
                href = planetary_computer.sign(assets[band].href)

                with rioxarray.open_rasterio(href) as da:
                    tile_crs = da.rio.crs

                    if tile_crs != "EPSG:4326":
                        bbox_in_tile_crs = transform_bounds(
                            "EPSG:4326", tile_crs, bbox[0], bbox[1], bbox[2], bbox[3]
                        )
                    else:
                        bbox_in_tile_crs = bbox

                    clipped = da.rio.clip_box(
                        minx=bbox_in_tile_crs[0], miny=bbox_in_tile_crs[1],
                        maxx=bbox_in_tile_crs[2], maxy=bbox_in_tile_crs[3],
                        auto_expand=True,
                    )

                    band_arr = clipped.values[0].astype(np.float32)

                    if transform is None:
                        transform = clipped.rio.transform()

                    if band_arr.shape != (region_size, region_size):
                        zoom_factors = (region_size / band_arr.shape[0], region_size / band_arr.shape[1])
                        band_arr = zoom(band_arr, zoom_factors, order=1)

                    band_arr = np.nan_to_num(band_arr, nan=0.0, posinf=10000.0, neginf=0.0)
                    bands_list.append(band_arr)

            region_data = np.stack(bands_list, axis=0)

            west, south, east, north = bbox
            transform = from_bounds(west, south, east, north, region_size, region_size)

            valid_pixels = np.isfinite(region_data).all(axis=0)
            good_ratio = valid_pixels.sum() / (region_size * region_size)

            if good_ratio > 0.3:
                print(f"    ✓ S2 downloaded: {region_data.shape}, {good_ratio*100:.1f}% good pixels")
                return region_data, transform

        except Exception as e:
            continue

    print(f"    No valid S2 data")
    return None, None


def download_sentinel1_rtc(
    lat: float,
    lon: float,
    year: int,
    region_size: int = 1000,
    buffer_km: float = 10.0,
) -> Tuple[Optional[np.ndarray], Optional[rasterio.transform.Affine]]:
    """
    Download Sentinel-1 RTC (Radiometric Terrain Corrected) data.
    Returns (VV, VH) polarization data.
    """
    print(f"  Searching Sentinel-1 RTC for lat={lat}, lon={lon}...")

    delta_deg = buffer_km / 111.0
    bbox = [lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg]

    api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(api_url, modifier=planetary_computer.sign_inplace)

    search = client.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        max_items=50,
        datetime=f"{year}-01-01/{year}-12-31",
    )

    items = list(search.items())

    if not items:
        print(f"    No Sentinel-1 RTC data found")
        return None, None

    print(f"    Found {len(items)} S1 items, trying best...")

    for item in items[:3]:
        try:
            assets = item.assets

            # Get VV and VH polarization bands
            bands_to_get = []
            if "vh" in assets:
                bands_to_get.append(("vh", "VH"))
            if "vv" in assets:
                bands_to_get.append(("vv", "VV"))

            if len(bands_to_get) == 0:
                continue

            bands_list = []
            transform = None

            for asset_key, band_name in bands_to_get:
                href = planetary_computer.sign(assets[asset_key].href)

                with rioxarray.open_rasterio(href) as da:
                    tile_crs = da.rio.crs

                    if tile_crs != "EPSG:4326":
                        bbox_in_tile_crs = transform_bounds(
                            "EPSG:4326", tile_crs, bbox[0], bbox[1], bbox[2], bbox[3]
                        )
                    else:
                        bbox_in_tile_crs = bbox

                    clipped = da.rio.clip_box(
                        minx=bbox_in_tile_crs[0], miny=bbox_in_tile_crs[1],
                        maxx=bbox_in_tile_crs[2], maxy=bbox_in_tile_crs[3],
                        auto_expand=True,
                    )

                    band_arr = clipped.values[0].astype(np.float32)

                    if transform is None:
                        transform = clipped.rio.transform()

                    if band_arr.shape != (region_size, region_size):
                        zoom_factors = (region_size / band_arr.shape[0], region_size / band_arr.shape[1])
                        band_arr = zoom(band_arr, zoom_factors, order=1)

                    # Convert to dB if in linear scale
                    band_arr = np.nan_to_num(band_arr, nan=0.0, posinf=0.0, neginf=-50.0)
                    if band_arr.max() > 10:  # Likely linear scale
                        band_arr = 10 * np.log10(np.clip(band_arr, 1e-10, None))

                    bands_list.append(band_arr)

            if len(bands_list) == 0:
                continue

            # Stack VH and VV
            region_data = np.stack(bands_list, axis=0)

            west, south, east, north = bbox
            transform = from_bounds(west, south, east, north, region_size, region_size)

            valid_pixels = np.isfinite(region_data).all(axis=0)
            good_ratio = valid_pixels.sum() / (region_size * region_size)

            if good_ratio > 0.5:
                print(f"    ✓ S1 RTC downloaded: {region_data.shape} ({len(bands_list)} bands), {good_ratio*100:.1f}% good pixels")
                return region_data, transform

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"    No valid S1 RTC data")
    return None, None


def download_copernicus_dem(
    lat: float,
    lon: float,
    buffer_km: float = 10.0,
    target_size: int = 1000
) -> Tuple[np.ndarray, Optional[rasterio.transform.Affine]]:
    """Download Copernicus DEM 30m."""
    print(f"  Downloading Copernicus DEM...")

    delta_deg = buffer_km / 111.0
    bbox = [lon - delta_deg, lat - delta_deg, lon + delta_deg, lat + delta_deg]

    api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(api_url, modifier=planetary_computer.sign_inplace)

    search = client.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
        max_items=10,
    )

    items = list(search.items())
    transform = None

    if not items:
        print("    No DEM found, using zeros")
        west, south, east, north = bbox
        transform = from_bounds(west, south, east, north, target_size, target_size)
        return np.zeros((target_size, target_size), dtype=np.float32), transform

    item = items[0]
    dem_href = planetary_computer.sign(item.assets["data"].href)

    try:
        with rioxarray.open_rasterio(dem_href) as dem_da:
            clipped = dem_da.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3],
                auto_expand=True,
            )
            dem_array = clipped.values[0]
            transform = clipped.rio.transform()
    except Exception as e:
        print(f"    Clipping failed: {e}, using center region")
        with rioxarray.open_rasterio(dem_href) as dem_da:
            cy, cx = dem_da.shape[1] // 2, dem_da.shape[2] // 2
            size = 500
            dem_array = dem_da.isel(
                y=slice(max(0, cy - size), min(dem_da.shape[1], cy + size)),
                x=slice(max(0, cx - size), min(dem_da.shape[2], cx + size))
            ).values[0]

    if dem_array.shape != (target_size, target_size):
        zoom_factors = (target_size / dem_array.shape[0], target_size / dem_array.shape[1])
        dem_array = zoom(dem_array, zoom_factors, order=1)

        # Recalculate transform for resampled size
        west, south, east, north = bbox
        transform = from_bounds(west, south, east, north, target_size, target_size)

    print(f"    ✓ DEM: {dem_array.shape}, range [{dem_array.min():.1f}, {dem_array.max():.1f}] m")
    return dem_array.astype(np.float32), transform


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_lulc_dem_from_s2(
    s2_region: np.ndarray,
    crop_size: int = 256,
    stride: int = 192,
    batch_size: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate LULC and DEM from Sentinel-2."""
    generator = init_s2_generator()
    if generator is None:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    print(f"  Generating LULC+DEM from S2 (device: {device})...")

    s2_region = np.nan_to_num(s2_region, nan=0.0, posinf=0.0, neginf=0.0)
    input_tensor = torch.from_numpy(s2_region).float().unsqueeze(0).to(device)

    # Generate LULC
    def model_forward_lulc(x):
        with torch.no_grad():
            return generator(x)['LULC']

    pred_lulc = tiled_inference(
        model_forward_lulc, input_tensor,
        crop=crop_size, stride=stride, batch_size=batch_size, verbose=False
    )
    lulc = pred_lulc.squeeze(0).argmax(dim=0).cpu().numpy()

    # Generate DEM
    def model_forward_dem(x):
        with torch.no_grad():
            return generator(x)['DEM']

    pred_dem = tiled_inference(
        model_forward_dem, input_tensor,
        crop=crop_size, stride=stride, batch_size=batch_size, verbose=False
    )
    dem = pred_dem.squeeze(0).squeeze(0).cpu().numpy()

    print(f"    ✓ Generated LULC: {lulc.shape}, classes: {len(np.unique(lulc))}")
    print(f"    ✓ Generated DEM: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}]")

    return lulc, dem


def generate_lulc_dem_from_s1(
    s1_region: np.ndarray,
    crop_size: int = 256,
    stride: int = 192,
    batch_size: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate LULC and DEM from Sentinel-1 RTC."""
    generator = init_s1_generator()
    if generator is None:
        print("    S1 generator not available")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    print(f"  Generating LULC+DEM from S1 RTC (device: {device})...")

    s1_region = np.nan_to_num(s1_region, nan=-50.0, posinf=0.0, neginf=-50.0)
    input_tensor = torch.from_numpy(s1_region).float().unsqueeze(0).to(device)

    # Generate LULC
    def model_forward_lulc(x):
        with torch.no_grad():
            return generator(x)['LULC']

    pred_lulc = tiled_inference(
        model_forward_lulc, input_tensor,
        crop=crop_size, stride=stride, batch_size=batch_size, verbose=False
    )
    lulc = pred_lulc.squeeze(0).argmax(dim=0).cpu().numpy()

    # Generate DEM
    def model_forward_dem(x):
        with torch.no_grad():
            return generator(x)['DEM']

    pred_dem = tiled_inference(
        model_forward_dem, input_tensor,
        crop=crop_size, stride=stride, batch_size=batch_size, verbose=False
    )
    dem = pred_dem.squeeze(0).squeeze(0).cpu().numpy()

    print(f"    ✓ Generated LULC: {lulc.shape}, classes: {len(np.unique(lulc))}")
    print(f"    ✓ Generated DEM: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}]")

    return lulc, dem


def generate_lulc_from_dem(
    dem_region: np.ndarray,
    crop_size: int = 256,
    stride: int = 192,
    batch_size: int = 4,
) -> Optional[np.ndarray]:
    """Generate LULC from DEM only."""
    generator = init_dem_generator()
    if generator is None:
        print("    DEM generator not available")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    print(f"  Generating LULC from DEM only (device: {device})...")

    dem_region = np.nan_to_num(dem_region, nan=0.0, posinf=10000.0, neginf=-1000.0)
    input_tensor = torch.from_numpy(dem_region).float().unsqueeze(0).unsqueeze(0).to(device)

    def model_forward_lulc(x):
        with torch.no_grad():
            return generator(x)['LULC']

    pred_lulc = tiled_inference(
        model_forward_lulc, input_tensor,
        crop=crop_size, stride=stride, batch_size=batch_size, verbose=False
    )
    lulc = pred_lulc.squeeze(0).argmax(dim=0).cpu().numpy()

    print(f"    ✓ Generated LULC: {lulc.shape}, classes: {len(np.unique(lulc))}")

    return lulc


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_volcano_multimodal(
    volcano: Dict[str, Any],
    region_size: int = 1000,
    buffer_km: float = 10.0,
    year: int = 2024,
) -> Dict[str, Any]:
    """
    Process a single volcano using SIMPLIFIED multi-modal strategy.

    NEW Strategy (No S2):
    1. Try S1 RTC → generate LULC + download DEM
    2. If no S1, use DEM → generate LULC only

    NO SENTINEL-2 IS USED (causes poor quality for these volcanoes)
    """
    name = volcano['name']
    lat = volcano['latitude']
    lon = volcano['longitude']

    print(f"\n{'='*70}")
    print(f"PROCESSING: {name}")
    print(f"Location: ({lat:.2f}, {lon:.2f})")
    print(f"Strategy: S1 RTC → DEM-only (NO S2)")
    print('='*70)

    result = {
        'name': name,
        'latitude': lat,
        'longitude': lon,
        'country': volcano['country'],
        'status': 'processing',
        'data_source': None,
    }

    dem_real = None
    lulc_ai = None
    dem_ai = None
    s1_region = None
    transform = None

    try:
        # STEP 1: Always download real DEM first
        print(f"\n[STEP 1] Downloading Real DEM (Copernicus)...")
        dem_real, dem_transform = download_copernicus_dem(lat, lon, buffer_km, region_size)
        transform = dem_transform

        # STEP 2: Try Sentinel-1 RTC
        print(f"\n[STEP 2] Trying Sentinel-1 RTC...")
        s1_region, s1_transform = download_sentinel1_rtc(lat, lon, year, region_size, buffer_km)

        if s1_region is not None:
            # VALIDATE S1 QUALITY
            s1_mean = s1_region.mean()
            s1_std = s1_region.std()
            s1_valid = (s1_mean > -40) and (s1_std > 2)

            print(f"  S1 Quality Check: mean={s1_mean:.1f}dB, std={s1_std:.1f}dB → {'PASS' if s1_valid else 'FAIL'}")

            if s1_valid:
                print(f"  ✓ Sentinel-1 RTC quality good! Using S1 → LULC")

                # Generate LULC from S1
                lulc_ai, dem_ai = generate_lulc_dem_from_s1(
                    s1_region,
                    crop_size=CONFIG['lulc_crop_size'],
                    stride=CONFIG['lulc_stride'],
                    batch_size=CONFIG['lulc_batch_size']
                )

                # VALIDATE GENERATION
                if lulc_ai is not None:
                    num_classes = len(np.unique(lulc_ai))
                    generation_valid = (num_classes >= 3)
                    print(f"  Generation Quality: classes={num_classes} → {'PASS' if generation_valid else 'FAIL'}")

                    if generation_valid:
                        result['data_source'] = 'Sentinel-1 RTC → LULC'
                        if s1_transform is not None:
                            transform = s1_transform
                        # Use real DEM (not AI-generated DEM from S1)
                        dem_ai = None
                    else:
                        print(f"  ✗ S1 generation poor quality, will try DEM-only...")
                        lulc_ai = None
            else:
                print(f"  ✗ S1 data poor quality, skipping to DEM-only...")
        else:
            print(f"  ✗ No Sentinel-1 RTC data found")

        # STEP 3: Final fallback - DEM only
        if lulc_ai is None:
            # Check if DEM is valid (not all zeros)
            dem_valid = dem_real is not None and dem_real.max() > 0

            if not dem_valid:
                print(f"\n[STEP 3] ✗ No valid DEM available, skipping volcano")
                print(f"  Reason: S1 RTC poor quality + DEM unavailable/invalid")
                result['status'] = 'skipped'
                result['error'] = 'No valid S1 RTC or DEM data available'
                return result

            print(f"\n[STEP 3] Using DEM-only → LULC")
            lulc_ai = generate_lulc_from_dem(
                dem_real,
                crop_size=CONFIG['lulc_crop_size'],
                stride=CONFIG['lulc_stride'],
                batch_size=CONFIG['lulc_batch_size']
            )

            if lulc_ai is not None:
                # Validate DEM-generated LULC
                num_classes = len(np.unique(lulc_ai))
                if num_classes < 2:
                    print(f"  ✗ DEM-generated LULC too poor (only {num_classes} classes), skipping")
                    result['status'] = 'skipped'
                    result['error'] = 'DEM-generated LULC insufficient quality'
                    return result

                result['data_source'] = 'DEM-only (Copernicus) → LULC'
                # No AI DEM generation, use real DEM
                dem_ai = None

        if lulc_ai is None:
            print(f"ERROR: Failed to generate LULC with any method")
            result['status'] = 'skipped'
            result['error'] = 'All data sources failed'
            return result

        # STEP 4: Save outputs
        print(f"\n[STEP 4] Saving outputs...")
        output_dir = f"{OUTPUT_BASE}/{name.replace(' ', '_')}"
        save_all_outputs(
            None,  # No S2
            s1_region,
            dem_real,
            dem_ai,
            lulc_ai,
            lat, lon, name, output_dir, transform, result['data_source']
        )

        result['status'] = 'success'
        result['output_dir'] = output_dir

        print(f"\n✓ Successfully processed {name} using {result['data_source']}")

    except Exception as e:
        print(f"\nERROR processing {name}: {e}")
        import traceback
        traceback.print_exc()
        result['status'] = 'skipped'
        result['error'] = str(e)
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user, skipping {name}")
        result['status'] = 'skipped'
        result['error'] = 'Interrupted by user'

    return result


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_all_outputs(
    s2_region, s1_region, dem_real, dem_ai, lulc_ai,
    lat, lon, name, output_dir, transform, data_source
):
    """Save all outputs including visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    buffer_km = CONFIG['buffer_km']

    # Save DEM real
    save_raster_as_geotiff(
        dem_real, lat, lon, buffer_km,
        f"{output_dir}/dem_copernicus_real.tif",
        transform=transform
    )

    # Save DEM AI (if different from real)
    if dem_ai is not None and not np.array_equal(dem_ai, dem_real):
        save_raster_as_geotiff(
            dem_ai, lat, lon, buffer_km,
            f"{output_dir}/dem_terramind_ai.tif",
            transform=transform
        )

    # Save LULC AI
    save_raster_as_geotiff(
        lulc_ai.astype(np.uint8), lat, lon, buffer_km,
        f"{output_dir}/lulc_terramind_ai.tif",
        transform=transform
    )

    # Save S2 if available
    if s2_region is not None:
        save_raster_as_geotiff(
            s2_region, lat, lon, buffer_km,
            f"{output_dir}/s2_rgb.tif",
            transform=transform
        )

    # Save S1 if available
    if s1_region is not None:
        save_raster_as_geotiff(
            s1_region, lat, lon, buffer_km,
            f"{output_dir}/s1_rtc.tif",
            transform=transform
        )

    # Create visualization
    create_comparison_figure(
        s2_region, s1_region, dem_real, dem_ai, lulc_ai,
        name, output_dir, data_source
    )

    print(f"  ✓ All outputs saved to {output_dir}")


def save_raster_as_geotiff(
    array: np.ndarray,
    lat: float,
    lon: float,
    buffer_km: float,
    output_path: str,
    transform: Optional[rasterio.transform.Affine] = None,
) -> None:
    """Save array as GeoTIFF."""
    if transform is None:
        delta_deg = buffer_km / 111.0
        west = lon - delta_deg
        east = lon + delta_deg
        south = lat - delta_deg
        north = lat + delta_deg

        if array.ndim == 2:
            height, width = array.shape
        else:
            _, height, width = array.shape

        transform = from_bounds(west, south, east, north, width, height)

    if array.ndim == 2:
        height, width = array.shape
        count = 1
        data = array.reshape(1, height, width)
    else:
        count, height, width = array.shape
        data = array

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def create_comparison_figure(
    s2_region, s1_region, dem_real, dem_ai, lulc_ai,
    name, output_dir, data_source
):
    """Create comparison figure showing all available data."""
    # Determine layout based on available data
    has_s2 = s2_region is not None
    has_s1 = s1_region is not None
    has_ai_dem = (dem_ai is not None) and not np.array_equal(dem_ai, dem_real)

    num_plots = 2 + (1 if has_s2 else 0) + (1 if has_s1 else 0) + (1 if has_ai_dem else 0)

    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 8*nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes

    plot_idx = 0

    # Plot S2 RGB if available
    if has_s2:
        rgb = s2_region[[3, 2, 1], :, :].transpose(1, 2, 0)
        rgb = np.clip(rgb / 2000, 0, 1)
        axes[plot_idx].imshow(rgb)
        axes[plot_idx].set_title('Sentinel-2 RGB', fontsize=14, fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Plot S1 if available
    if has_s1:
        if s1_region.shape[0] == 2:
            # VV+VH composite
            s1_display = np.stack([
                np.clip((s1_region[0] + 30) / 30, 0, 1),  # VV
                np.clip((s1_region[1] + 30) / 30, 0, 1),  # VH
                np.clip((s1_region[0] + 30) / 30, 0, 1),  # VV again
            ], axis=2)
        else:
            s1_display = np.clip((s1_region[0] + 30) / 30, 0, 1)

        axes[plot_idx].imshow(s1_display, cmap='gray' if s1_region.shape[0] == 1 else None)
        axes[plot_idx].set_title('Sentinel-1 RTC', fontsize=14, fontweight='bold')
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Plot Real DEM
    dem_plot = axes[plot_idx].imshow(dem_real, cmap='terrain')
    axes[plot_idx].set_title('Real DEM (Copernicus)', fontsize=14, fontweight='bold')
    axes[plot_idx].axis('off')
    plt.colorbar(dem_plot, ax=axes[plot_idx], label='Elevation (m)', fraction=0.046)
    plot_idx += 1

    # Plot AI DEM if available and different
    if has_ai_dem:
        dem_ai_plot = axes[plot_idx].imshow(dem_ai, cmap='terrain')
        axes[plot_idx].set_title('AI DEM (TerraMind)', fontsize=14, fontweight='bold')
        axes[plot_idx].axis('off')
        plt.colorbar(dem_ai_plot, ax=axes[plot_idx], label='Elevation (m)', fraction=0.046)
        plot_idx += 1

    # Plot LULC
    tm_cmap = mcolors.ListedColormap([
        '#000000', '#0000FF', '#228B22', '#90EE90', '#FFD700',
        '#FF0000', '#D2B48C', '#FFFFFF', '#808080', '#9ACD32',
    ])
    axes[plot_idx].imshow(lulc_ai, cmap=tm_cmap, vmin=0, vmax=9)
    axes[plot_idx].set_title(f'AI LULC (from {data_source})', fontsize=14, fontweight='bold')
    axes[plot_idx].axis('off')
    plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{name} - Multi-Modal Processing', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = f"{output_dir}/comparison_multimodal.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def get_processed_volcanoes(output_base=OUTPUT_BASE):
    """Get list of already processed volcanoes."""
    if not os.path.exists(output_base):
        return set()

    processed = set()
    for folder in os.listdir(output_base):
        folder_path = os.path.join(output_base, folder)
        if os.path.isdir(folder_path):
            volcano_name = folder.replace('_', ' ')

            # Check for comparison image
            if os.path.exists(os.path.join(folder_path, 'comparison_multimodal.png')):
                processed.add(volcano_name)

    return processed


def get_missing_volcanoes(csv_path=CSV_PATH, existing_base="validation_outputs"):
    """Get volcanoes that don't have data in the original validation_outputs."""
    df = pd.read_csv(csv_path)

    missing = []
    for _, row in df.iterrows():
        name = row['name']
        folder_name = name.replace(' ', '_')
        output_dir = os.path.join(existing_base, folder_name)

        # Check if it exists in original validation_outputs
        if not os.path.exists(output_dir):
            missing.append(row.to_dict())
        else:
            # Check if it has the required files
            required_files = ['dem_copernicus_real.tif', 'lulc_terramind_ai.tif']
            has_all = all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)
            if not has_all:
                missing.append(row.to_dict())

    return missing


def main():
    print("="*70)
    print("MULTI-MODAL VOLCANO PROCESSING")
    print("Strategy: S1 RTC → DEM-only (NO S2)")
    print("="*70)

    # Find missing volcanoes
    print(f"\n[1/4] Finding volcanoes without S2 data...")
    missing_volcanoes = get_missing_volcanoes()

    print(f"  Found {len(missing_volcanoes)} volcanoes without S2 data")

    # Check already processed
    print(f"\n[2/4] Checking for already processed volcanoes...")
    processed = get_processed_volcanoes()

    if processed:
        print(f"  Found {len(processed)} already processed")
        missing_volcanoes = [v for v in missing_volcanoes if v['name'] not in processed]

    print(f"\n  Remaining to process: {len(missing_volcanoes)}")

    if not missing_volcanoes:
        print("\n✓ All volcanoes processed!")
        return

    # Show first 10
    print(f"\n  Next volcanoes to process:")
    for i, v in enumerate(missing_volcanoes[:10], 1):
        print(f"    {i}. {v['name']} ({v['country']}) - {v['latitude']:.2f}, {v['longitude']:.2f}")
    if len(missing_volcanoes) > 10:
        print(f"    ... and {len(missing_volcanoes) - 10} more")

    # Process each volcano
    print(f"\n[3/4] Processing volcanoes...")

    all_results = []

    for i, volcano in enumerate(missing_volcanoes, 1):
        print(f"\n{'='*70}")
        print(f"VOLCANO {i}/{len(missing_volcanoes)}")
        print('='*70)

        try:
            result = process_volcano_multimodal(
                volcano,
                region_size=CONFIG['region_size'],
                buffer_km=CONFIG['buffer_km'],
                year=YEAR
            )
            all_results.append(result)

            # Save intermediate results after each volcano
            results_path = f"{OUTPUT_BASE}/processing_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)

        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user. Saving progress...")
            # Save what we have so far
            results_path = f"{OUTPUT_BASE}/processing_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            generate_summary(all_results)
            print(f"\n✓ Progress saved. You can resume by running the script again.")
            sys.exit(0)

    # Generate summary
    print(f"\n[4/4] Generating summary...")
    generate_summary(all_results)

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print('='*70)
    print(f"\nOutputs saved to: {OUTPUT_BASE}/")
    print(f"\nNext steps:")
    print(f"  1. Review comparison images in individual folders")
    print(f"  2. Run compute_all_risks.py with validation_base='{OUTPUT_BASE}'")


def generate_summary(results):
    """Generate processing summary."""
    df = pd.DataFrame(results)

    total = len(df)
    success = len(df[df['status'] == 'success'])
    skipped = len(df[df['status'] == 'skipped'])
    failed = len(df[df['status'] == 'failed'])

    print(f"\nProcessing Summary:")
    print(f"  Total: {total}")
    print(f"  Successful: {success}")
    print(f"  Skipped (no valid data): {skipped}")
    print(f"  Failed (errors): {failed}")

    if success > 0:
        success_df = df[df['status'] == 'success']

        print(f"\nData Sources Used:")
        for source in success_df['data_source'].unique():
            count = len(success_df[success_df['data_source'] == source])
            print(f"  {source}: {count}")

    if skipped > 0:
        print(f"\nSkipped Volcanoes:")
        skipped_df = df[df['status'] == 'skipped']
        for _, row in skipped_df.head(10).iterrows():
            print(f"  • {row['name']}: {row['error']}")
        if skipped > 10:
            print(f"  ... and {skipped - 10} more")

    # Save CSV
    csv_path = f"{OUTPUT_BASE}/processing_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")


if __name__ == "__main__":
    main()