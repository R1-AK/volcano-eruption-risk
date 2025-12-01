import os

# Core configuration for TerraMind volcano risk pipeline
CONFIG = {
    # Models
    "terramind_model": "terramind_v1_base_generate",
    "terramind_encoder": "terramind_v1_base",

    # UPDATED: 10 timesteps (validated, was 50)
    "generation_timesteps": 10,

    # Tile parameters
    "tile_size": 224,

    # NEW: Tiled inference for LULC generation
    "lulc_crop_size": 256,
    "lulc_stride": 192,
    "lulc_batch_size": 4,
    "default_region_size": 1000,

    # Spatial parameters
    "radius_km_default": 10.0,
    "max_tiles_per_volcano": 100,

    # Sentinel-2 data
    "sentinel_start_date": "2024-01-01",
    "sentinel_end_date": "2024-12-31",
    "max_cloud_cover": 10,  # NEW

    # Output
    "output_dir": os.path.join("data", "outputs"),

    # Population
    "worldpop_year": 2020,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)