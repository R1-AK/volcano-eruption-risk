"""
Export HydroSHEDS Rivers from Google Earth Engine
Matches the same extent as DEM and LULC exports for each volcano
"""

import ee
import os
import json
import pandas as pd
from typing import Dict, List, Optional
import geojson


def initialize_earth_engine(
        service_account: str = 'riska-mining@oceanic-depth-426609-d4.iam.gserviceaccount.com',
        key_file: str = 'D:/Imagery_DEM/Get_Image_DEM/oceanic-depth-426609-d4-96840cbbf840.json'
) -> bool:
    """Initialize Google Earth Engine with service account."""
    try:
        if not os.path.exists(key_file):
            print(f"ERROR: Key file not found: {key_file}")
            return False

        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)
        print("✓ Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"ERROR: Earth Engine initialization failed: {e}")
        return False


def export_rivers_for_volcano(
        volcano_name: str,
        lat: float,
        lon: float,
        buffer_km: float = 10.0,
        output_dir: str = "public/rivers"
) -> Optional[str]:
    """
    Export HydroSHEDS rivers as GeoJSON for a specific volcano.

    Args:
        volcano_name: Name of volcano
        lat: Latitude
        lon: Longitude
        buffer_km: Buffer radius in km
        output_dir: Output directory for GeoJSON files

    Returns:
        Path to exported GeoJSON file, or None if failed
    """
    try:
        print(f"\nExporting rivers for: {volcano_name}")
        print(f"  Location: ({lat:.4f}, {lon:.4f})")
        print(f"  Buffer: {buffer_km} km")

        # Create bounding box
        delta_deg = buffer_km / 111.0
        bbox = ee.Geometry.Rectangle([
            lon - delta_deg, lat - delta_deg,
            lon + delta_deg, lat + delta_deg
        ])

        # Get HydroSHEDS Free Flowing Rivers
        rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(bbox)

        # Get HydroRIVERS (more detailed)
        # Note: HydroRIVERS is more comprehensive but larger dataset
        try:
            hydro_rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/HydroRIVERS/v1").filterBounds(bbox)
            river_count = hydro_rivers.size().getInfo()
            print(f"  Found {river_count} river segments (HydroRIVERS)")
        except:
            hydro_rivers = None
            print("  HydroRIVERS not available, using FreeFlowingRivers only")

        # Combine both datasets if available
        if hydro_rivers:
            all_rivers = rivers.merge(hydro_rivers)
        else:
            all_rivers = rivers

        # Get count
        total_count = all_rivers.size().getInfo()
        print(f"  Total river features: {total_count}")

        if total_count == 0:
            print("  No rivers found in this region")
            return None

        # Limit to reasonable size (prevent huge files)
        if total_count > 1000:
            print(f"  Limiting to 1000 features (from {total_count})")
            all_rivers = all_rivers.limit(1000)

        # Export as GeoJSON
        safe_name = volcano_name.replace(' ', '_').replace('/', '_')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_name}_rivers.geojson")

        # Get GeoJSON data
        print("  Downloading river data...")
        river_data = all_rivers.getInfo()

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(river_data, f)

        print(f"  ✓ Exported to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

        return output_path

    except Exception as e:
        print(f"  ERROR exporting rivers: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_osm_waterways(
        volcano_name: str,
        lat: float,
        lon: float,
        buffer_km: float = 10.0,
        output_dir: str = "public/rivers"
) -> Optional[str]:
    """
    Export OpenStreetMap waterways as alternative to HydroSHEDS.
    This uses the JRC Global Surface Water dataset in GEE.

    Args:
        volcano_name: Name of volcano
        lat: Latitude
        lon: Longitude
        buffer_km: Buffer radius in km
        output_dir: Output directory for GeoJSON files

    Returns:
        Path to exported GeoJSON file, or None if failed
    """
    try:
        print(f"\n  Trying JRC Global Surface Water as backup...")

        # Create bounding box
        delta_deg = buffer_km / 111.0
        bbox = ee.Geometry.Rectangle([
            lon - delta_deg, lat - delta_deg,
            lon + delta_deg, lat + delta_deg
        ])

        # Use JRC Global Surface Water for permanent water
        gsw = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
        occurrence = gsw.select('occurrence')

        # Get pixels with >50% water occurrence (permanent water)
        permanent_water = occurrence.gt(50)

        # Vectorize to get water bodies/rivers
        water_vectors = permanent_water.reduceToVectors(
            geometry=bbox,
            scale=30,  # 30m resolution
            geometryType='polygon',
            maxPixels=1e8
        )

        count = water_vectors.size().getInfo()
        print(f"  Found {count} water features (JRC Global Surface Water)")

        if count == 0:
            return None

        # Export
        safe_name = volcano_name.replace(' ', '_').replace('/', '_')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_name}_water.geojson")

        print("  Downloading water data...")
        water_data = water_vectors.limit(500).getInfo()

        with open(output_path, 'w') as f:
            json.dump(water_data, f)

        print(f"  ✓ Exported water bodies to: {output_path}")
        return output_path

    except Exception as e:
        print(f"  ERROR exporting water bodies: {e}")
        return None


def batch_export_rivers(
        volcano_csv: str = "volcanoes.csv",
        name_col: str = "name",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        buffer_km: float = 10.0,
        output_dir: str = "public/rivers",
        service_account: Optional[str] = None,
        key_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch export rivers for all volcanoes in CSV.

    Args:
        volcano_csv: Path to volcano CSV file
        name_col: Column name for volcano name
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        buffer_km: Buffer radius in km
        output_dir: Output directory for GeoJSON files
        service_account: GEE service account email
        key_file: Path to GEE key file

    Returns:
        DataFrame with export results
    """
    # Initialize Earth Engine
    if service_account and key_file:
        if not initialize_earth_engine(service_account, key_file):
            return pd.DataFrame()
    else:
        if not initialize_earth_engine():
            return pd.DataFrame()

    # Load volcanoes
    df = pd.read_csv(volcano_csv)
    print(f"\nLoaded {len(df)} volcanoes from {volcano_csv}")

    results = []

    for idx, row in df.iterrows():
        name = str(row[name_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        print(f"\n{'=' * 70}")
        print(f"Processing {idx + 1}/{len(df)}: {name}")
        print('=' * 70)

        # Try HydroSHEDS first
        river_path = export_rivers_for_volcano(
            name, lat, lon, buffer_km, output_dir
        )

        # If HydroSHEDS fails, try water bodies
        if not river_path:
            river_path = export_osm_waterways(
                name, lat, lon, buffer_km, output_dir
            )

        results.append({
            'volcano_name': name,
            'latitude': lat,
            'longitude': lon,
            'river_file': river_path if river_path else 'FAILED',
            'status': 'SUCCESS' if river_path else 'FAILED'
        })

    results_df = pd.DataFrame(results)

    # Save summary
    summary_path = os.path.join(output_dir, 'export_summary.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"Export complete! Summary saved to: {summary_path}")
    print(f"Success: {len(results_df[results_df.status == 'SUCCESS'])}/{len(results_df)}")
    print('=' * 70)

    return results_df


def create_river_index_json(
        output_dir: str = "public/rivers"
) -> str:
    """
    Create an index.json file listing all exported river GeoJSON files.
    This makes it easier to load rivers in the web viewer.

    Args:
        output_dir: Directory containing river GeoJSON files

    Returns:
        Path to index.json file
    """
    river_files = []

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.geojson'):
                volcano_name = filename.replace('_rivers.geojson', '').replace('_water.geojson', '').replace('_', ' ')
                river_files.append({
                    'volcano_name': volcano_name,
                    'filename': filename,
                    'path': f'/rivers/{filename}'
                })

    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'total_files': len(river_files),
            'rivers': river_files
        }, f, indent=2)

    print(f"\n✓ Created river index: {index_path}")
    print(f"  Total river files: {len(river_files)}")

    return index_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    SERVICE_ACCOUNT = 'riska-mining@oceanic-depth-426609-d4.iam.gserviceaccount.com'
    KEY_FILE = 'D:/Imagery_DEM/Get_Image_DEM/oceanic-depth-426609-d4-96840cbbf840.json'
    VOLCANO_CSV = 'data/volcano_catalog_clean.csv'
    OUTPUT_DIR = 'public/rivers'
    BUFFER_KM = 10.0

    print("=" * 70)
    print("HYDROSHEDS RIVER EXPORT FOR VOLCANO RISK DASHBOARD")
    print("=" * 70)

    # Batch export rivers
    results = batch_export_rivers(
        volcano_csv=VOLCANO_CSV,
        buffer_km=BUFFER_KM,
        output_dir=OUTPUT_DIR,
        service_account=SERVICE_ACCOUNT,
        key_file=KEY_FILE
    )

    # Create index file
    create_river_index_json(OUTPUT_DIR)

    print("\n✓ All done! Rivers exported to:", OUTPUT_DIR)
    print("  You can now load these in your web viewer.")