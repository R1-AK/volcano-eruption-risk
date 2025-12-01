"""
OPTIMIZED: Compute advanced risk assessment using PRE-COMPUTED DEM/LULC.
Reads from validation_outputs folder to avoid redundant downloads.
WITH RESUME CAPABILITY - skips already processed volcanoes.
"""

import pandas as pd
import json
import os
import math
from datetime import datetime
from volcanorisk.optimized_risk_assessment import (
    batch_assess_volcanoes_optimized,
    initialize_earth_engine_with_service_account
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
CSV_PATH = "data/volcano_catalog_clean.csv"
OUTPUT_JSON = "data/volcano_risk_complete.json"
VALIDATION_BASE = "validation_outputs"  # Folder with pre-computed DEM/LULC

# Processing parameters
NUM_VOLCANOES = None  # None for all, or set a number
BUFFER_KM = 10.0
YEAR = 2024

# Google Earth Engine credentials (OPTIONAL - for WorldPop and HydroSHEDS)
USE_GEE = True
SERVICE_ACCOUNT = 'riska-mining@oceanic-depth-426609-d4.iam.gserviceaccount.com'
KEY_FILE = 'D:/Imagery_DEM/Get_Image_DEM/oceanic-depth-426609-d4-96840cbbf840.json'


def safe_number(value):
    """Return None if value is NaN, otherwise the number."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_existing_results(json_path: str) -> dict:
    """Load previously computed results if they exist."""
    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create dict: volcano_name -> feature
        existing = {}
        for feature in data.get('features', []):
            name = feature['properties']['name']
            existing[name] = feature

        print(f"  Found {len(existing)} previously processed volcanoes")
        return existing

    except Exception as e:
        print(f"  Warning: Could not load existing results: {e}")
        return {}


def filter_volcanoes_with_data(volcano_df: pd.DataFrame, validation_base: str, existing_results: dict) -> pd.DataFrame:
    """Filter to only volcanoes with pre-computed data that haven't been processed yet."""
    valid_volcanoes = []

    for _, row in volcano_df.iterrows():
        name = row['name']

        # Skip if already processed
        if name in existing_results:
            continue

        folder_name = name.replace(' ', '_')
        output_dir = os.path.join(validation_base, folder_name)

        required_files = [
            "dem_copernicus_real.tif",
            "lulc_terramind_ai.tif",
            "s2_rgb.tif"
        ]

        if os.path.exists(output_dir):
            has_all = all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)
            if has_all:
                valid_volcanoes.append(row)

    filtered_df = pd.DataFrame(valid_volcanoes)
    return filtered_df


def build_geojson_feature(row: pd.Series, csv_lookup: dict) -> dict:
    """Build GeoJSON feature with all properties including last_eruption_year."""
    volcano_name = row['volcano_name']

    # Get CSV data for this volcano
    csv_data = csv_lookup.get(volcano_name, {})

    # Check status
    status = row.get('status', 'SUCCESS')
    has_error = (status == 'FAILED') or ('error' in row and row['error'])

    # Extract risk metrics
    risk_score = row.get('composite_risk_score', 0.0)
    risk_category = row.get('risk_category', 'UNKNOWN')
    predicted_fatalities = row.get('predicted_fatalities', 0)
    confidence = row.get('confidence_level', 0.0)

    # Extract population
    total_population = 0
    high_risk_population = 0
    if 'exposure_assessment' in row and isinstance(row['exposure_assessment'], dict):
        pop_data = row['exposure_assessment'].get('population', {})
        total_population = pop_data.get('total_population', 0)
        high_risk_population = pop_data.get('high_risk_population', 0)

    # Extract economic loss
    economic_loss = 0
    if 'vulnerability_assessment' in row and isinstance(row['vulnerability_assessment'], dict):
        econ_data = row['vulnerability_assessment'].get('economic', {})
        economic_loss = econ_data.get('total_estimated_loss_usd', 0)

    # Extract hazard metrics
    slope_mean = 0.0
    high_hazard_pct = 0.0
    if 'hazard_assessment' in row and isinstance(row['hazard_assessment'], dict):
        terrain = row['hazard_assessment'].get('terrain', {})
        slope_mean = terrain.get('slope_mean', 0.0)
        high_hazard_pct = terrain.get('combined_high_extreme_pct', 0.0)

    # Extract environmental metrics
    forest_at_risk = 0.0
    if 'vulnerability_assessment' in row and isinstance(row['vulnerability_assessment'], dict):
        env = row['vulnerability_assessment'].get('environmental', {})
        forest_at_risk = env.get('forest_at_risk_km2', 0.0)

    # Get last eruption year from CSV
    last_eruption_year = safe_number(csv_data.get('last_eruption_year_num'))
    if last_eruption_year is not None:
        last_eruption_year = int(float(last_eruption_year))

    # Build feature
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(row['longitude']), float(row['latitude'])]
        },
        "properties": {
            "id": csv_data.get('id', ''),
            "name": volcano_name,
            "country": csv_data.get('country', row.get('country', '')),
            "status": csv_data.get('status', ''),
            "last_eruption_year": last_eruption_year,

            # Risk metrics
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "predicted_fatalities": int(predicted_fatalities),
            "confidence_level": float(confidence),

            # Population
            "total_population": int(total_population),
            "high_risk_population": int(high_risk_population),

            # Economic
            "economic_loss_usd": float(economic_loss),

            # Hazard
            "slope_mean": float(slope_mean),
            "high_hazard_area_pct": float(high_hazard_pct),

            # Environmental
            "forest_at_risk_km2": float(forest_at_risk),

            # Metadata
            "assessment_date": row.get('assessment_date', datetime.utcnow().isoformat()),
            "assessment_year": int(YEAR),
            "buffer_radius_km": float(BUFFER_KM),
            "has_error": has_error,
            "error_message": row.get('error', '') if has_error else '',

            # Raster availability
            "has_dem_raster": True,
            "has_lulc_raster": True,
        }
    }

    return feature


def main():
    print("=" * 70)
    print("OPTIMIZED COMPREHENSIVE RISK ASSESSMENT")
    print("Using Pre-Computed DEM/LULC from validation_outputs/")
    print("WITH RESUME CAPABILITY")
    print("=" * 70)

    # Step 0: Load volcano catalog for metadata
    print(f"\n[0/6] Loading volcano catalog...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found!")
        return

    df_csv = pd.read_csv(CSV_PATH)

    # Create lookup dict: name -> row data
    csv_lookup = {}
    for _, row in df_csv.iterrows():
        csv_lookup[row['name']] = row.to_dict()

    print(f"  ✓ Loaded {len(df_csv)} volcanoes from catalog")

    # Step 1: Load existing results
    print(f"\n[1/6] Checking for existing results...")
    existing_results = load_existing_results(OUTPUT_JSON)

    if existing_results:
        print(f"  ✓ Found {len(existing_results)} previously processed volcanoes")
        print(f"  ✓ Will skip these and only process new ones")
    else:
        print(f"  ℹ️  No existing results found - will process all")

    # Step 2: Initialize Earth Engine (optional)
    if USE_GEE and SERVICE_ACCOUNT and KEY_FILE:
        print(f"\n[2/6] Initializing Google Earth Engine...")
        gee_success = initialize_earth_engine_with_service_account(SERVICE_ACCOUNT, KEY_FILE)

        if gee_success:
            print("  ✓ Will use WorldPop and HydroSHEDS data")
        else:
            print("  ⚠️  Will use LULC-based estimates")
    else:
        print(f"\n[2/6] Skipping Google Earth Engine (using LULC estimates)")
        gee_success = False

    # Step 3: Filter to unprocessed volcanoes with data
    print(f"\n[3/6] Filtering to new volcanoes with pre-computed data...")

    df = df_csv.copy()
    if NUM_VOLCANOES is not None:
        df = df.head(NUM_VOLCANOES)
        print(f"  Limiting to first {NUM_VOLCANOES} volcanoes total")

    df_filtered = filter_volcanoes_with_data(df, VALIDATION_BASE, existing_results)

    if len(df_filtered) == 0:
        print("\n✓ All volcanoes with available data have been processed!")
        print(f"  Total in results: {len(existing_results)}")
        print(f"\n💡 To process more volcanoes:")
        print(f"  1. Run export_dem_lulc.py for additional volcanoes")
        print(f"  2. Run this script again")
        return

    print(f"\n✓ Found {len(df_filtered)} new volcanoes to process")
    print(f"  Already processed: {len(existing_results)}")
    print(f"  New to process: {len(df_filtered)}")

    # Step 4: Run optimized batch assessment
    print(f"\n[4/6] Running risk assessments for new volcanoes...")
    print(f"  Mode: OPTIMIZED (using pre-computed DEM/LULC)")
    print(f"  Buffer: {BUFFER_KM} km")
    print(f"  Year: {YEAR}")
    print(f"  Earth Engine: {'ENABLED' if gee_success else 'DISABLED'}")

    results_df = batch_assess_volcanoes_optimized(
        volcano_df=df_filtered,
        name_col="name",
        lat_col="latitude",
        lon_col="longitude",
        year=YEAR,
        buffer_km=BUFFER_KM,
        validation_base=VALIDATION_BASE,
        service_account=SERVICE_ACCOUNT if USE_GEE else None,
        key_file=KEY_FILE if USE_GEE else None
    )

    # Step 5: Build GeoJSON features
    print(f"\n[5/6] Building GeoJSON...")

    new_features = []
    successful = 0
    failed = 0

    for _, row in results_df.iterrows():
        feature = build_geojson_feature(row, csv_lookup)

        if not feature['properties']['has_error']:
            successful += 1
        else:
            failed += 1

        new_features.append(feature)

    # Combine with existing results
    all_features = list(existing_results.values()) + new_features

    print(f"\n  New processed: {len(new_features)} ({successful} success, {failed} failed)")
    print(f"  Previous results: {len(existing_results)}")
    print(f"  Total features: {len(all_features)}")

    # Step 6: Create final GeoJSON
    print(f"\n[6/6] Saving final GeoJSON...")

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Global Volcano Risk Assessment (Optimized)",
            "generated_at": datetime.utcnow().isoformat(),
            "total_volcanoes": len(all_features),
            "successful_assessments": len(existing_results) + successful,
            "failed_assessments": failed,
            "assessment_year": YEAR,
            "buffer_radius_km": BUFFER_KM,
            "methodology": "TerraMind-Enhanced Multi-Criteria Risk Assessment (Optimized)",
            "model_version": "v1.0",
            "data_source": "Pre-computed DEM/LULC from validation_outputs",
            "earth_engine_enabled": gee_success
        },
        "features": all_features
    }

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)

    # Also save new results CSV
    csv_output = OUTPUT_JSON.replace('.json', '_latest.csv')
    results_df.to_csv(csv_output, index=False)

    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print('='*70)
    print(f"✓ Saved GeoJSON to: {OUTPUT_JSON}")
    print(f"✓ Saved latest batch CSV to: {csv_output}")
    print(f"\n📊 SUMMARY:")
    print(f"  Total volcanoes in JSON: {len(all_features)}")
    print(f"  Previously processed: {len(existing_results)}")
    print(f"  Newly processed: {len(new_features)} ({successful} success, {failed} failed)")

    if successful > 0:
        success_df = results_df[results_df['status'] == 'SUCCESS']

        print(f"\n  📈 NEW BATCH STATISTICS:")
        print(f"    Average Risk Score: {success_df['composite_risk_score'].mean():.1f}")
        print(f"    Highest Risk: {success_df['composite_risk_score'].max():.1f}")

        # Risk categories
        print(f"\n    Risk Distribution:")
        for cat in ['EXTREME', 'HIGH', 'MODERATE', 'LOW']:
            count = len(success_df[success_df['risk_category'] == cat])
            if count > 0:
                print(f"      {cat}: {count}")

        # Top 5 from new batch
        print(f"\n    Top 5 Highest Risk (from new batch):")
        top_5 = success_df.nlargest(5, 'composite_risk_score')
        for idx, row in top_5.iterrows():
            print(f"      {row['volcano_name']:30s} {row['composite_risk_score']:5.1f} ({row['risk_category']})")

    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Copy to Next.js:")
    print(f"     copy {OUTPUT_JSON} ..\\volcano-risk-web\\public\\data\\volcano_risk.json")
    print(f"  2. Restart Next.js dev server")
    print(f"  3. Open web app and explore {len(all_features)} volcanoes!")
    print(f"\n💡 TO CONTINUE PROCESSING:")
    print(f"  • Just run this script again - it will automatically skip processed volcanoes")
    print(f"  • To process more, first run export_dem_lulc.py for new volcanoes")
    print(f"  • Total available in catalog: {len(df_csv)}")
    print(f"  • Remaining to process: {len(df_csv) - len(all_features)}")


if __name__ == "__main__":
    main()
