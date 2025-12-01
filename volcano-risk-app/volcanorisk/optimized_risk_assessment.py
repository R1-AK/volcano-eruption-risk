"""
Optimized Advanced Risk Assessment - Uses Pre-computed DEM/LULC
Reads from validation_outputs folder instead of regenerating data.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import os
import rasterio
from scipy.ndimage import gaussian_filter, generic_filter
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Import your existing modules
from .config import CONFIG
from .terramind_engine import (
    init_models,
    extract_embeddings_from_region,
)

# Import the advanced assessment functions
from .advanced_risk_assessment import (
    calculate_terrain_hazard,
    model_volcanic_flow_paths,
    terramind_hazard_embedding_analysis,
    detect_high_risk_anomalies,
    calculate_population_exposure,
    assess_infrastructure_exposure,
    assess_environmental_vulnerability,
    assess_economic_impact,
    compute_comprehensive_risk_score,
)


# ============================================================================
# EARTH ENGINE INITIALIZATION WITH SERVICE ACCOUNT
# ============================================================================

def initialize_earth_engine_with_service_account(
        service_account: str = 'riska-mining@oceanic-depth-426609-d4.iam.gserviceaccount.com',
        key_file: str = 'D:/Imagery_DEM/Get_Image_DEM/oceanic-depth-426609-d4-96840cbbf840.json'
) -> bool:
    """
    Initialize Google Earth Engine with service account.

    Args:
        service_account: GEE service account email
        key_file: Path to JSON key file

    Returns:
        True if successful, False otherwise
    """
    try:
        import ee

        # Check if key file exists
        if not os.path.exists(key_file):
            print(f"WARNING: Key file not found: {key_file}")
            return False

        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)

        print(f"✓ Earth Engine initialized with service account")
        return True

    except Exception as e:
        print(f"WARNING: Earth Engine initialization failed: {e}")
        return False


# ============================================================================
# LOAD PRE-COMPUTED DATA FROM VALIDATION OUTPUTS
# ============================================================================

def load_precomputed_data(
        volcano_name: str,
        validation_base: str = "validation_outputs"
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load pre-computed DEM and LULC from validation_outputs folder.

    Args:
        volcano_name: Name of volcano
        validation_base: Base directory for validation outputs

    Returns:
        Dictionary with s2, dem, lulc arrays, or None if not found
    """
    # Convert volcano name to folder name (replace spaces with underscores)
    folder_name = volcano_name.replace(' ', '_')
    output_dir = os.path.join(validation_base, folder_name)

    if not os.path.exists(output_dir):
        print(f"  Pre-computed data not found for {volcano_name}")
        return None

    # Paths to raster files
    dem_real_path = os.path.join(output_dir, "dem_copernicus_real.tif")
    lulc_ai_path = os.path.join(output_dir, "lulc_terramind_ai.tif")
    s2_path = os.path.join(output_dir, "s2_rgb.tif")

    # Check if files exist
    if not all(os.path.exists(p) for p in [dem_real_path, lulc_ai_path, s2_path]):
        print(f"  Missing raster files for {volcano_name}")
        return None

    try:
        # Load DEM
        with rasterio.open(dem_real_path) as src:
            dem_array = src.read(1).astype(np.float32)
            transform = src.transform

        # Load LULC
        with rasterio.open(lulc_ai_path) as src:
            lulc_array = src.read(1).astype(np.uint8)

        # Load S2 (for embeddings)
        with rasterio.open(s2_path) as src:
            s2_array = src.read().astype(np.float32)

        print(f"  ✓ Loaded pre-computed data from {output_dir}")
        print(f"    - DEM: {dem_array.shape}, range [{dem_array.min():.1f}, {dem_array.max():.1f}] m")
        print(f"    - LULC: {lulc_array.shape}, classes: {len(np.unique(lulc_array))}")
        print(f"    - S2: {s2_array.shape}")

        return {
            's2_region': s2_array,
            'dem_region': dem_array,
            'lulc_region': lulc_array,
            'transform': transform
        }

    except Exception as e:
        print(f"  ERROR loading pre-computed data: {e}")
        return None


# ============================================================================
# OPTIMIZED POPULATION EXPOSURE WITH GEE SERVICE ACCOUNT
# ============================================================================

def calculate_population_exposure_gee(
        lat: float,
        lon: float,
        lulc_region: np.ndarray,
        hazard_zones: np.ndarray,
        buffer_km: float = 10.0,
        pixel_size_m: float = 10.0,
        use_gee: bool = True
) -> Dict[str, Any]:
    """
    Population exposure with GEE service account support.

    Args:
        lat, lon: Coordinates
        lulc_region: LULC array
        hazard_zones: Hazard zone array
        buffer_km: Buffer radius
        pixel_size_m: Pixel resolution
        use_gee: Try to use Earth Engine (if initialized)

    Returns:
        Dictionary with population exposure metrics
    """
    print("  Calculating population exposure...")

    total_population = 0
    population_density = 0.0
    data_source = 'LULC-estimated'

    if use_gee:
        try:
            import ee

            delta_deg = buffer_km / 111.0
            bbox = ee.Geometry.Rectangle([
                lon - delta_deg, lat - delta_deg,
                lon + delta_deg, lat + delta_deg
            ])

            # WorldPop 2020
            population = ee.ImageCollection('WorldPop/GP/100m/pop') \
                .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
                .mosaic() \
                .clip(bbox)

            stats = population.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=bbox,
                scale=100,
                maxPixels=1e9
            )

            total_population = int(stats.get('population').getInfo() or 0)
            area_km2 = buffer_km * buffer_km * np.pi
            population_density = total_population / area_km2
            data_source = 'WorldPop'

            print(f"    ✓ WorldPop: {total_population:,} people")

        except Exception as e:
            print(f"    WorldPop unavailable, using LULC estimate")
            use_gee = False

    if not use_gee or total_population == 0:
        # Fallback: Estimate from settlement areas
        settlement_pixels = (lulc_region == 5).sum()
        settlement_km2 = settlement_pixels * (pixel_size_m / 1000) ** 2
        # Conservative estimate: 3000 people per km² in settlement areas
        total_population = int(settlement_km2 * 3000)
        population_density = total_population / (buffer_km * buffer_km * np.pi)
        data_source = 'LULC-estimated'

    # Settlement area from LULC
    settlement_mask = (lulc_region == 5)
    settlement_area_km2 = float(settlement_mask.sum() * (pixel_size_m / 1000) ** 2)

    # Population by hazard zone
    pop_by_zone = {}
    for zone in [1, 2, 3, 4]:
        zone_mask = (hazard_zones == zone)
        zone_area_pct = zone_mask.sum() / hazard_zones.size * 100
        pop_by_zone[f'zone_{zone}'] = int(total_population * (zone_area_pct / 100))

    high_risk_population = pop_by_zone.get('zone_3', 0) + pop_by_zone.get('zone_4', 0)

    # Population in settlement areas within high-hazard zones
    settlement_in_hazard = settlement_mask & (hazard_zones >= 3)
    critical_settlement_km2 = float(settlement_in_hazard.sum() * (pixel_size_m / 1000) ** 2)

    # Exposure score
    if total_population == 0:
        exposure_score = 0.0
    else:
        pop_score = np.log10(max(total_population, 1)) / 7.0
        risk_proportion = high_risk_population / max(total_population, 1)
        exposure_score = float(np.clip(pop_score * 0.6 + risk_proportion * 0.4, 0, 1))

    return {
        'total_population': total_population,
        'population_density_per_km2': float(population_density),
        'settlement_area_km2': settlement_area_km2,
        'population_by_hazard_zone': pop_by_zone,
        'high_risk_population': high_risk_population,
        'critical_settlement_area_km2': critical_settlement_km2,
        'data_source': data_source,
        'exposure_score': exposure_score
    }


# ============================================================================
# OPTIMIZED FLOW PATHS WITH GEE SERVICE ACCOUNT
# ============================================================================

def model_volcanic_flow_paths_gee(
        dem_region: np.ndarray,
        lat: float,
        lon: float,
        buffer_km: float = 10.0,
        use_gee: bool = True
) -> Dict[str, Any]:
    """
    Flow path modeling with optional HydroSHEDS integration.

    Args:
        dem_region: DEM array
        lat, lon: Coordinates
        buffer_km: Buffer radius
        use_gee: Try to use Earth Engine

    Returns:
        Dictionary with flow path metrics
    """
    print("  Modeling volcanic flow paths...")

    # Calculate flow direction (D8)
    from .advanced_risk_assessment import (
        calculate_flow_direction_d8,
        calculate_flow_accumulation,
        identify_convergence_zones
    )

    flow_dir = calculate_flow_direction_d8(dem_region)
    flow_accum = calculate_flow_accumulation(flow_dir)

    # Identify major channels
    threshold = np.percentile(flow_accum, 95)
    major_channels = flow_accum > threshold
    channel_density = float(major_channels.sum() / major_channels.size)

    # Convergence zones
    convergence_zones = identify_convergence_zones(flow_dir)
    convergence_count = int(np.sum(convergence_zones))

    # Try to get HydroSHEDS rivers
    river_count = 0
    if use_gee:
        try:
            import ee

            delta_deg = buffer_km / 111.0
            bbox = ee.Geometry.Rectangle([
                lon - delta_deg, lat - delta_deg,
                lon + delta_deg, lat + delta_deg
            ])

            rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(bbox)
            river_count = rivers.size().getInfo()
            print(f"    ✓ HydroSHEDS: Found {river_count} rivers")

        except Exception as e:
            print(f"    HydroSHEDS unavailable")

    return {
        'flow_accumulation': flow_accum,
        'major_flow_paths': major_channels,
        'channel_density': channel_density,
        'convergence_zones': convergence_zones,
        'convergence_count': convergence_count,
        'hydrosheds_rivers': river_count,
        'flow_hazard_score': float(np.clip(channel_density * 10, 0, 1))
    }


# ============================================================================
# OPTIMIZED MAIN ASSESSMENT FUNCTION
# ============================================================================

def assess_volcano_comprehensive_optimized(
        lat: float,
        lon: float,
        volcano_name: str,
        year: int = 2024,
        buffer_km: float = 10.0,
        pixel_size_m: float = 10.0,
        region_gdp_per_capita: float = 10000,
        validation_base: str = "validation_outputs",
        use_gee: bool = True
) -> Dict[str, Any]:
    """
    OPTIMIZED: Uses pre-computed DEM/LULC from validation_outputs.

    Args:
        lat, lon: Volcano coordinates
        volcano_name: Name of volcano
        year: Year for assessment
        buffer_km: Buffer radius
        pixel_size_m: Pixel resolution
        region_gdp_per_capita: Regional GDP per capita
        validation_base: Base directory for pre-computed data
        use_gee: Use Earth Engine (if initialized)

    Returns:
        Dictionary with comprehensive risk assessment
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZED RISK ASSESSMENT: {volcano_name}")
    print(f"Location: ({lat:.4f}, {lon:.4f})")
    print(f"Using pre-computed data from: {validation_base}/")
    print('=' * 70)

    # Step 1: Load pre-computed data
    print("\n[1/8] Loading pre-computed DEM/LULC...")
    data = load_precomputed_data(volcano_name, validation_base)

    if data is None:
        return {
            'volcano_name': volcano_name,
            'latitude': lat,
            'longitude': lon,
            'error': 'Pre-computed data not found. Run export_dem_lulc.py first.',
            'status': 'FAILED'
        }

    s2_region = data['s2_region']
    dem_region = data['dem_region']
    lulc_region = data['lulc_region']
    transform = data['transform']

    # Step 2: Extract embeddings
    print("\n[2/8] Extracting TerraMind embeddings...")
    from .terramind_engine import init_models
    models = init_models(CONFIG)

    embeddings = extract_embeddings_from_region(
        s2_region, dem_region, lulc_region,
        tile_size=CONFIG.get('tile_size', 224)
    )

    # Step 3: Hazard assessment
    print("\n[3/8] Assessing terrain hazards...")
    terrain_metrics = calculate_terrain_hazard(dem_region, lat, lon, pixel_size_m)

    print("\n[4/8] Modeling flow paths...")
    flow_metrics = model_volcanic_flow_paths_gee(dem_region, lat, lon, buffer_km, use_gee)

    print("\n[5/8] Analyzing TerraMind embeddings...")
    terramind_hazard = terramind_hazard_embedding_analysis(
        embeddings, dem_region, lulc_region
    )
    ai_anomalies = detect_high_risk_anomalies(
        s2_region, dem_region, lulc_region, embeddings
    )
    terramind_metrics = {**terramind_hazard, **ai_anomalies}

    # Step 4: Exposure assessment
    print("\n[6/8] Calculating population exposure...")
    population_metrics = calculate_population_exposure_gee(
        lat, lon, lulc_region, terrain_metrics['hazard_zones'],
        buffer_km, pixel_size_m, use_gee
    )

    print("\n[7/8] Assessing infrastructure...")
    infrastructure_metrics = assess_infrastructure_exposure(
        lulc_region, terrain_metrics['hazard_zones'], pixel_size_m
    )

    # Step 5: Vulnerability assessment
    print("\n[8/8] Evaluating environmental & economic impact...")
    environmental_metrics = assess_environmental_vulnerability(
        lulc_region, terrain_metrics['hazard_zones'], pixel_size_m
    )

    economic_metrics = assess_economic_impact(
        lulc_region, population_metrics, terrain_metrics['hazard_zones'],
        pixel_size_m, region_gdp_per_capita
    )

    # Step 6: Compute comprehensive risk score
    print("\n[FINAL] Computing comprehensive risk score...")
    comprehensive_risk = compute_comprehensive_risk_score(
        terrain_metrics,
        flow_metrics,
        population_metrics,
        infrastructure_metrics,
        environmental_metrics,
        economic_metrics,
        terramind_metrics
    )

    # Compile final results
    result = {
        'volcano_name': volcano_name,
        'latitude': lat,
        'longitude': lon,
        'buffer_radius_km': buffer_km,
        'assessment_date': datetime.utcnow().isoformat(),
        'year': year,
        'status': 'SUCCESS',

        # Core risk metrics
        **comprehensive_risk,

        # Detailed component metrics
        'hazard_assessment': {
            'terrain': {k: v for k, v in terrain_metrics.items() if k != 'hazard_zones'},
            'flow_paths': {k: v for k, v in flow_metrics.items() if
                           k not in ['flow_accumulation', 'major_flow_paths', 'convergence_zones']},
            'ai_insights': terramind_metrics
        },
        'exposure_assessment': {
            'population': population_metrics,
            'infrastructure': infrastructure_metrics
        },
        'vulnerability_assessment': {
            'environmental': environmental_metrics,
            'economic': economic_metrics
        },

        # Methodology
        'methodology': 'TerraMind-Enhanced Multi-Criteria Risk Assessment v1.0 (Optimized)',
        'data_sources': [
            'Pre-computed Sentinel-2 L2A (from validation)',
            'Pre-computed Copernicus DEM 30m (from validation)',
            'Pre-computed TerraMind AI LULC (from validation)',
            population_metrics.get('data_source', 'LULC-estimated'),
            'HydroSHEDS (optional)' if flow_metrics['hydrosheds_rivers'] > 0 else 'HydroSHEDS (unavailable)'
        ],
        'model_version': {
            'terramind': CONFIG.get('terramind_model', 'terramind_v1_base_generate'),
            'framework': 'Advanced Risk Assessment v1.0 (Optimized)'
        }
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print("ASSESSMENT COMPLETE")
    print('=' * 70)
    print(f"Risk Score: {comprehensive_risk['composite_risk_score']:.1f}/100")
    print(f"Risk Category: {comprehensive_risk['risk_category']}")
    print(f"Predicted Fatalities: {comprehensive_risk['predicted_fatalities']:,}")
    print(f"Economic Loss: ${economic_metrics['total_estimated_loss_usd']:,.0f}")
    print(f"Population at Risk: {population_metrics['high_risk_population']:,}")
    print(f"Data Source: {population_metrics['data_source']}")
    print(f"Confidence: {comprehensive_risk['confidence_level']:.2f}")
    print('=' * 70)

    return result


# ============================================================================
# BATCH PROCESSING WITH OPTIMIZATION
# ============================================================================

def batch_assess_volcanoes_optimized(
        volcano_df: pd.DataFrame,
        name_col: str = "name",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        year: int = 2024,
        buffer_km: float = 10.0,
        validation_base: str = "validation_outputs",
        service_account: Optional[str] = None,
        key_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Optimized batch processing using pre-computed data.

    Args:
        volcano_df: DataFrame with volcano data
        name_col: Column name for volcano name
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        year: Year for analysis
        buffer_km: Buffer radius
        validation_base: Base directory for pre-computed data
        service_account: GEE service account email (optional)
        key_file: Path to GEE key file (optional)

    Returns:
        DataFrame with risk assessments
    """
    from .terramind_engine import init_models
    init_models(CONFIG)

    # Initialize Earth Engine if credentials provided
    use_gee = False
    if service_account and key_file:
        use_gee = initialize_earth_engine_with_service_account(service_account, key_file)

    results = []

    for idx, row in volcano_df.iterrows():
        name = str(row[name_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        print(f"\n{'=' * 70}")
        print(f"Processing {idx + 1}/{len(volcano_df)}: {name}")
        print('=' * 70)

        try:
            result = assess_volcano_comprehensive_optimized(
                lat=lat,
                lon=lon,
                volcano_name=name,
                year=year,
                buffer_km=buffer_km,
                validation_base=validation_base,
                use_gee=use_gee
            )
            results.append(result)

        except Exception as e:
            print(f"ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'volcano_name': name,
                'latitude': lat,
                'longitude': lon,
                'status': 'FAILED',
                'error': str(e)
            })

    return pd.DataFrame(results)