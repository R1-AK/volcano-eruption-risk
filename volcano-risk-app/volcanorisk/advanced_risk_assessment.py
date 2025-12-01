"""
Advanced Volcano Risk Assessment Framework
Integrates TerraMind AI with comprehensive geospatial risk modeling.

Author: Geospatial Risk Analysis Team
Date: 2024
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, generic_filter
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Import your existing modules
from .config import CONFIG
from .terramind_engine import (
    download_sentinel2_region,
    download_copernicus_dem,
    download_copernicus_lulc,
    generate_lulc_from_s2_region,
    generate_dem_from_s2_region,
    analyze_location_with_terramind,
    extract_embeddings_from_region,
)


# ============================================================================
# 1. HAZARD ASSESSMENT - Terrain & Flow Dynamics
# ============================================================================

def calculate_terrain_hazard(
        dem_region: np.ndarray,
        lat: float,
        lon: float,
        pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Multi-scale slope analysis for volcanic hazards.

    Args:
        dem_region: Digital Elevation Model array (H, W) in meters
        lat, lon: Coordinates for context
        pixel_size_m: Pixel resolution in meters

    Returns:
        Dictionary with terrain hazard metrics
    """
    print("  Calculating terrain hazards...")

    # Calculate slope (degrees)
    dy, dx = np.gradient(dem_region)
    pixel_spacing = pixel_size_m  # meters per pixel
    slope = np.degrees(np.arctan(np.sqrt((dx / pixel_spacing) ** 2 + (dy / pixel_spacing) ** 2)))

    # Hazard zones based on volcanic hazard literature
    hazard_zones = np.zeros_like(slope, dtype=np.uint8)
    hazard_zones[slope > 30] = 4  # EXTREME: Rockfall/debris avalanche prone
    hazard_zones[(slope > 15) & (slope <= 30)] = 3  # HIGH: Lahar channels
    hazard_zones[(slope > 5) & (slope <= 15)] = 2  # MODERATE: Flow accumulation
    hazard_zones[slope <= 5] = 1  # LOW: Depositional zones

    # Topographic roughness (terrain complexity indicator)
    roughness = gaussian_filter(slope, sigma=3) - gaussian_filter(slope, sigma=10)
    roughness_std = float(np.std(roughness))

    # Terrain ruggedness index (TRI)
    tri = calculate_tri(dem_region)

    # Calculate area statistics
    total_pixels = hazard_zones.size
    extreme_hazard_pct = float((hazard_zones == 4).sum() / total_pixels * 100)
    high_hazard_pct = float((hazard_zones == 3).sum() / total_pixels * 100)
    moderate_hazard_pct = float((hazard_zones == 2).sum() / total_pixels * 100)
    low_hazard_pct = float((hazard_zones == 1).sum() / total_pixels * 100)

    combined_high_extreme = extreme_hazard_pct + high_hazard_pct

    return {
        'slope_mean': float(np.mean(slope)),
        'slope_max': float(np.max(slope)),
        'slope_std': float(np.std(slope)),
        'roughness_index': roughness_std,
        'terrain_ruggedness_index': float(tri),
        'hazard_zones': hazard_zones,
        'extreme_hazard_area_pct': extreme_hazard_pct,
        'high_hazard_area_pct': high_hazard_pct,
        'moderate_hazard_area_pct': moderate_hazard_pct,
        'low_hazard_area_pct': low_hazard_pct,
        'combined_high_extreme_pct': combined_high_extreme,
        'terrain_hazard_score': calculate_terrain_score(slope, roughness_std, tri)
    }


def calculate_tri(dem_region: np.ndarray) -> float:
    """Calculate Terrain Ruggedness Index."""

    def tri_function(values):
        center = values[len(values) // 2]
        return np.sqrt(np.mean((values - center) ** 2))

    tri_array = generic_filter(dem_region, tri_function, size=3, mode='reflect')
    return float(np.mean(tri_array))


def calculate_terrain_score(slope: np.ndarray, roughness: float, tri: float) -> float:
    """Composite terrain hazard score (0-1)."""
    slope_norm = np.clip(np.mean(slope) / 45.0, 0, 1)  # Normalize by 45°
    roughness_norm = np.clip(roughness / 10.0, 0, 1)
    tri_norm = np.clip(tri / 50.0, 0, 1)

    score = (slope_norm * 0.5 + roughness_norm * 0.3 + tri_norm * 0.2)
    return float(score)


def model_volcanic_flow_paths(
        dem_region: np.ndarray,
        lat: float,
        lon: float,
        buffer_km: float = 10.0
) -> Dict[str, Any]:
    """
    Simulate lahar/pyroclastic flow paths using flow accumulation.

    Args:
        dem_region: Digital Elevation Model array
        lat, lon: Coordinates
        buffer_km: Buffer radius

    Returns:
        Dictionary with flow path metrics
    """
    print("  Modeling volcanic flow paths...")

    # Flow direction (D8 algorithm - simplified version)
    flow_dir = calculate_flow_direction_d8(dem_region)

    # Flow accumulation
    flow_accum = calculate_flow_accumulation(flow_dir)

    # Identify major drainage channels (high flow accumulation)
    threshold = np.percentile(flow_accum, 95)
    major_channels = flow_accum > threshold

    # Channel density
    channel_density = float(major_channels.sum() / major_channels.size)

    # Identify convergence zones (where multiple flows meet)
    convergence_zones = identify_convergence_zones(flow_dir)
    convergence_count = int(np.sum(convergence_zones))

    # Optional: Try to connect to HydroSHEDS (requires Earth Engine)
    river_count = 0
    try:
        import ee
        ee.Initialize()

        delta_deg = buffer_km / 111.0
        bbox = ee.Geometry.Rectangle([
            lon - delta_deg, lat - delta_deg,
            lon + delta_deg, lat + delta_deg
        ])

        rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(bbox)
        river_count = rivers.size().getInfo()
        print(f"    Found {river_count} HydroSHEDS rivers in region")
    except Exception as e:
        print(f"    Note: HydroSHEDS data unavailable (Earth Engine not initialized)")

    return {
        'flow_accumulation': flow_accum,
        'major_flow_paths': major_channels,
        'channel_density': channel_density,
        'convergence_zones': convergence_zones,
        'convergence_count': convergence_count,
        'hydrosheds_rivers': river_count,
        'flow_hazard_score': float(np.clip(channel_density * 10, 0, 1))
    }


def calculate_flow_direction_d8(dem: np.ndarray) -> np.ndarray:
    """Calculate D8 flow direction."""
    padded = np.pad(dem, 1, mode='edge')
    directions = np.zeros_like(dem, dtype=np.uint8)

    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            center = padded[i, j]
            neighbors = [
                padded[i - 1, j], padded[i - 1, j + 1], padded[i, j + 1],
                padded[i + 1, j + 1], padded[i + 1, j], padded[i + 1, j - 1],
                padded[i, j - 1], padded[i - 1, j - 1]
            ]
            # Find steepest descent
            slopes = [(center - n) for n in neighbors]
            directions[i - 1, j - 1] = np.argmax(slopes) if max(slopes) > 0 else 0

    return directions


def calculate_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """Calculate flow accumulation (simplified version)."""
    # Initialize with ones (each cell contributes 1)
    flow_accum = np.ones_like(flow_dir, dtype=np.float32)

    # Simple accumulation (real implementation would need topological sorting)
    # This is a placeholder - for production use a proper flow accumulation algorithm
    flow_accum = gaussian_filter(flow_accum, sigma=5)

    return flow_accum


def identify_convergence_zones(flow_dir: np.ndarray) -> np.ndarray:
    """Identify where multiple flows converge (highest risk zones)."""

    # Count number of unique flow directions in neighborhood
    def count_inflows(values):
        return len(np.unique(values))

    convergence = generic_filter(flow_dir, count_inflows, size=5, mode='reflect')
    high_convergence = convergence > np.percentile(convergence, 90)

    return high_convergence


# ============================================================================
# 2. TERRAMIND AI-ENHANCED HAZARD DETECTION
# ============================================================================

def terramind_hazard_embedding_analysis(
        embeddings: np.ndarray,
        dem_region: np.ndarray,
        lulc_region: np.ndarray
) -> Dict[str, Any]:
    """
    Use TerraMind embeddings to identify HIGH-RISK patterns.

    Embeddings capture subtle terrain-vegetation-water interactions
    that traditional methods miss.

    Args:
        embeddings: TerraMind feature embeddings (N, D)
        dem_region: DEM array
        lulc_region: LULC array

    Returns:
        Dictionary with AI-derived hazard insights
    """
    print("  Analyzing TerraMind embeddings for hazard patterns...")

    if embeddings.size == 0 or len(embeddings) == 0:
        return {
            'terrain_complexity_score': 0.0,
            'anomalous_terrain_clusters': 0,
            'outlier_zones_count': 0,
            'risk_flag': 'UNKNOWN',
            'embedding_variance': 0.0
        }

    # Cluster embeddings to find anomalous terrain patterns
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # DBSCAN clustering to identify unusual patterns
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings_scaled)
    labels = clustering.labels_

    # Count clusters and outliers
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    outliers = int((labels == -1).sum())

    # Calculate embedding variance (high variance = complex/unstable terrain)
    embedding_variance = float(np.var(embeddings, axis=0).mean())

    # Determine risk level based on patterns
    if embedding_variance > 0.8 or outliers > len(embeddings) * 0.15:
        risk_flag = 'HIGH'
    elif embedding_variance > 0.5 or outliers > len(embeddings) * 0.08:
        risk_flag = 'MODERATE'
    else:
        risk_flag = 'LOW'

    return {
        'terrain_complexity_score': embedding_variance,
        'anomalous_terrain_clusters': n_clusters,
        'outlier_zones_count': outliers,
        'outlier_percentage': float(outliers / max(len(embeddings), 1) * 100),
        'risk_flag': risk_flag,
        'embedding_variance': embedding_variance
    }


def detect_high_risk_anomalies(
        s2_region: np.ndarray,
        dem_region: np.ndarray,
        lulc_region: np.ndarray,
        embeddings: np.ndarray,
        tile_size: int = 224
) -> Dict[str, Any]:
    """
    Use TerraMind embeddings to find unusual terrain-vegetation combinations
    that indicate hidden risks (e.g., unstable slopes with dense forest).

    Args:
        s2_region: Sentinel-2 data (12, H, W)
        dem_region: DEM array (H, W)
        lulc_region: LULC array (H, W)
        embeddings: TerraMind embeddings (N, D)
        tile_size: Size of tiles used for embeddings

    Returns:
        Dictionary with anomaly detection results
    """
    print("  Detecting terrain anomalies with Isolation Forest...")

    if embeddings.size == 0 or len(embeddings) == 0:
        return {
            'anomaly_percentage': 0.0,
            'high_risk_anomaly_tiles': 0,
            'anomaly_cluster_flag': False,
            'interpretation': 'Insufficient data for anomaly detection'
        }

    # Anomaly detection using Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(embeddings)

    anomaly_count = int((anomalies == -1).sum())
    anomaly_pct = float(anomaly_count / len(anomalies) * 100)

    # Spatial distribution of anomalies
    n_tiles_x = s2_region.shape[2] // tile_size
    n_tiles_y = s2_region.shape[1] // tile_size
    expected_tiles = n_tiles_x * n_tiles_y

    # Reshape anomalies to spatial grid (if dimensions match)
    if len(anomalies) == expected_tiles:
        anomaly_map = anomalies.reshape((n_tiles_y, n_tiles_x))

        # Calculate slope statistics per tile
        slope = calculate_terrain_hazard(dem_region, 0, 0)['slope_mean']

        # Count anomalies in high-slope areas (simplified)
        high_risk_tiles = anomaly_count  # Simplified version
    else:
        anomaly_map = None
        high_risk_tiles = 0

    # Determine if anomalies are clustered (concerning)
    anomaly_cluster_flag = anomaly_count > (len(embeddings) * 0.15)

    interpretation = "Normal terrain variability"
    if anomaly_cluster_flag and anomaly_pct > 15:
        interpretation = "Clustered anomalies suggest unstable terrain or data quality issues"
    elif anomaly_pct > 20:
        interpretation = "High anomaly rate indicates complex terrain requiring detailed assessment"

    return {
        'anomaly_percentage': anomaly_pct,
        'high_risk_anomaly_tiles': high_risk_tiles,
        'anomaly_cluster_flag': anomaly_cluster_flag,
        'total_anomaly_tiles': anomaly_count,
        'interpretation': interpretation
    }


# ============================================================================
# 3. EXPOSURE ASSESSMENT - Population & Infrastructure
# ============================================================================

def calculate_population_exposure(
        lat: float,
        lon: float,
        lulc_region: np.ndarray,
        hazard_zones: np.ndarray,
        buffer_km: float = 10.0,
        pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Multi-zone population exposure with vulnerability factors.

    Args:
        lat, lon: Coordinates
        lulc_region: LULC array
        hazard_zones: Hazard zone array (1=low, 4=extreme)
        buffer_km: Buffer radius
        pixel_size_m: Pixel resolution

    Returns:
        Dictionary with population exposure metrics
    """
    print("  Calculating population exposure...")

    # Try to get WorldPop data via Earth Engine
    total_population = 0
    population_density = 0.0
    ee_available = False

    try:
        import ee
        ee.Initialize()
        ee_available = True

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

        print(f"    WorldPop data: {total_population:,} people")

    except Exception as e:
        print(f"    Note: WorldPop unavailable (using LULC-based estimate)")
        # Fallback: Estimate from settlement areas in LULC
        settlement_pixels = (lulc_region == 5).sum()
        settlement_km2 = settlement_pixels * (pixel_size_m / 1000) ** 2
        # Rough estimate: 5000 people per km² in settlement areas
        total_population = int(settlement_km2 * 5000)
        population_density = total_population / (buffer_km * buffer_km * np.pi)

    # Settlement area from LULC (class 5 = Built-up in TerraMind)
    settlement_mask = (lulc_region == 5)
    settlement_area_km2 = float(settlement_mask.sum() * (pixel_size_m / 1000) ** 2)

    # Population by hazard zone (proportional allocation)
    pop_by_zone = {}
    for zone in [1, 2, 3, 4]:
        zone_mask = (hazard_zones == zone)
        zone_area_pct = zone_mask.sum() / hazard_zones.size * 100
        pop_by_zone[f'zone_{zone}'] = int(total_population * (zone_area_pct / 100))

    high_risk_population = pop_by_zone.get('zone_3', 0) + pop_by_zone.get('zone_4', 0)

    # Population in settlement areas within high-hazard zones
    settlement_in_hazard = settlement_mask & (hazard_zones >= 3)
    critical_settlement_km2 = float(settlement_in_hazard.sum() * (pixel_size_m / 1000) ** 2)

    return {
        'total_population': total_population,
        'population_density_per_km2': float(population_density),
        'settlement_area_km2': settlement_area_km2,
        'population_by_hazard_zone': pop_by_zone,
        'high_risk_population': high_risk_population,
        'critical_settlement_area_km2': critical_settlement_km2,
        'data_source': 'WorldPop' if ee_available else 'LULC-estimated',
        'exposure_score': calculate_exposure_score(total_population, high_risk_population)
    }


def calculate_exposure_score(total_pop: int, high_risk_pop: int) -> float:
    """Calculate normalized exposure score (0-1)."""
    if total_pop == 0:
        return 0.0

    # Log scale for population (handles wide range)
    pop_score = np.log10(max(total_pop, 1)) / 7.0  # Normalize to ~0-1

    # Proportion at high risk
    risk_proportion = high_risk_pop / max(total_pop, 1)

    score = (pop_score * 0.6 + risk_proportion * 0.4)
    return float(np.clip(score, 0, 1))


def assess_infrastructure_exposure(
        lulc_region: np.ndarray,
        hazard_zones: np.ndarray,
        pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Quantify built environment at risk using TerraMind LULC.

    Args:
        lulc_region: LULC array
        hazard_zones: Hazard zone array
        pixel_size_m: Pixel resolution

    Returns:
        Dictionary with infrastructure exposure metrics
    """
    print("  Assessing infrastructure exposure...")

    # Built-up areas (class 5 in TerraMind)
    built_up = (lulc_region == 5)

    # Cross with hazard zones
    built_in_extreme_hazard = built_up & (hazard_zones == 4)
    built_in_high_hazard = built_up & (hazard_zones == 3)
    built_in_moderate_hazard = built_up & (hazard_zones == 2)

    total_built_pixels = built_up.sum()
    at_risk_built = (built_in_extreme_hazard.sum() + built_in_high_hazard.sum())

    # Convert to km²
    km2_per_pixel = (pixel_size_m / 1000) ** 2
    total_built_km2 = float(total_built_pixels * km2_per_pixel)
    at_risk_km2 = float(at_risk_built * km2_per_pixel)

    # Calculate percentages
    if total_built_pixels > 0:
        built_at_risk_pct = float(at_risk_built / total_built_pixels * 100)
        extreme_pct = float(built_in_extreme_hazard.sum() / total_built_pixels * 100)
        high_pct = float(built_in_high_hazard.sum() / total_built_pixels * 100)
    else:
        built_at_risk_pct = 0.0
        extreme_pct = 0.0
        high_pct = 0.0

    # Flag critical infrastructure risk
    critical_flag = at_risk_built > (total_built_pixels * 0.3)

    return {
        'total_built_area_km2': total_built_km2,
        'built_at_risk_km2': at_risk_km2,
        'built_at_risk_pct': built_at_risk_pct,
        'built_in_extreme_hazard_pct': extreme_pct,
        'built_in_high_hazard_pct': high_pct,
        'critical_infrastructure_flag': critical_flag,
        'infrastructure_score': float(np.clip(built_at_risk_pct / 100, 0, 1))
    }


# ============================================================================
# 4. VULNERABILITY ASSESSMENT - Environmental & Economic
# ============================================================================

def assess_environmental_vulnerability(
        lulc_region: np.ndarray,
        hazard_zones: np.ndarray,
        pixel_size_m: float = 10.0
) -> Dict[str, Any]:
    """
    Quantify ecological assets at risk.

    TerraMind LULC Classes:
        0: No data, 1: Water, 2: Trees, 3: Flooded veg, 4: Crops,
        5: Built, 6: Bare, 7: Snow/ice, 8: Clouds, 9: Rangeland

    Args:
        lulc_region: LULC array
        hazard_zones: Hazard zone array
        pixel_size_m: Pixel resolution

    Returns:
        Dictionary with environmental vulnerability metrics
    """
    print("  Assessing environmental vulnerability...")

    km2_per_pixel = (pixel_size_m / 1000) ** 2

    # Forest/Trees (class 2)
    forest = (lulc_region == 2)
    forest_area_km2 = float(forest.sum() * km2_per_pixel)

    # Agricultural land (class 4)
    agriculture = (lulc_region == 4)
    ag_area_km2 = float(agriculture.sum() * km2_per_pixel)

    # Wetlands/Water bodies (classes 1, 3)
    water_wetland = np.isin(lulc_region, [1, 3])
    water_area_km2 = float(water_wetland.sum() * km2_per_pixel)

    # Rangeland (class 9)
    rangeland = (lulc_region == 9)
    rangeland_area_km2 = float(rangeland.sum() * km2_per_pixel)

    # Calculate exposure by hazard zone
    forest_at_risk = forest & (hazard_zones >= 3)
    ag_at_risk = agriculture & (hazard_zones >= 3)
    water_at_risk = water_wetland & (hazard_zones >= 3)

    forest_at_risk_km2 = float(forest_at_risk.sum() * km2_per_pixel)
    ag_at_risk_km2 = float(ag_at_risk.sum() * km2_per_pixel)
    water_at_risk_km2 = float(water_at_risk.sum() * km2_per_pixel)

    # Carbon stock estimate (rough approximation)
    # Tropical forests: ~200 tons C/ha, Temperate: ~100
    # Using conservative estimate of 150 tons C/ha
    carbon_at_risk_tons = forest_at_risk_km2 * 100 * 150  # ha * tons/ha

    # Biodiversity proxy (diverse land cover = higher biodiversity)
    unique_classes = len(np.unique(lulc_region))
    biodiversity_index = float(unique_classes / 10.0)  # Normalize by max expected classes

    # Calculate environmental risk score
    env_score = calculate_env_risk_score(
        forest_at_risk, ag_at_risk, water_wetland, lulc_region.size
    )

    return {
        'forest_area_km2': forest_area_km2,
        'forest_at_risk_km2': forest_at_risk_km2,
        'forest_at_risk_pct': float(forest_at_risk_km2 / max(forest_area_km2, 0.001) * 100),
        'agricultural_area_km2': ag_area_km2,
        'agriculture_at_risk_km2': ag_at_risk_km2,
        'agriculture_at_risk_pct': float(ag_at_risk_km2 / max(ag_area_km2, 0.001) * 100),
        'water_wetland_area_km2': water_area_km2,
        'water_at_risk_km2': water_at_risk_km2,
        'rangeland_area_km2': rangeland_area_km2,
        'estimated_carbon_at_risk_tons': float(carbon_at_risk_tons),
        'biodiversity_index': biodiversity_index,
        'environmental_risk_score': env_score
    }


def calculate_env_risk_score(
        forest_at_risk: np.ndarray,
        ag_at_risk: np.ndarray,
        water_wetland: np.ndarray,
        total_pixels: int
) -> float:
    """Composite environmental risk score (0-1)."""
    forest_weight = 0.4
    ag_weight = 0.3
    water_weight = 0.3

    score = (
            forest_weight * (forest_at_risk.sum() / total_pixels) +
            ag_weight * (ag_at_risk.sum() / total_pixels) +
            water_weight * (water_wetland.sum() / total_pixels)
    )

    return float(np.clip(score * 10, 0, 1))  # Scale to 0-1


def assess_economic_impact(
        lulc_region: np.ndarray,
        population_metrics: Dict[str, Any],
        hazard_zones: np.ndarray,
        pixel_size_m: float = 10.0,
        region_gdp_per_capita: float = 10000  # USD, adjust by region
) -> Dict[str, Any]:
    """
    Estimate economic losses using land use and population data.

    Args:
        lulc_region: LULC array
        population_metrics: Population metrics dictionary
        hazard_zones: Hazard zone array
        pixel_size_m: Pixel resolution
        region_gdp_per_capita: Regional GDP per capita in USD

    Returns:
        Dictionary with economic impact estimates
    """
    print("  Estimating economic impact...")

    km2_per_pixel = (pixel_size_m / 1000) ** 2

    # Agricultural productivity loss
    ag_at_risk = ((lulc_region == 4) & (hazard_zones >= 3))
    ag_area_at_risk_km2 = ag_at_risk.sum() * km2_per_pixel

    # Agricultural value varies by region, using global average
    avg_ag_value_per_km2 = 50000  # USD per km² per year
    ag_loss = ag_area_at_risk_km2 * avg_ag_value_per_km2

    # Built environment loss
    built_at_risk = ((lulc_region == 5) & (hazard_zones >= 3))
    built_area_at_risk_km2 = built_at_risk.sum() * km2_per_pixel

    # Urban area value (buildings, infrastructure)
    avg_built_value_per_km2 = 5000000  # USD per km²
    built_loss = built_area_at_risk_km2 * avg_built_value_per_km2

    # Economic disruption cost (lost productivity)
    pop_at_risk = population_metrics.get('high_risk_population', 0)
    disruption_cost_per_capita = region_gdp_per_capita * 2  # 2 years of GDP as disruption
    disruption_loss = pop_at_risk * disruption_cost_per_capita

    # Forest/ecosystem services loss
    forest_at_risk = ((lulc_region == 2) & (hazard_zones >= 3))
    forest_area_at_risk_km2 = forest_at_risk.sum() * km2_per_pixel
    ecosystem_services_value = 5000  # USD per km² per year
    ecosystem_loss = forest_area_at_risk_km2 * ecosystem_services_value * 10  # 10-year value

    total_loss = ag_loss + built_loss + disruption_loss + ecosystem_loss

    # Loss category
    loss_category = categorize_loss(total_loss)

    return {
        'agricultural_loss_usd': float(ag_loss),
        'built_environment_loss_usd': float(built_loss),
        'infrastructure_disruption_usd': float(disruption_loss),
        'ecosystem_services_loss_usd': float(ecosystem_loss),
        'total_estimated_loss_usd': float(total_loss),
        'loss_category': loss_category,
        'economic_risk_score': float(np.clip(np.log10(max(total_loss, 1)) / 10, 0, 1))
    }


def categorize_loss(loss: float) -> str:
    """Categorize economic loss."""
    if loss > 1e9:
        return 'CATASTROPHIC'
    elif loss > 1e8:
        return 'SEVERE'
    elif loss > 1e7:
        return 'HIGH'
    elif loss > 1e6:
        return 'MODERATE'
    else:
        return 'LOW'


# ============================================================================
# 5. COMPOSITE RISK SCORING
# ============================================================================

def compute_comprehensive_risk_score(
        terrain_metrics: Dict[str, Any],
        flow_metrics: Dict[str, Any],
        population_metrics: Dict[str, Any],
        infrastructure_metrics: Dict[str, Any],
        environmental_metrics: Dict[str, Any],
        economic_metrics: Dict[str, Any],
        terramind_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Multi-criteria risk score (0-100) with weighted components.

    Args:
        terrain_metrics: Terrain hazard metrics
        flow_metrics: Flow path metrics
        population_metrics: Population exposure metrics
        infrastructure_metrics: Infrastructure metrics
        environmental_metrics: Environmental vulnerability metrics
        economic_metrics: Economic impact metrics
        terramind_metrics: AI-derived metrics

    Returns:
        Dictionary with composite risk assessment
    """
    print("  Computing comprehensive risk score...")

    # Component weights (sum to 1.0)
    weights = {
        'hazard': 0.30,  # Terrain + flow paths
        'exposure': 0.25,  # Population + infrastructure
        'vulnerability': 0.20,  # Environmental + economic
        'terramind_ai': 0.15,  # Embedding-based anomalies
        'confidence': 0.10  # Data quality adjustment
    }

    # Normalize each component to 0-100 scale

    # 1. HAZARD SCORE (terrain + flow)
    terrain_score = terrain_metrics.get('terrain_hazard_score', 0) * 100
    flow_score = flow_metrics.get('flow_hazard_score', 0) * 100
    hazard_score = terrain_score * 0.6 + flow_score * 0.4

    # 2. EXPOSURE SCORE (population + infrastructure)
    pop_score = population_metrics.get('exposure_score', 0) * 100
    infra_score = infrastructure_metrics.get('infrastructure_score', 0) * 100
    exposure_score = pop_score * 0.6 + infra_score * 0.4

    # 3. VULNERABILITY SCORE (environment + economics)
    env_score = environmental_metrics.get('environmental_risk_score', 0) * 100
    econ_score = economic_metrics.get('economic_risk_score', 0) * 100
    vulnerability_score = env_score * 0.4 + econ_score * 0.6

    # 4. AI INSIGHTS SCORE (TerraMind embeddings)
    embedding_variance = terramind_metrics.get('embedding_variance', 0)
    anomaly_pct = terramind_metrics.get('anomaly_percentage', 0)
    ai_score = (embedding_variance * 100 * 0.6 + anomaly_pct * 0.4)

    # 5. CONFIDENCE FACTOR
    confidence = calculate_confidence(terrain_metrics, population_metrics)

    # Weighted composite score
    composite_score = (
            weights['hazard'] * hazard_score +
            weights['exposure'] * exposure_score +
            weights['vulnerability'] * vulnerability_score +
            weights['terramind_ai'] * ai_score
    )

    # Apply confidence adjustment
    composite_score = composite_score * confidence

    # Ensure score is in valid range
    composite_score = float(np.clip(composite_score, 0, 100))

    # Risk category and fatality estimate
    category, fatalities = categorize_risk(
        composite_score,
        population_metrics.get('high_risk_population', 0)
    )

    return {
        'composite_risk_score': composite_score,
        'risk_category': category,
        'predicted_fatalities': fatalities,
        'component_scores': {
            'hazard': float(hazard_score),
            'exposure': float(exposure_score),
            'vulnerability': float(vulnerability_score),
            'ai_insights': float(ai_score)
        },
        'confidence_level': confidence,
        'risk_interpretation': interpret_risk(composite_score, category)
    }


def categorize_risk(score: float, pop_at_risk: int) -> Tuple[str, int]:
    """
    Categorize risk and estimate fatalities.

    Args:
        score: Composite risk score (0-100)
        pop_at_risk: Population in high-risk zones

    Returns:
        Tuple of (category, estimated_fatalities)
    """
    if score > 75:
        category = 'EXTREME'
        fatality_rate = 0.10  # 10% in extreme scenarios
    elif score > 50:
        category = 'HIGH'
        fatality_rate = 0.05  # 5%
    elif score > 25:
        category = 'MODERATE'
        fatality_rate = 0.01  # 1%
    else:
        category = 'LOW'
        fatality_rate = 0.001  # 0.1%

    fatalities = int(pop_at_risk * fatality_rate)

    return category, fatalities


def calculate_confidence(
        terrain_metrics: Dict[str, Any],
        population_metrics: Dict[str, Any]
) -> float:
    """
    Assess confidence in risk estimate based on data quality.

    Args:
        terrain_metrics: Terrain metrics
        population_metrics: Population metrics

    Returns:
        Confidence factor (0-1)
    """
    confidence = 1.0

    # Reduce confidence if population data is estimated
    if population_metrics.get('data_source') == 'LULC-estimated':
        confidence *= 0.85

    # Reduce confidence if population is very low (less reliable data)
    if population_metrics.get('total_population', 0) < 1000:
        confidence *= 0.80

    # Reduce if terrain is very flat (less hazard gradient)
    if terrain_metrics.get('slope_mean', 0) < 5:
        confidence *= 0.90

    # Reduce if no settlement areas detected
    if population_metrics.get('settlement_area_km2', 0) < 0.1:
        confidence *= 0.75

    return float(confidence)


def interpret_risk(score: float, category: str) -> str:
    """Generate human-readable risk interpretation."""
    interpretations = {
        'EXTREME': f"Critical risk level ({score:.1f}/100). Immediate evacuation planning recommended. Multiple hazard factors converge with significant population exposure.",
        'HIGH': f"Substantial risk level ({score:.1f}/100). Enhanced monitoring and preparedness measures essential. Significant potential for casualties and economic disruption.",
        'MODERATE': f"Moderate risk level ({score:.1f}/100). Regular monitoring advised. Limited but notable exposure to volcanic hazards.",
        'LOW': f"Low risk level ({score:.1f}/100). Standard monitoring protocols sufficient. Minimal immediate threat to population and assets."
    }

    return interpretations.get(category, f"Risk level: {category} ({score:.1f}/100)")


# ============================================================================
# 6. MASTER ASSESSMENT FUNCTION
# ============================================================================

def assess_volcano_comprehensive(
        lat: float,
        lon: float,
        volcano_name: str,
        year: int = 2024,
        buffer_km: float = 10.0,
        region_size: int = 1000,
        pixel_size_m: float = 10.0,
        region_gdp_per_capita: float = 10000
) -> Dict[str, Any]:
    """
    MASTER FUNCTION: Complete volcano risk assessment.
    Integrates all components with TerraMind at the core.

    Args:
        lat, lon: Volcano coordinates
        volcano_name: Name of volcano
        year: Year for Sentinel-2 data
        buffer_km: Analysis buffer radius
        region_size: Size of region to download (pixels)
        pixel_size_m: Pixel resolution in meters
        region_gdp_per_capita: Regional GDP per capita (USD)

    Returns:
        Dictionary with comprehensive risk assessment
    """
    print(f"\n{'=' * 70}")
    print(f"COMPREHENSIVE RISK ASSESSMENT: {volcano_name}")
    print(f"Location: ({lat:.4f}, {lon:.4f})")
    print(f"Buffer: {buffer_km} km | Year: {year}")
    print('=' * 70)

    # Step 1: Acquire all data using TerraMind
    print("\n[1/8] Acquiring geospatial data with TerraMind...")
    try:
        analysis = analyze_location_with_terramind(
            lat=lat,
            lon=lon,
            year=year,
            region_size=region_size,
            generate_lulc=True,
            use_real_dem=True,
            crop_size=CONFIG.get("lulc_crop_size", 256),
            stride=CONFIG.get("lulc_stride", 192),
            batch_size=CONFIG.get("lulc_batch_size", 4),
        )
    except Exception as e:
        print(f"ERROR: Data acquisition failed - {e}")
        return {
            'volcano_name': volcano_name,
            'latitude': lat,
            'longitude': lon,
            'error': f'Data acquisition failed: {str(e)}',
            'status': 'FAILED'
        }

    if analysis is None:
        return {
            'volcano_name': volcano_name,
            'latitude': lat,
            'longitude': lon,
            'error': 'No data available for this location',
            'status': 'FAILED'
        }

    s2_region = analysis['s2_region']
    dem_region = analysis['dem_region']
    lulc_region = analysis['lulc_region']
    embeddings = analysis['embeddings']

    # Step 2: Hazard assessment
    print("\n[2/8] Assessing terrain hazards...")
    terrain_metrics = calculate_terrain_hazard(dem_region, lat, lon, pixel_size_m)

    print("\n[3/8] Modeling flow paths...")
    flow_metrics = model_volcanic_flow_paths(dem_region, lat, lon, buffer_km)

    print("\n[4/8] Analyzing TerraMind embeddings...")
    terramind_hazard = terramind_hazard_embedding_analysis(
        embeddings, dem_region, lulc_region
    )
    ai_anomalies = detect_high_risk_anomalies(
        s2_region, dem_region, lulc_region, embeddings
    )

    # Combine AI metrics
    terramind_metrics = {**terramind_hazard, **ai_anomalies}

    # Step 3: Exposure assessment
    print("\n[5/8] Calculating population exposure...")
    population_metrics = calculate_population_exposure(
        lat, lon, lulc_region, terrain_metrics['hazard_zones'],
        buffer_km, pixel_size_m
    )

    print("\n[6/8] Assessing infrastructure...")
    infrastructure_metrics = assess_infrastructure_exposure(
        lulc_region, terrain_metrics['hazard_zones'], pixel_size_m
    )

    # Step 4: Vulnerability assessment
    print("\n[7/8] Evaluating environmental impact...")
    environmental_metrics = assess_environmental_vulnerability(
        lulc_region, terrain_metrics['hazard_zones'], pixel_size_m
    )

    print("\n[8/8] Estimating economic impact...")
    economic_metrics = assess_economic_impact(
        lulc_region, population_metrics, terrain_metrics['hazard_zones'],
        pixel_size_m, region_gdp_per_capita
    )

    # Step 5: Compute comprehensive risk score
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
        'methodology': 'TerraMind-Enhanced Multi-Criteria Risk Assessment v1.0',
        'data_sources': [
            f'Sentinel-2 L2A ({year})',
            'Copernicus DEM 30m',
            'TerraMind AI LULC Generation',
            population_metrics.get('data_source', 'WorldPop 2020'),
            'HydroSHEDS (optional)'
        ],
        'model_version': {
            'terramind': CONFIG.get('terramind_model', 'terramind_v1_base_generate'),
            'framework': 'Advanced Risk Assessment v1.0'
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
    print(f"Confidence Level: {comprehensive_risk['confidence_level']:.2f}")
    print(f"\n{comprehensive_risk['risk_interpretation']}")
    print('=' * 70)

    return result


# ============================================================================
# 7. BATCH PROCESSING WRAPPER
# ============================================================================

def batch_assess_volcanoes_comprehensive(
        volcano_df: pd.DataFrame,
        name_col: str = "name",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        year: int = 2024,
        buffer_km: float = 10.0,
        region_size: int = 1000
) -> pd.DataFrame:
    """
    Run comprehensive assessment for multiple volcanoes.

    Args:
        volcano_df: DataFrame with volcano data
        name_col: Column name for volcano name
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        year: Year for analysis
        buffer_km: Buffer radius
        region_size: Region size in pixels

    Returns:
        DataFrame with comprehensive risk assessments
    """
    from .risk_logic import bootstrap
    bootstrap()

    results = []

    for idx, row in volcano_df.iterrows():
        name = str(row[name_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        print(f"\n{'=' * 70}")
        print(f"Processing {idx + 1}/{len(volcano_df)}: {name}")
        print('=' * 70)

        try:
            result = assess_volcano_comprehensive(
                lat=lat,
                lon=lon,
                volcano_name=name,
                year=year,
                buffer_km=buffer_km,
                region_size=region_size
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