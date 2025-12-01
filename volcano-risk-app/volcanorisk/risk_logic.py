"""High-level volcano risk API - UPDATED with Advanced Risk Assessment."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from .config import CONFIG
from .terramind_engine import init_models
from .advanced_risk_assessment import (
    assess_volcano_comprehensive,
    batch_assess_volcanoes_comprehensive
)

# Import optimized version that uses pre-computed data
try:
    from .optimized_risk_assessment import (
        assess_volcano_comprehensive_optimized,
        batch_assess_volcanoes_optimized
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

_GLOBAL_MODELS: Optional[Dict[str, Any]] = None


def bootstrap() -> None:
    """Ensure models are loaded once."""
    global _GLOBAL_MODELS
    if _GLOBAL_MODELS is None:
        _GLOBAL_MODELS = init_models(CONFIG)


def assess_volcano_detailed(
    lat: float,
    lon: float,
    volcano_name: str,
    year: int = 2024,
    buffer_km: Optional[float] = None,
    use_advanced_assessment: bool = True,  # NEW: Use advanced framework
) -> Dict[str, Any]:
    """
    Run full risk assessment for a volcano.

    Args:
        lat, lon: Volcano coordinates
        volcano_name: Name of volcano
        year: Year for Sentinel-2 data
        buffer_km: Buffer radius (default from CONFIG)
        use_advanced_assessment: Use comprehensive risk framework (recommended)

    Returns:
        Dictionary with comprehensive risk assessment
    """
    bootstrap()

    radius_km = buffer_km if buffer_km is not None else CONFIG["radius_km_default"]
    region_size = CONFIG.get("default_region_size", 1000)

    if use_advanced_assessment:
        # NEW: Use comprehensive multi-criteria risk assessment
        print(f"Using ADVANCED RISK ASSESSMENT for {volcano_name}")

        try:
            result = assess_volcano_comprehensive(
                lat=lat,
                lon=lon,
                volcano_name=volcano_name,
                year=year,
                buffer_km=radius_km,
                region_size=region_size,
                pixel_size_m=10.0,
                region_gdp_per_capita=10000  # Adjust based on region
            )

            return result

        except Exception as e:
            print(f"ERROR in advanced assessment for {volcano_name}: {e}")
            import traceback
            traceback.print_exc()

            return {
                "volcano_name": volcano_name,
                "latitude": lat,
                "longitude": lon,
                "buffer_radius_km": radius_km,
                "assessment_date": datetime.utcnow().isoformat(),
                "year": year,
                "status": "FAILED",
                "error": f"Assessment error: {str(e)}",
            }

    else:
        # LEGACY: Simple assessment (kept for compatibility)
        print(f"Using LEGACY assessment for {volcano_name}")

        from .terramind_engine import analyze_location_with_terramind

        try:
            analysis_results = analyze_location_with_terramind(
                lat=lat,
                lon=lon,
                year=year,
                region_size=region_size,
                generate_lulc=True,
                use_real_lulc=False,
                crop_size=CONFIG.get("lulc_crop_size", 256),
                stride=CONFIG.get("lulc_stride", 192),
                batch_size=CONFIG.get("lulc_batch_size", 4),
            )

            if analysis_results is None:
                return {
                    "volcano_name": volcano_name,
                    "latitude": lat,
                    "longitude": lon,
                    "buffer_radius_km": radius_km,
                    "assessment_date": datetime.utcnow().isoformat(),
                    "year": year,
                    "error": "Analysis failed - no data available",
                }

            metrics = analysis_results["metrics"]

            result: Dict[str, Any] = {
                "volcano_name": volcano_name,
                "latitude": lat,
                "longitude": lon,
                "buffer_radius_km": radius_km,
                "assessment_date": datetime.utcnow().isoformat(),
                "year": year,
                "method": "legacy",
            }
            result.update(metrics)

            return result

        except Exception as e:
            print(f"ERROR in legacy assessment for {volcano_name}: {e}")
            return {
                "volcano_name": volcano_name,
                "latitude": lat,
                "longitude": lon,
                "buffer_radius_km": radius_km,
                "assessment_date": datetime.utcnow().isoformat(),
                "year": year,
                "error": f"Analysis error: {str(e)}",
            }


def batch_assess_volcanoes_detailed(
    volcano_df: pd.DataFrame,
    name_col: str = "name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    buffer_km: Optional[float] = None,
    year: int = 2024,
    use_advanced_assessment: bool = True,  # NEW: Use advanced framework
) -> pd.DataFrame:
    """
    Run assessment for multiple volcanoes.

    Args:
        volcano_df: DataFrame with volcano data
        name_col: Column name for volcano name
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        buffer_km: Buffer radius (default from CONFIG)
        year: Year for Sentinel-2 data
        use_advanced_assessment: Use comprehensive risk framework (recommended)

    Returns:
        DataFrame with risk assessments for all volcanoes
    """
    bootstrap()

    radius_km = buffer_km if buffer_km is not None else CONFIG["radius_km_default"]
    region_size = CONFIG.get("default_region_size", 1000)

    if use_advanced_assessment:
        # Use advanced batch processing
        print("Using ADVANCED RISK ASSESSMENT (batch mode)")

        return batch_assess_volcanoes_comprehensive(
            volcano_df=volcano_df,
            name_col=name_col,
            lat_col=lat_col,
            lon_col=lon_col,
            year=year,
            buffer_km=radius_km,
            region_size=region_size
        )

    else:
        # Legacy batch processing
        print("Using LEGACY assessment (batch mode)")

        results: List[Dict[str, Any]] = []

        for idx, row in volcano_df.iterrows():
            name = str(row[name_col])
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            print(f"\n{'='*70}")
            print(f"Processing {idx+1}/{len(volcano_df)}: {name}")
            print('='*70)

            res = assess_volcano_detailed(
                lat=lat,
                lon=lon,
                volcano_name=name,
                year=year,
                buffer_km=radius_km,
                use_advanced_assessment=False,
            )
            results.append(res)

        return pd.DataFrame(results)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_risk_assessment(
    lat: float,
    lon: float,
    volcano_name: str = "Unknown Volcano"
) -> Dict[str, Any]:
    """
    Quick risk assessment with default parameters.

    Args:
        lat, lon: Coordinates
        volcano_name: Volcano name

    Returns:
        Risk assessment dictionary
    """
    return assess_volcano_detailed(
        lat=lat,
        lon=lon,
        volcano_name=volcano_name,
        year=2024,
        buffer_km=10.0,
        use_advanced_assessment=True
    )


def print_risk_summary(result: Dict[str, Any]) -> None:
    """
    Print a formatted summary of risk assessment results.

    Args:
        result: Risk assessment dictionary
    """
    print(f"\n{'='*70}")
    print(f"RISK ASSESSMENT SUMMARY: {result['volcano_name']}")
    print('='*70)

    if result.get('status') == 'FAILED':
        print(f"❌ Assessment Failed")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print(f"\n📍 Location: ({result['latitude']:.4f}, {result['longitude']:.4f})")
    print(f"📅 Assessment Date: {result['assessment_date']}")
    print(f"🎯 Year: {result['year']}")

    print(f"\n🚨 RISK LEVEL")
    print(f"  Score: {result['composite_risk_score']:.1f}/100")
    print(f"  Category: {result['risk_category']}")
    print(f"  Confidence: {result['confidence_level']:.2f}")

    print(f"\n👥 POPULATION IMPACT")
    pop = result['exposure_assessment']['population']
    print(f"  Total Population: {pop['total_population']:,}")
    print(f"  High Risk Population: {pop['high_risk_population']:,}")
    print(f"  Predicted Fatalities: {result['predicted_fatalities']:,}")

    print(f"\n💰 ECONOMIC IMPACT")
    econ = result['vulnerability_assessment']['economic']
    print(f"  Total Estimated Loss: ${econ['total_estimated_loss_usd']:,.0f}")
    print(f"  Loss Category: {econ['loss_category']}")

    print(f"\n🏔️ HAZARD ASSESSMENT")
    hazard = result['hazard_assessment']['terrain']
    print(f"  Mean Slope: {hazard['slope_mean']:.1f}°")
    print(f"  High/Extreme Hazard Area: {hazard['combined_high_extreme_pct']:.1f}%")
    print(f"  Terrain Ruggedness: {hazard['terrain_ruggedness_index']:.2f}")

    print(f"\n🌳 ENVIRONMENTAL IMPACT")
    env = result['vulnerability_assessment']['environmental']
    print(f"  Forest at Risk: {env['forest_at_risk_km2']:.2f} km²")
    print(f"  Agriculture at Risk: {env['agriculture_at_risk_km2']:.2f} km²")
    print(f"  Carbon at Risk: {env['estimated_carbon_at_risk_tons']:,.0f} tons")

    print(f"\n🤖 AI INSIGHTS")
    ai = result['hazard_assessment']['ai_insights']
    print(f"  Terrain Complexity: {ai.get('terrain_complexity_score', 0):.3f}")
    print(f"  Anomaly Detection: {ai.get('anomaly_percentage', 0):.1f}%")
    print(f"  Risk Flag: {ai.get('risk_flag', 'UNKNOWN')}")

    print(f"\n📊 COMPONENT SCORES")
    scores = result['component_scores']
    print(f"  Hazard: {scores['hazard']:.1f}")
    print(f"  Exposure: {scores['exposure']:.1f}")
    print(f"  Vulnerability: {scores['vulnerability']:.1f}")
    print(f"  AI Insights: {scores['ai_insights']:.1f}")

    print(f"\n💬 INTERPRETATION")
    print(f"  {result['risk_interpretation']}")

    print(f"\n{'='*70}\n")


def export_results_to_csv(
    results_df: pd.DataFrame,
    output_path: str = "volcano_risk_assessment_results.csv"
) -> str:
    """
    Export assessment results to CSV.

    Args:
        results_df: DataFrame with assessment results
        output_path: Output file path

    Returns:
        Path to saved file
    """
    import os

    # Select key columns for CSV export
    key_columns = [
        'volcano_name',
        'latitude',
        'longitude',
        'composite_risk_score',
        'risk_category',
        'predicted_fatalities',
        'confidence_level',
        'assessment_date'
    ]

    # Add nested columns if they exist
    if 'exposure_assessment' in results_df.columns:
        # Flatten nested dictionaries
        flattened_data = []
        for _, row in results_df.iterrows():
            flat_row = {col: row[col] for col in key_columns if col in row}

            # Add exposure data
            if 'exposure_assessment' in row and isinstance(row['exposure_assessment'], dict):
                exp = row['exposure_assessment']
                if 'population' in exp:
                    flat_row['total_population'] = exp['population'].get('total_population', 0)
                    flat_row['high_risk_population'] = exp['population'].get('high_risk_population', 0)

            # Add economic data
            if 'vulnerability_assessment' in row and isinstance(row['vulnerability_assessment'], dict):
                vuln = row['vulnerability_assessment']
                if 'economic' in vuln:
                    flat_row['total_loss_usd'] = vuln['economic'].get('total_estimated_loss_usd', 0)

            flattened_data.append(flat_row)

        export_df = pd.DataFrame(flattened_data)
    else:
        export_df = results_df[key_columns] if all(c in results_df.columns for c in key_columns) else results_df

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    export_df.to_csv(output_path, index=False)

    print(f"✅ Results exported to: {output_path}")
    return output_path
