"""
Validate TerraMind LULC and DEM generation by comparing with real data
for volcanoes from volcano_catalog_clean.csv.

This script:
1. Loads volcanoes from volcano_catalog_clean.csv
2. Downloads real DEM (Copernicus) and LULC (ESA WorldCover)
3. Generates AI LULC and DEM using TerraMind
4. Compares real vs AI and saves results
5. AUTOMATICALLY SKIPS ALREADY PROCESSED VOLCANOES (resume capability)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from datetime import datetime

# Import from your existing modules
from volcanorisk.terramind_engine import (
    download_sentinel2_region,
    download_copernicus_dem,
    download_copernicus_lulc,
    generate_lulc_from_s2_region,
    generate_dem_from_s2_region,
    save_raster_as_geotiff,
)
from volcanorisk.config import CONFIG

# Class definitions
ESA_WORLDCOVER_CLASSES = {
    0: "No data", 1: "Tree", 2: "Shrub", 3: "Grass", 4: "Crop",
    5: "Built", 6: "Bare", 7: "Snow", 8: "Water", 9: "Wetland", 10: "Moss"
}

TERRAMIND_CLASSES = {
    0: "No data", 1: "Water", 2: "Trees", 3: "Flooded veg", 4: "Crops",
    5: "Built", 6: "Bare", 7: "Snow/ice", 8: "Clouds", 9: "Rangeland"
}

# Mapping for comparison
ESA_TO_TERRAMIND = {
    0: 0, 1: 2, 2: 9, 3: 9, 4: 4, 5: 5,
    6: 6, 7: 7, 8: 1, 9: 3, 10: 2,
}


def get_processed_volcanoes(output_base="validation_outputs"):
    """Get list of already processed volcano names by checking output folders."""
    if not os.path.exists(output_base):
        return set()

    processed = set()
    for folder in os.listdir(output_base):
        folder_path = os.path.join(output_base, folder)
        if os.path.isdir(folder_path):
            # Convert folder name back to volcano name (replace underscores with spaces)
            volcano_name = folder.replace('_', ' ')

            # Verify it has expected output files (at least the comparison image)
            expected_files = ['comparison_6panel_with_dem.png', 'comparison_4panel.png']
            has_comparison = any(
                os.path.exists(os.path.join(folder_path, f)) for f in expected_files
            )

            if has_comparison:
                processed.add(volcano_name)

    return processed


def load_volcanoes_from_csv(csv_path, limit=None):
    """Load volcanoes from volcano_catalog_clean.csv."""
    print(f"Loading volcanoes from {csv_path}...")

    df = pd.read_csv(csv_path)

    if limit is not None:
        print(f"  Limiting to first {limit} volcanoes")
        df = df.head(limit)
    else:
        print(f"  Processing all {len(df)} volcanoes")

    volcanoes = []
    for _, row in df.iterrows():
        volcano = {
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "country": row['country'],
            "elevation_m": None,
            "status": row.get('status_simple', 'Unknown'),
        }
        volcanoes.append(volcano)

    print(f"Loaded {len(volcanoes)} volcanoes")
    return volcanoes


def process_volcano(volcano, region_size=1000, generate_dem=True):
    """Process a single volcano: download data and compare.

    Args:
        volcano: Dictionary with volcano information
        region_size: Size of region to download (pixels)
        generate_dem: If True, generate DEM using TerraMind
    """
    name = volcano['name']
    lat = volcano['latitude']
    lon = volcano['longitude']

    print(f"\n{'='*70}")
    print(f"PROCESSING: {name}")
    print(f"Location: ({lat:.2f}, {lon:.2f})")
    print(f"Country: {volcano['country']}")
    print('='*70)

    results = {
        'name': name,
        'latitude': lat,
        'longitude': lon,
        'country': volcano['country'],
        'status': 'processing',
    }

    try:
        # Step 1: Download Sentinel-2
        print(f"\n[1/5] Downloading Sentinel-2 data...")
        s2_region, transform = download_sentinel2_region(lat, lon, 2024, region_size)

        if s2_region is None:
            print(f"ERROR: No Sentinel-2 data available")
            results['status'] = 'failed'
            results['error'] = 'No Sentinel-2 data'
            return results

        # Step 2: Download Real DEM
        print(f"\n[2/5] Downloading Real DEM (Copernicus 30m)...")
        dem_real = download_copernicus_dem(lat, lon, 10, target_size=region_size)

        # Step 2b: Generate AI DEM
        dem_ai = None
        if generate_dem:
            print(f"\n[2b/5] Generating AI DEM (TerraMind)...")
            dem_ai = generate_dem_from_s2_region(
                s2_region,
                crop_size=CONFIG.get('lulc_crop_size', 256),
                stride=CONFIG.get('lulc_stride', 192),
                batch_size=CONFIG.get('lulc_batch_size', 4),
            )

            if dem_ai is None:
                print(f"WARNING: Failed to generate AI DEM")

        # Step 3: Download Real LULC
        print(f"\n[3/5] Downloading Real LULC (ESA WorldCover)...")
        lulc_real = download_copernicus_lulc(lat, lon, 10, target_size=region_size)

        # Step 4: Generate AI LULC
        print(f"\n[4/5] Generating AI LULC (TerraMind)...")
        lulc_ai = generate_lulc_from_s2_region(
            s2_region,
            crop_size=CONFIG.get('lulc_crop_size', 256),
            stride=CONFIG.get('lulc_stride', 192),
            batch_size=CONFIG.get('lulc_batch_size', 4),
        )

        if lulc_ai is None:
            print(f"ERROR: Failed to generate AI LULC")
            results['status'] = 'failed'
            results['error'] = 'LULC generation failed'
            return results

        # Step 5: Analyze results
        print(f"\n[5/5] Analyzing results...")
        print(f"\n{'='*70}")
        print("ANALYSIS")
        print('='*70)

        results.update(analyze_comparison(
            dem_real, dem_ai, lulc_real, lulc_ai, name
        ))

        # Save outputs
        output_dir = f"validation_outputs/{name.replace(' ', '_')}"
        save_all_outputs(
            s2_region, dem_real, dem_ai, lulc_real, lulc_ai,
            lat, lon, name, output_dir, transform
        )

        results['status'] = 'success'
        results['output_dir'] = output_dir

    except Exception as e:
        print(f"\nERROR processing {name}: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def analyze_comparison(dem_real, dem_ai, lulc_real, lulc_ai, name):
    """Analyze and compare real vs AI data for both DEM and LULC."""
    results = {}

    # DEM ANALYSIS
    print(f"\n{'─'*70}")
    print("DEM ANALYSIS")
    print('─'*70)

    print(f"\nReal DEM (Copernicus) Statistics:")
    print(f"  Elevation range: {dem_real.min():.1f} - {dem_real.max():.1f} m")
    print(f"  Mean elevation: {dem_real.mean():.1f} m")
    print(f"  Std deviation: {dem_real.std():.1f} m")

    results['dem_real_min'] = float(dem_real.min())
    results['dem_real_max'] = float(dem_real.max())
    results['dem_real_mean'] = float(dem_real.mean())
    results['dem_real_std'] = float(dem_real.std())

    if dem_ai is not None:
        print(f"\nAI DEM (TerraMind) Statistics:")
        print(f"  Elevation range: {dem_ai.min():.1f} - {dem_ai.max():.1f} m")
        print(f"  Mean elevation: {dem_ai.mean():.1f} m")
        print(f"  Std deviation: {dem_ai.std():.1f} m")

        results['dem_ai_min'] = float(dem_ai.min())
        results['dem_ai_max'] = float(dem_ai.max())
        results['dem_ai_mean'] = float(dem_ai.mean())
        results['dem_ai_std'] = float(dem_ai.std())

        if dem_real.shape == dem_ai.shape:
            mae = np.abs(dem_real - dem_ai).mean()
            rmse = np.sqrt(((dem_real - dem_ai) ** 2).mean())
            correlation = np.corrcoef(dem_real.flatten(), dem_ai.flatten())[0, 1]
            elevation_range = dem_real.max() - dem_real.min()
            relative_mae = (mae / elevation_range * 100) if elevation_range > 0 else 0

            print(f"\nDEM Comparison Metrics:")
            print(f"  Mean Absolute Error (MAE): {mae:.1f} m")
            print(f"  Root Mean Square Error (RMSE): {rmse:.1f} m")
            print(f"  Relative MAE: {relative_mae:.1f}%")
            print(f"  Correlation: {correlation:.3f}")

            results['dem_mae'] = float(mae)
            results['dem_rmse'] = float(rmse)
            results['dem_relative_mae'] = float(relative_mae)
            results['dem_correlation'] = float(correlation)

            dem_score = 0
            if correlation > 0.7:
                dem_score += 1
            if relative_mae < 20:
                dem_score += 1
            if rmse < (elevation_range * 0.25):
                dem_score += 1

            results['dem_validation_score'] = dem_score
            results['dem_validation_status'] = 'PASS' if dem_score >= 2 else 'FAIL'

            print(f"  DEM Validation Score: {dem_score}/3")
            print(f"  DEM Validation Status: {results['dem_validation_status']}")
        else:
            print(f"  WARNING: Shape mismatch - cannot compare")
            results['dem_validation_score'] = 0
            results['dem_validation_status'] = 'FAIL'
    else:
        print(f"\nAI DEM: Not generated")
        results['dem_ai_min'] = None
        results['dem_ai_max'] = None
        results['dem_ai_mean'] = None
        results['dem_ai_std'] = None
        results['dem_mae'] = None
        results['dem_rmse'] = None
        results['dem_relative_mae'] = None
        results['dem_correlation'] = None
        results['dem_validation_score'] = 0
        results['dem_validation_status'] = 'SKIPPED'

    # LULC ANALYSIS
    print(f"\n{'─'*70}")
    print("LULC ANALYSIS")
    print('─'*70)

    print(f"\nReal LULC (ESA WorldCover):")
    unique_real, counts_real = np.unique(lulc_real, return_counts=True)
    real_dist = {}
    for cls, count in zip(unique_real, counts_real):
        pct = count / lulc_real.size * 100
        if pct > 1.0:
            class_name = ESA_WORLDCOVER_CLASSES.get(cls, 'Unknown')
            print(f"  {cls:2d} ({class_name:15s}): {pct:5.1f}%")
            real_dist[int(cls)] = float(pct)

    results['real_lulc_distribution'] = real_dist
    results['real_lulc_classes'] = len(unique_real)

    print(f"\nAI LULC (TerraMind):")
    unique_ai, counts_ai = np.unique(lulc_ai, return_counts=True)
    ai_dist = {}
    for cls, count in zip(unique_ai, counts_ai):
        pct = count / lulc_ai.size * 100
        if pct > 1.0:
            class_name = TERRAMIND_CLASSES.get(cls, 'Unknown')
            print(f"  {cls:2d} ({class_name:15s}): {pct:5.1f}%")
            ai_dist[int(cls)] = float(pct)

    results['ai_lulc_distribution'] = ai_dist
    results['ai_lulc_classes'] = len(unique_ai)

    lulc_real_mapped = np.zeros_like(lulc_real, dtype=np.uint8)
    for esa_cls, tm_cls in ESA_TO_TERRAMIND.items():
        lulc_real_mapped[lulc_real == esa_cls] = tm_cls

    if lulc_ai.shape == lulc_real_mapped.shape:
        agreement = (lulc_ai == lulc_real_mapped).sum() / lulc_real.size * 100
        print(f"\nLULC Pixel-wise Agreement: {agreement:.1f}%")
        results['lulc_agreement_pct'] = float(agreement)
    else:
        results['lulc_agreement_pct'] = 0.0

    max_class_pct = (counts_ai / lulc_ai.size).max() * 100

    lulc_score = 0
    if len(unique_ai) >= 3:
        lulc_score += 1
    if max_class_pct < 90:
        lulc_score += 1
    if results['lulc_agreement_pct'] > 40:
        lulc_score += 1

    results['lulc_validation_score'] = lulc_score
    results['lulc_validation_status'] = 'PASS' if lulc_score >= 2 else 'FAIL'

    print(f"\nLULC Validation Score: {lulc_score}/3")
    print(f"LULC Validation Status: {results['lulc_validation_status']}")

    # OVERALL VALIDATION
    print(f"\n{'─'*70}")
    print("OVERALL VALIDATION")
    print('─'*70)

    overall_score = lulc_score
    if dem_ai is not None:
        overall_score += results['dem_validation_score']
        overall_status = 'PASS' if overall_score >= 4 else 'FAIL'
        print(f"  Combined Score: {overall_score}/6")
    else:
        overall_status = 'PASS' if overall_score >= 2 else 'FAIL'
        print(f"  LULC Score: {overall_score}/3 (DEM not generated)")

    results['overall_validation_score'] = overall_score
    results['overall_validation_status'] = overall_status
    print(f"  Overall Status: {overall_status}")

    return results


def save_all_outputs(s2_region, dem_real, dem_ai, lulc_real, lulc_ai,
                     lat, lon, name, output_dir, transform):
    """Save all data outputs including AI-generated DEM."""
    print(f"\nSaving outputs to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)

    save_raster_as_geotiff(
        s2_region, lat, lon, 10,
        f"{output_dir}/s2_rgb.tif",
        transform=transform
    )

    save_raster_as_geotiff(
        dem_real, lat, lon, 10,
        f"{output_dir}/dem_copernicus_real.tif",
        transform=transform
    )

    if dem_ai is not None:
        save_raster_as_geotiff(
            dem_ai, lat, lon, 10,
            f"{output_dir}/dem_terramind_ai.tif",
            transform=transform
        )

    save_raster_as_geotiff(
        lulc_real, lat, lon, 10,
        f"{output_dir}/lulc_esa_real.tif",
        transform=transform
    )

    save_raster_as_geotiff(
        lulc_ai.astype(np.uint8), lat, lon, 10,
        f"{output_dir}/lulc_terramind_ai.tif",
        transform=transform
    )

    if dem_ai is not None:
        create_comparison_figure_with_dem(
            s2_region, dem_real, dem_ai, lulc_real, lulc_ai,
            name, output_dir
        )
    else:
        create_comparison_figure(
            s2_region, dem_real, lulc_real, lulc_ai,
            name, output_dir
        )

    print(f"✓ All outputs saved")


def create_comparison_figure_with_dem(s2_region, dem_real, dem_ai,
                                      lulc_real, lulc_ai, name, output_dir):
    """Create 6-panel comparison figure including DEM comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Panel 1: Sentinel-2 RGB
    rgb = s2_region[[3, 2, 1], :, :].transpose(1, 2, 0)
    rgb = np.clip(rgb / 2000, 0, 1)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Sentinel-2 RGB', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Real DEM
    dem_plot_real = axes[0, 1].imshow(dem_real, cmap='terrain')
    axes[0, 1].set_title('Real DEM (Copernicus 30m)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(dem_plot_real, ax=axes[0, 1], label='Elevation (m)', fraction=0.046)

    # Panel 3: AI DEM
    dem_plot_ai = axes[0, 2].imshow(dem_ai, cmap='terrain')
    axes[0, 2].set_title('AI DEM (TerraMind)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(dem_plot_ai, ax=axes[0, 2], label='Elevation (m)', fraction=0.046)

    # Panel 4: DEM Difference
    dem_diff = dem_ai - dem_real
    max_diff = max(abs(dem_diff.min()), abs(dem_diff.max()))
    dem_diff_plot = axes[1, 0].imshow(dem_diff, cmap='RdBu_r',
                                       vmin=-max_diff, vmax=max_diff)
    axes[1, 0].set_title('DEM Difference (AI - Real)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(dem_diff_plot, ax=axes[1, 0], label='Elevation Diff (m)', fraction=0.046)

    # Panel 5: Real LULC
    esa_cmap = mcolors.ListedColormap([
        '#000000', '#228B22', '#90EE90', '#9ACD32', '#FFD700',
        '#FF0000', '#D2B48C', '#FFFFFF', '#0000FF', '#00CED1', '#006400',
    ])
    axes[1, 1].imshow(lulc_real, cmap=esa_cmap, vmin=0, vmax=10)
    axes[1, 1].set_title('Real LULC (ESA WorldCover)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Panel 6: AI LULC
    tm_cmap = mcolors.ListedColormap([
        '#000000', '#0000FF', '#228B22', '#90EE90', '#FFD700',
        '#FF0000', '#D2B48C', '#FFFFFF', '#808080', '#9ACD32',
    ])
    axes[1, 2].imshow(lulc_ai, cmap=tm_cmap, vmin=0, vmax=9)
    axes[1, 2].set_title('AI LULC (TerraMind)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'Validation: {name}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = f"{output_dir}/comparison_6panel_with_dem.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def create_comparison_figure(s2_region, dem_real, lulc_real, lulc_ai, name, output_dir):
    """Create 4-panel comparison figure (fallback without AI DEM)."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Panel 1: Sentinel-2 RGB
    rgb = s2_region[[3, 2, 1], :, :].transpose(1, 2, 0)
    rgb = np.clip(rgb / 2000, 0, 1)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Sentinel-2 RGB', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Real LULC
    esa_cmap = mcolors.ListedColormap([
        '#000000', '#228B22', '#90EE90', '#9ACD32', '#FFD700',
        '#FF0000', '#D2B48C', '#FFFFFF', '#0000FF', '#00CED1', '#006400',
    ])
    axes[0, 1].imshow(lulc_real, cmap=esa_cmap, vmin=0, vmax=10)
    axes[0, 1].set_title('Real LULC (ESA WorldCover)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: AI LULC
    tm_cmap = mcolors.ListedColormap([
        '#000000', '#0000FF', '#228B22', '#90EE90', '#FFD700',
        '#FF0000', '#D2B48C', '#FFFFFF', '#808080', '#9ACD32',
    ])
    axes[1, 0].imshow(lulc_ai, cmap=tm_cmap, vmin=0, vmax=9)
    axes[1, 0].set_title('AI LULC (TerraMind)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Panel 4: DEM
    dem_plot = axes[1, 1].imshow(dem_real, cmap='terrain')
    axes[1, 1].set_title('DEM (Copernicus 30m)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(dem_plot, ax=axes[1, 1], label='Elevation (m)', fraction=0.046)

    plt.suptitle(f'Validation: {name}', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = f"{output_dir}/comparison_4panel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_summary_report(all_results):
    """Generate summary report of all validations."""
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY REPORT")
    print('='*70)

    df = pd.DataFrame(all_results)

    total = len(df)
    success = len(df[df['status'] == 'success'])
    failed = total - success

    print(f"\nProcessing Statistics:")
    print(f"  Total volcanoes: {total}")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")

    if success > 0:
        success_df = df[df['status'] == 'success']

        print(f"\nOverall Validation Results:")
        passed = len(success_df[success_df['overall_validation_status'] == 'PASS'])
        print(f"  Passed: {passed}/{success}")
        print(f"  Failed: {success - passed}/{success}")

        print(f"\nLULC Metrics:")
        print(f"  Average Agreement: {success_df['lulc_agreement_pct'].mean():.1f}%")
        print(f"  Average Score: {success_df['lulc_validation_score'].mean():.1f}/3")
        print(f"  Average Classes: {success_df['ai_lulc_classes'].mean():.1f}")

        lulc_passed = len(success_df[success_df['lulc_validation_status'] == 'PASS'])
        print(f"  LULC Passed: {lulc_passed}/{success}")

        dem_generated = success_df['dem_ai_min'].notna().sum()
        if dem_generated > 0:
            print(f"\nDEM Metrics ({dem_generated} volcanoes):")
            dem_df = success_df[success_df['dem_ai_min'].notna()]
            print(f"  Average MAE: {dem_df['dem_mae'].mean():.1f} m")
            print(f"  Average RMSE: {dem_df['dem_rmse'].mean():.1f} m")
            print(f"  Average Relative MAE: {dem_df['dem_relative_mae'].mean():.1f}%")
            print(f"  Average Correlation: {dem_df['dem_correlation'].mean():.3f}")

            dem_passed = len(dem_df[dem_df['dem_validation_status'] == 'PASS'])
            print(f"  DEM Passed: {dem_passed}/{dem_generated}")

        print(f"\nIndividual Results:")
        print(f"  {'Volcano':<25} {'Overall':<8} {'LULC':<8} {'DEM':<8} {'Agreement':<10} {'DEM MAE':<10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        for _, row in success_df.iterrows():
            overall_icon = "✓" if row['overall_validation_status'] == 'PASS' else "✗"
            lulc_icon = "✓" if row['lulc_validation_status'] == 'PASS' else "✗"

            dem_status = row.get('dem_validation_status', 'SKIPPED')
            if dem_status == 'PASS':
                dem_icon = "✓"
            elif dem_status == 'FAIL':
                dem_icon = "✗"
            else:
                dem_icon = "-"

            agreement = row['lulc_agreement_pct']
            dem_mae = row.get('dem_mae', None)
            dem_mae_str = f"{dem_mae:.1f}m" if dem_mae is not None else "N/A"

            print(f"  {row['name']:<25} {overall_icon:^8} {lulc_icon:^8} {dem_icon:^8} "
                  f"{agreement:>5.1f}%     {dem_mae_str:>10}")

    report_path = "validation_outputs/validation_report.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"\n✓ Saved detailed report: {report_path}")

    summary = {
        'generated_at': datetime.utcnow().isoformat(),
        'total_volcanoes': int(total),
        'successful': int(success),
        'failed': int(failed),
        'overall_validation_passed': int(passed) if success > 0 else 0,
    }

    if success > 0:
        summary['lulc'] = {
            'average_agreement_pct': float(success_df['lulc_agreement_pct'].mean()),
            'average_validation_score': float(success_df['lulc_validation_score'].mean()),
            'passed': int(lulc_passed),
        }

        if dem_generated > 0:
            dem_df = success_df[success_df['dem_ai_min'].notna()]
            summary['dem'] = {
                'volcanoes_with_dem': int(dem_generated),
                'average_mae': float(dem_df['dem_mae'].mean()),
                'average_rmse': float(dem_df['dem_rmse'].mean()),
                'average_relative_mae': float(dem_df['dem_relative_mae'].mean()),
                'average_correlation': float(dem_df['dem_correlation'].mean()),
                'passed': int(dem_passed),
            }

    summary_path = "validation_outputs/validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")


def main():
    """Main execution."""
    print("="*70)
    print("TERRAMIND VALIDATION: LULC + DEM GENERATION")
    print("Real Data vs AI-Generated (LULC + DEM)")
    print("WITH AUTOMATIC RESUME CAPABILITY")
    print("="*70)

    # Configuration
    CSV_PATH = "data/volcano_catalog_clean.csv"
    NUM_VOLCANOES = None  # None for all, or set a number like 20, 100
    REGION_SIZE = 1000
    GENERATE_DEM = True

    # Check if file exists
    if not os.path.exists(CSV_PATH):
        print(f"\nERROR: {CSV_PATH} not found!")
        print("Please make sure volcano_catalog_clean.csv exists in data/ folder.")
        sys.exit(1)

    # Load all volcanoes
    print(f"\n[1/4] Loading volcanoes from {CSV_PATH}...")
    all_volcanoes = load_volcanoes_from_csv(CSV_PATH, limit=NUM_VOLCANOES)

    if not all_volcanoes:
        print("ERROR: No volcanoes loaded")
        sys.exit(1)

    # Check for already processed volcanoes
    print(f"\n[2/4] Checking for already processed volcanoes...")
    processed = get_processed_volcanoes()

    if processed:
        print(f"  Found {len(processed)} already processed volcanoes:")
        for name in sorted(list(processed))[:5]:
            print(f"    ✓ {name}")
        if len(processed) > 5:
            print(f"    ... and {len(processed) - 5} more")
    else:
        print(f"  No previously processed volcanoes found")

    # Filter out already processed
    volcanoes = [v for v in all_volcanoes if v['name'] not in processed]

    print(f"\n  Processing Summary:")
    print(f"    Total volcanoes in CSV: {len(all_volcanoes)}")
    print(f"    Already processed: {len(processed)}")
    print(f"    Remaining to process: {len(volcanoes)}")

    if not volcanoes:
        print("\n✓ All volcanoes have been processed!")
        print("  Run generate_summary_report() to regenerate summary if needed.")
        sys.exit(0)

    print(f"\n  Next volcanoes to process:")
    for i, v in enumerate(volcanoes[:10], 1):
        print(f"    {i}. {v['name']} ({v['country']}) - {v['latitude']:.2f}, {v['longitude']:.2f}")
    if len(volcanoes) > 10:
        print(f"    ... and {len(volcanoes) - 10} more")

    # Process each volcano
    print(f"\n[3/4] Processing volcanoes...")
    print(f"  Region size: {REGION_SIZE}x{REGION_SIZE}")
    print(f"  Generate DEM: {'YES' if GENERATE_DEM else 'NO'}")
    print(f"  Generate LULC: YES (always)")

    all_results = []

    for i, volcano in enumerate(volcanoes, 1):
        print(f"\n{'='*70}")
        print(f"VOLCANO {i}/{len(volcanoes)} (Overall: {len(processed) + i}/{len(all_volcanoes)})")
        print('='*70)

        result = process_volcano(
            volcano,
            region_size=REGION_SIZE,
            generate_dem=GENERATE_DEM
        )
        all_results.append(result)

        # Save intermediate results after each volcano
        if result['status'] == 'success':
            print(f"\n✓ Successfully processed {volcano['name']}")
        else:
            print(f"\n✗ Failed to process {volcano['name']}: {result.get('error', 'Unknown error')}")

    # Generate summary
    print(f"\n[4/4] Generating summary report...")

    # Load previously processed results if they exist
    report_path = "validation_outputs/validation_report.csv"
    if os.path.exists(report_path):
        print(f"  Loading previous results from {report_path}...")
        previous_df = pd.read_csv(report_path)
        previous_results = previous_df.to_dict('records')

        # Combine with new results (avoid duplicates)
        processed_names = {r['name'] for r in all_results}
        for prev_result in previous_results:
            if prev_result['name'] not in processed_names:
                all_results.append(prev_result)

        print(f"  Combined {len(previous_results)} previous + {len([r for r in all_results if r['name'] in processed_names])} new results")

    generate_summary_report(all_results)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE!")
    print('='*70)
    print(f"\nOutputs saved to: validation_outputs/")
    print(f"  - Individual volcano folders with:")
    print(f"    • GeoTIFFs (S2, DEM real, DEM AI, LULC real, LULC AI)")
    print(f"    • 6-panel comparison image (with DEM comparison)")
    print(f"  - validation_report.csv (detailed metrics)")
    print(f"  - validation_summary.json (summary statistics)")
    print(f"\nProcessing Statistics:")
    print(f"  Total volcanoes in catalog: {len(all_volcanoes)}")
    print(f"  Processed in this run: {len(volcanoes)}")
    print(f"  Total processed: {len(all_results)}")
    print(f"\nNext steps:")
    print(f"  1. Review 6-panel comparison images (now includes DEM!)")
    print(f"  2. Open GeoTIFFs in QGIS for detailed inspection")
    print(f"  3. Check validation_report.csv for quantitative metrics")
    print(f"  4. Compare DEM MAE and correlation values")
    print(f"\nTo resume if interrupted: Just run this script again!")


if __name__ == "__main__":
    main()