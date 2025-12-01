import json
import math
import pandas as pd
import os

CSV_PATH = "data/volcano_catalog_clean.csv"
OUT_PATH = "data/volcano_risk.json"

def safe_number(value):
    """Return None if value is NaN, otherwise the float/int."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def main():
    df = pd.read_csv(CSV_PATH)

    features = []
    for _, row in df.iterrows():
        last_year = row.get("last_eruption_year_num", None)
        last_year = safe_number(last_year)

        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": {
                "id": str(row["id"]),
                "name": row["name"],
                "country": row["country"],
                "status": row.get("status", ""),
                "last_eruption_year": last_year,
                # placeholders for now
                "risk_score": 0.0,
                "risk_category": "LOW",
                "predicted_fatalities": 0,
                "has_error": False,
            },
        }
        features.append(feat)

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "total_volcanoes": len(features),
            "successful_assessments": 0,
        },
        "features": features,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"Saved {len(features)} volcanoes to {OUT_PATH}")

if __name__ == "__main__":
    main()
