import pandas as pd
import os

SRC = r"D:\Riska\Terramind\volcano-risk-app\data\volcano_catalog.csv"
DST = r"D:\Riska\Terramind\volcano-risk-app\data\volcano_catalog_clean.csv"

# add encoding
df = pd.read_csv(SRC, encoding="latin1")  # or encoding="cp1252"


# 1) Normalize status to a simpler flag (active / other)
def map_status(s: str) -> str:
    s = str(s).strip().lower()
    if any(k in s for k in ["eruption dated", "eruption observed", "unrest"]):
        return "active"
    return "other"

df["status_simple"] = df["status"].apply(map_status)

# 2) Parse last_eruption_year to integer where possible
def parse_year(s: str):
    s = str(s).strip()
    if s.lower() == "unknown" or s == "":
        return None
    # e.g. "2025 CE" or "500 BCE"
    parts = s.split()
    try:
        year = int(parts[0])
    except Exception:
        return None
    if len(parts) > 1 and parts[1].upper() == "BCE":
        return -year
    return year

df["last_eruption_year_num"] = df["last_eruption_year"].apply(parse_year)

# Optionally filter to "active" only
df_active = df[df["status_simple"] == "active"].copy()

os.makedirs(os.path.dirname(DST), exist_ok=True)
df_active.to_csv(DST, index=False)
print(f"Saved cleaned catalog to {DST}, n={len(df_active)}")
