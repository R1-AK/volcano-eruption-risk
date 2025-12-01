"""
Data I/O utilities:
- load volcano catalog
- save risk results
"""

from typing import Optional

import os
import pandas as pd


def load_volcano_catalog(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join("data", "volcano_catalog_clean.csv")
    return pd.read_csv(path)



def save_risk_results(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """
    Save risk assessment results to CSV and return the file path.
    """
    if path is None:
        path = os.path.join("data", "precomputed_risk.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
