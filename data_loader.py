"""Load Japan power-market datasets from local CSV files."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

THERMAL_STACK_EXPORT_DATASET_KEY = "thermal_stack"

_DATA_ROOT = Path(os.environ.get("THERMAL_STACK_DATA_ROOT", Path(__file__).resolve().parent))
_WINDOWS_DATA_ROOT = Path(r"C:\Develop\data")


def _resolve_data_file(*relative_paths: str) -> Path:
    """Return the first existing path among repo root and legacy Windows locations."""
    candidates: list[Path] = []
    for relative_path in relative_paths:
        candidates.append(_DATA_ROOT / relative_path)
        candidates.append(_WINDOWS_DATA_ROOT / relative_path)
        if relative_path.startswith("Japan/"):
            candidates.append(_DATA_ROOT / relative_path.split("/", 1)[1])

    for path in candidates:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find data file. Searched: {searched}")


def resolve_dataset_output_path(dataset_key: str) -> Path:
    """Resolve export path for thermal stack Excel output."""
    if dataset_key == THERMAL_STACK_EXPORT_DATASET_KEY:
        return _resolve_data_file("Japan/thermalStack.xlsx", "thermalStack.xlsx")
    return _DATA_ROOT / f"{dataset_key}.xlsx"


def load_jepx_spot() -> pd.DataFrame:
    """Load 30-minute JEPX spot prices."""
    path = _resolve_data_file("jepx_spot.csv", "jepx/jepx_spot.parquet")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.set_index("datetime")
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("Asia/Tokyo")
    return frame.sort_index()


def load_df30min() -> pd.DataFrame:
    """Load 30-minute regional consumption and generation wide table."""
    path = _resolve_data_file("JP_30minWide.csv", "JP_30minWide.parquet")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, header=[0, 1], index_col=0)

    frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame.sort_index()


def _normalize_fuel_cocktail(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize fuel cocktail columns for marginal-cost computation."""
    cocktail = frame.copy()
    if "date" in cocktail.columns:
        cocktail["date"] = pd.to_datetime(cocktail["date"], errors="coerce")
        cocktail = cocktail.set_index("date")
    cocktail.index = pd.to_datetime(cocktail.index)
    cocktail = cocktail.sort_index()

    if "coal_cif_eurmwh" not in cocktail.columns and "newc_eurmwh" in cocktail.columns:
        cocktail["coal_cif_eurmwh"] = cocktail["newc_eurmwh"]

    rename_map = {"EURUSD": "EURUSD", "eurusd": "EURUSD", "EURJPY": "EURJPY", "eurjpy": "EURJPY"}
    cocktail = cocktail.rename(columns={source: target for source, target in rename_map.items() if source in cocktail.columns})
    return cocktail


def load_japan_fuel_cocktail() -> pd.DataFrame:
    """Load daily Japan fuel cocktail prices."""
    path = _resolve_data_file("japan_fuel_cocktail.csv", "Japan/japan_fuel_cocktail.csv")
    return _normalize_fuel_cocktail(pd.read_csv(path))


def load_unit_production() -> pd.DataFrame:
    """Load HJKS unit production history."""
    path = _resolve_data_file("Japan/unit_production.csv", "unit_production.csv")
    return pd.read_csv(path)


def load_thermal_efficiency_registry() -> pd.DataFrame:
    """Load thermal plant efficiency registry."""
    path = _resolve_data_file("Japan/thermal_efficiency_registry.csv", "thermal_efficiency_registry.csv")
    return pd.read_csv(path)


def load_nuclear_registry() -> pd.DataFrame:
    """Load nuclear plant registry."""
    path = _resolve_data_file("Japan/nuclear_registry.csv", "nuclear_registry.csv")
    return pd.read_csv(path)


def load_outages() -> pd.DataFrame:
    """Load plant outage events."""
    path = _resolve_data_file("Japan/outages.csv", "outages.csv")
    return pd.read_csv(path)
