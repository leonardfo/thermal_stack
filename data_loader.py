"""Load Japan power-market datasets from local CSV files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

THERMAL_STACK_EXPORT_DATASET_KEY = "thermal_stack"
MMBTU_TO_MWH = 0.293071
NEWC_MMBTU_PER_TONNE = 22.0

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


def _resolve_jpyusd(frame: pd.DataFrame) -> pd.Series:
    """Return JPY per USD series from cocktail columns."""
    if "jpyusd" in frame.columns:
        return pd.to_numeric(frame["jpyusd"], errors="coerce")
    if "JPYUSD" in frame.columns:
        return pd.to_numeric(frame["JPYUSD"], errors="coerce")
    if "EURJPY" in frame.columns and "EURUSD" in frame.columns:
        return pd.to_numeric(frame["EURJPY"], errors="coerce") / pd.to_numeric(frame["EURUSD"], errors="coerce")
    raise ValueError("fuel cocktail must contain jpyusd, JPYUSD, or EURJPY/EURUSD.")


def usd_mmbtu_to_jpy_kwh(usd_mmbtu: pd.Series, jpyusd: pd.Series) -> pd.Series:
    """Convert USD/MMBtu to JPY/kWh of fuel energy."""
    return pd.to_numeric(usd_mmbtu, errors="coerce") * jpyusd / MMBTU_TO_MWH / 1000.0


def enrich_fuel_cocktail_jpy_kwh(cocktail: pd.DataFrame) -> pd.DataFrame:
    """Add or refresh Japan fuel prices in JPY/kWh."""
    frame = cocktail.copy()
    jpyusd = _resolve_jpyusd(frame)

    if "jcc_usd_mmbtu" in frame.columns:
        frame["jcc_jpykwh"] = usd_mmbtu_to_jpy_kwh(frame["jcc_usd_mmbtu"], jpyusd)

    if "jlc_usd_mmbtu" in frame.columns:
        frame["jlc_jpykwh"] = usd_mmbtu_to_jpy_kwh(frame["jlc_usd_mmbtu"], jpyusd)

    lng_usd_mmbtu = None
    if "jkm_usd_mmbtu" in frame.columns:
        lng_usd_mmbtu = pd.to_numeric(frame["jkm_usd_mmbtu"], errors="coerce")
    elif "JKM" in frame.columns:
        lng_usd_mmbtu = pd.to_numeric(frame["JKM"], errors="coerce")
    if lng_usd_mmbtu is not None:
        frame["jkm_jpykwh"] = usd_mmbtu_to_jpy_kwh(lng_usd_mmbtu, jpyusd)

    ttf_usd_mmbtu = None
    if "ttf_usd_mmbtu" in frame.columns:
        ttf_usd_mmbtu = pd.to_numeric(frame["ttf_usd_mmbtu"], errors="coerce")
    elif "TTF" in frame.columns:
        ttf_usd_mmbtu = pd.to_numeric(frame["TTF"], errors="coerce")
    if ttf_usd_mmbtu is not None:
        frame["ttf_jpykwh"] = usd_mmbtu_to_jpy_kwh(ttf_usd_mmbtu, jpyusd)

    coal_usd_t = None
    for column in ("coal_jpn_cif_usd_t", "coal_cif_usd_t", "newc"):
        if column in frame.columns:
            coal_usd_t = pd.to_numeric(frame[column], errors="coerce")
            break
    if coal_usd_t is not None:
        frame["coal_cif_jpykwh"] = usd_mmbtu_to_jpy_kwh(
            coal_usd_t / NEWC_MMBTU_PER_TONNE,
            jpyusd,
        )

    return frame


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
    cocktail = cocktail.rename(
        columns={source: target for source, target in rename_map.items() if source in cocktail.columns}
    )
    return enrich_fuel_cocktail_jpy_kwh(cocktail)


def load_japan_fuel_cocktail() -> pd.DataFrame:
    """Load daily Japan fuel cocktail prices."""
    path = _resolve_data_file("japan_fuel_cocktail.csv", "Japan/japan_fuel_cocktail.csv")
    return _normalize_fuel_cocktail(pd.read_csv(path))


def load_unit_production() -> pd.DataFrame:
    """Load and normalize HJKS unit production from CSV.
    
    Returns
    -------
    pd.DataFrame
        Columns: Plant Code, Market Area, Name, Unit Name, Type, Date,
        Last Update, 48 half-hour columns (00:30 … 24:00), daily_kwh.
        Half-hour values are in MW (converted from kWh).
    """
    from glob import glob
    
    # Mapping dictionaries for normalization
    COLUMNS_JP_ENG = {
        "エリア": "Market Area",
        "発電所コード": "Plant Code",
        "発電所名": "Name",
        "ユニット名": "Unit Name",
        "発電方式・燃種": "Type",
        "対象日": "Date",
        "日量[kWh]": "daily_kwh",
        "更新日時": "Last Update",
    }
    
    AREA_MAP_JP_EN = {
        "北海道": "Hokkaido",
        "東北": "Tohoku",
        "東京": "Tokyo",
        "中部": "Chubu",
        "北陸": "Hokuriku",
        "関西": "Kansai",
        "中国": "Chugoku",
        "四国": "Shikoku",
        "九州": "Kyushu",
        "沖縄": "Okinawa",
    }
    
    FUEL_MAP_JP_EN = {
        "水力": "Hydro",
        "火力（ガス）": "LNG",
        "火力（石炭）": "Coal",
        "火力（石油）": "Oil",
        "火力（ＬＮＧ）": "LNG",
        "火力（その他）": "Other_Thermal",
        "原子力": "Nuclear",
        "太陽光": "Solar",
        "風力": "Wind",
        "地熱": "Geothermal",
        "バイオマス": "Biomass",
        "その他": "Other",
    }
    
    # Find latest CSV file
    for base in [_WINDOWS_DATA_ROOT, _DATA_ROOT]:
        pattern = str(base / "HJKS" / "unit30min" / "ユニット別発電実績*.csv")
        matches = sorted(glob(pattern), reverse=True)
        if matches:
            df = pd.read_csv(matches[0])
            
            # Rename columns to English
            df = df.rename(columns=COLUMNS_JP_ENG)
            
            # Convert area and fuel type names
            df["Market Area"] = df["Market Area"].map(AREA_MAP_JP_EN)
            df["Type"] = df["Type"].map(FUEL_MAP_JP_EN)
            
            # Convert 30-min kWh columns to MW (kWh / 2 = MWh for 30min slot)
            time_cols = [col for col in df.columns if col.endswith("[kWh]")]
            for col in time_cols:
                clean_col = col.replace("[kWh]", "")
                df[clean_col] = df[col] / 2.0  # Convert 30-min kWh to MW
                df = df.drop(columns=[col])
            
            # Convert daily_kwh to MWh
            if "daily_kwh" in df.columns:
                df["daily_kwh"] = df["daily_kwh"] / 1000.0
            
            # Parse date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            
            # Drop Okinawa as per jpPublicData convention
            df = df[df["Market Area"] != "Okinawa"].copy()
            
            return df.reset_index(drop=True)
    
    raise FileNotFoundError(f"No HJKS unit production CSV files found in {_WINDOWS_DATA_ROOT / 'HJKS' / 'unit30min'} or {_DATA_ROOT / 'HJKS' / 'unit30min'}")


def load_thermal_efficiency_registry() -> pd.DataFrame:
    """Load thermal plant efficiency registry from MCP-catalogued Excel file."""
    path = None
    
    # Try MCP paths first
    for base in [_WINDOWS_DATA_ROOT, _DATA_ROOT]:
        for filename in ["plant_efficiency.xlsx", "thermalRegistry.xlsx", "thermal_efficiency_registry.csv"]:
            candidate = base / "Japan" / filename
            if candidate.exists():
                path = candidate
                break
        if path:
            break
    
    if not path:
        raise FileNotFoundError("Could not find thermal efficiency registry in Japan/ folder")
    
    if path.suffix == ".xlsx":
        excel_file = pd.ExcelFile(path)
        sheet = next(
            (s for s in ["Efficiency_v2", "Efficiency", "thermal_stack"] if s in excel_file.sheet_names),
            excel_file.sheet_names[0]
        )
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def load_nuclear_registry() -> pd.DataFrame:
    """Load nuclear plant registry from MCP-catalogued Excel file."""
    path = None
    
    # Try MCP paths first
    for base in [_WINDOWS_DATA_ROOT, _DATA_ROOT]:
        for filename in ["japan_nuclear_registry.xlsx", "nuclearRegistry.xlsx", "nuclear_registry.csv"]:
            candidate = base / "Japan" / filename
            if candidate.exists():
                path = candidate
                break
        if path:
            break
    
    if not path:
        raise FileNotFoundError("Could not find nuclear registry in Japan/ folder")
    
    if path.suffix == ".xlsx":
        excel_file = pd.ExcelFile(path)
        sheet = next(
            (s for s in ["nuclear_registry", "Nuclear"] if s in excel_file.sheet_names),
            excel_file.sheet_names[0]
        )
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def load_outages() -> pd.DataFrame:
    """Load HJKS plant outage events from latest timestamped file."""
    from glob import glob
    
    COLUMNS_JP_ENG = {
        "エリア": "Market Area",
        "発電事業者": "Operator",
        "発電所コード": "Plant Code",
        "発電所名": "Name",
        "発電形式": "Type",
        "ユニット名": "Unit Name",
        "認可出力": "Capacity (GW)",
        "停止区分": "StopType",
        "種別": "OutageDetail",
        "低下量": "Impact (GW)",
        "停止日時": "From",
        "復旧見通し": "Perspective",
        "復旧予定日": "To",
        "停止原因": "Reason",
        "最終更新日時": "Publish Time",
    }
    
    AREA_MAP_JP_EN = {
        "北海道": "Hokkaido",
        "東北": "Tohoku",
        "東京": "Tokyo",
        "中部": "Chubu",
        "北陸": "Hokuriku",
        "関西": "Kansai",
        "中国": "Chugoku",
        "四国": "Shikoku",
        "九州": "Kyushu",
        "沖縄": "Okinawa",
    }
    
    FUEL_MAP_JP_EN = {
        "水力": "Hydro",
        "火力（ガス）": "LNG",
        "火力（石炭）": "Coal",
        "火力（石油）": "Oil",
        "火力（ＬＮＧ）": "LNG",
        "火力（その他）": "Other_Thermal",
        "原子力": "Nuclear",
        "太陽光": "Solar",
        "風力": "Wind",
        "地熱": "Geothermal",
        "バイオマス": "Biomass",
        "その他": "Other",
    }
    
    # Find latest outages file
    for base in [_WINDOWS_DATA_ROOT, _DATA_ROOT]:
        pattern = str(base / "HJKS" / "outages" / "outages_*.csv")
        matches = sorted(glob(pattern), reverse=True)
        if matches:
            # Try different encodings
            for encoding in ["cp932", "shift-jis", "utf-8"]:
                try:
                    df = pd.read_csv(matches[0], encoding=encoding)
                    
                    # Rename columns to English
                    df = df.rename(columns=COLUMNS_JP_ENG)
                    
                    # Convert area and fuel type names
                    if "Market Area" in df.columns:
                        df["Market Area"] = df["Market Area"].map(AREA_MAP_JP_EN)
                    if "Type" in df.columns:
                        df["Type"] = df["Type"].map(FUEL_MAP_JP_EN)
                    
                    # Parse dates
                    for date_col in ["From", "To", "Publish Time"]:
                        if date_col in df.columns:
                            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    
                    return df
                except (UnicodeDecodeError, LookupError):
                    continue
            raise ValueError(f"Could not decode {matches[0]} with any supported encoding")
    
    raise FileNotFoundError(f"No HJKS outage files found in {_WINDOWS_DATA_ROOT / 'HJKS' / 'outages'} or {_DATA_ROOT / 'HJKS' / 'outages'}")
