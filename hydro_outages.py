"""Parse and consolidate HJKS hydro outage events."""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import pandas as pd

from hydro_parser import (
    AREA_MAP_JP_EN,
    enrich_hjks_with_official,
    load_hjks_hydro_units,
    parse_hydro_pdfs,
    plant_name_key,
    resolve_plant_name_key,
    unit_digit,
)

_DATA_ROOT = Path(__file__).resolve().parent
_WINDOWS_DATA_ROOT = Path(r"C:\Develop\data")

OUTAGES_COLUMNS_JP_ENG = {
    "エリア": "Market Area",
    "発電事業者": "Operator",
    "発電所コード": "Plant Code",
    "発電所名": "Name",
    "発電形式": "Type",
    "ユニット名": "Unit Name",
    "認可出力": "Capacity (kW)",
    "停止区分": "StopType",
    "種別": "OutageDetail",
    "低下量": "Impact (kW)",
    "停止日時": "From",
    "復旧見通し": "Perspective",
    "復旧予定日": "To",
    "停止原因": "Reason",
    "最終更新日時": "Publish Time",
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

OUTAGE_DETAIL_CANONICAL = {
    "停止・定期検査等": "停止・定期検査等",
    "停止・設備故障": "停止・設備故障",
    "停止・送電線等制約": "停止・送電線等制約",
    "停止・燃料制約": "停止・燃料制約",
    "停止・その他": "停止・その他",
    "停止・長期計画停止": "停止・長期計画停止",
    "低下・設備故障": "低下・設備故障",
    "低下・送電線等制約": "低下・送電線等制約",
    "低下・燃料制約": "低下・燃料制約",
    "低下・その他": "低下・その他",
}

MARKET_IMPACT_MAP = {
    "停止・設備故障": "forced",
    "低下・設備故障": "forced",
    "停止・定期検査等": "planned",
    "停止・送電線等制約": "network",
    "低下・送電線等制約": "network",
    "停止・燃料制約": "fuel",
    "低下・燃料制約": "fuel",
    "停止・その他": "other",
    "低下・その他": "other",
    "停止・長期計画停止": "retired_or_mothballed",
}

REASON_CATEGORY_RULES: list[tuple[str, str]] = [
    ("試運転", "commissioning"),
    ("発電機作業", "generator_work"),
    ("系統作業", "grid_work"),
    ("送電", "grid_work"),
    ("点検", "inspection"),
    ("故障", "failure"),
    ("流入量", "flow_constraint"),
    ("運用制約", "operational_constraint"),
    ("調査", "investigation"),
]


def _resolve_outages_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    for base in [_DATA_ROOT, _WINDOWS_DATA_ROOT]:
        matches = sorted(glob(str(base / "outages_*.csv")), reverse=True)
        if matches:
            return Path(matches[0])
        matches = sorted(glob(str(base / "HJKS" / "outages" / "outages_*.csv")), reverse=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("No HJKS outages CSV found.")


def read_outages_csv(path: Path) -> pd.DataFrame:
    """Read outages CSV handling cp932 encoding and trailing-comma column shift."""
    for encoding in ("cp932", "shift-jis", "utf-8"):
        try:
            frame = pd.read_csv(path, encoding=encoding, index_col=False)
            frame = frame.loc[:, ~frame.columns.astype(str).str.contains("^Unnamed", regex=True)]
            return frame
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Could not decode outages file: {path}")


def fix_sentinel_year(timestamp: pd.Timestamp, publish: pd.Timestamp) -> pd.Timestamp:
    """Replace MLIT 9999 year placeholder using publish timestamp."""
    if pd.isna(timestamp) or timestamp.year < 9000 or pd.isna(publish):
        return timestamp
    candidate = timestamp.replace(year=int(publish.year))
    if candidate < publish:
        candidate = timestamp.replace(year=int(publish.year) + 1)
    return candidate


def end_of_calendar_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Treat date-only midnight timestamps as end of calendar day."""
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        return timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return timestamp


def resolve_open_ended_to(from_ts: pd.Timestamp, reason: object) -> pd.Timestamp:
    """Infer end time for outages without 復旧予定日."""
    if pd.isna(from_ts):
        return pd.NaT
    reason_text = str(reason or "")
    if any(keyword in reason_text for keyword in ("流入量", "発電制約", "運用制約", "調査中", "調査")):
        return end_of_calendar_day(from_ts)
    return end_of_calendar_day(from_ts)


def normalize_outage_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix sentinel years and infer missing end timestamps."""
    outages = frame.copy()
    for column in ("From", "To", "Publish Time"):
        outages[column] = pd.to_datetime(outages[column], errors="coerce")

    outages["From"] = [
        fix_sentinel_year(from_ts, publish)
        for from_ts, publish in zip(outages["From"], outages["Publish Time"])
    ]
    outages["To"] = [
        fix_sentinel_year(to_ts, publish) if pd.notna(to_ts) else pd.NaT
        for to_ts, publish in zip(outages["To"], outages["Publish Time"])
    ]

    outages["To"] = outages.apply(
        lambda row: (
            resolve_open_ended_to(row["From"], row["Reason"])
            if pd.isna(row["To"]) and str(row.get("Perspective", "")).strip() == "なし"
            else end_of_calendar_day(row["To"]) if pd.notna(row["To"]) else pd.NaT
        ),
        axis=1,
    )
    outages["from_sentinel_fixed"] = outages["Publish Time"].notna()
    outages["to_inferred"] = frame["To"].isna() & (outages["Perspective"].astype(str).str.strip() == "なし")
    outages["to_eod_adjusted"] = outages["To"].notna()
    return outages


def classify_reason_category(reason: object) -> str:
    text = str(reason or "")
    for keyword, category in REASON_CATEGORY_RULES:
        if keyword in text:
            return category
    return "other"


def classify_market_impact(row: pd.Series) -> str:
    if row.get("StopType") == "計画外停止":
        return "forced"
    if row.get("StopType") == "計画停止" and row.get("OutageDetail") == "停止・定期検査等":
        return "planned"
    reason = str(row.get("Reason") or "")
    if "発電機作業" in reason or "点検" in reason:
        return "planned"
    if "故障" in reason and row.get("StopType") == "計画外停止":
        return "forced"
    return MARKET_IMPACT_MAP.get(str(row.get("OutageDetail")), "other")


def consolidate_outages(df_outages: pd.DataFrame) -> pd.DataFrame:
    """Merge HJKS republications while preserving distinct overlapping events."""
    if df_outages.empty:
        return df_outages.copy()

    required = ["Name", "Unit Name", "OutageDetail", "From", "To", "Publish Time"]
    missing = [column for column in required if column not in df_outages.columns]
    if missing:
        raise ValueError(f"Outages missing columns: {missing}")

    outages = df_outages.sort_values(
        ["Name", "Unit Name", "OutageDetail", "From", "To"],
        ascending=[True, True, True, True, False],
    ).copy()

    keep_index: list[int] = []
    for _, group in outages.groupby(["Name", "Unit Name", "OutageDetail"], sort=False):
        cluster_end = pd.Timestamp.min
        cluster_rows: list[int] = []
        for index, row in group.iterrows():
            row_end = row["To"] if pd.notna(row["To"]) else pd.Timestamp.max
            if pd.isna(row["From"]):
                continue
            if row["From"] > cluster_end:
                if cluster_rows:
                    keep_index.append(_latest_publish_index(outages, cluster_rows))
                cluster_rows = [index]
                cluster_end = row_end
            else:
                cluster_rows.append(index)
                cluster_end = max(cluster_end, row_end)
        if cluster_rows:
            keep_index.append(_latest_publish_index(outages, cluster_rows))

    return outages.loc[keep_index].reset_index(drop=True)


def _latest_publish_index(df_outages: pd.DataFrame, indexes: list[int]) -> int:
    return max(
        indexes,
        key=lambda index: (
            df_outages.loc[index, "Publish Time"]
            if pd.notna(df_outages.loc[index, "Publish Time"])
            else pd.Timestamp.min
        ),
    )


def load_raw_outages(path: Path | None = None) -> pd.DataFrame:
    """Load latest HJKS outages CSV with English column names."""
    outages_path = _resolve_outages_path(path)
    frame = read_outages_csv(outages_path)
    frame = frame.rename(columns=OUTAGES_COLUMNS_JP_ENG)
    if "Market Area" in frame.columns:
        frame["Market Area"] = frame["Market Area"].map(AREA_MAP_JP_EN)
    if "Type" in frame.columns:
        frame["Type"] = frame["Type"].map(FUEL_MAP_JP_EN)
    return frame


def process_hydro_outages(
    path: Path | None = None,
    *,
    consolidate: bool = True,
    attach_registry: bool = True,
) -> pd.DataFrame:
    """Full hydro outage pipeline: parse, clean dates, deduplicate, enrich."""
    outages = load_raw_outages(path)
    outages = outages[outages["Type"] == "Hydro"].copy()
    outages["OutageDetail"] = outages["OutageDetail"].map(
        lambda value: OUTAGE_DETAIL_CANONICAL.get(value, value)
    )
    outages["plant_name_key"] = outages["Name"].map(plant_name_key)
    outages["registry_lookup_key"] = outages["Name"].map(resolve_plant_name_key)
    outages["unit_digit"] = outages["Unit Name"].map(unit_digit)
    outages["capacity_mw"] = pd.to_numeric(outages["Capacity (kW)"], errors="coerce") / 1000
    outages["impact_mw"] = pd.to_numeric(outages["Impact (kW)"], errors="coerce") / 1000

    outages = normalize_outage_timestamps(outages)
    outages["duration_days"] = (outages["To"] - outages["From"]).dt.total_seconds() / 86400
    outages["missing_to"] = outages["To"].isna()
    outages["missing_from"] = outages["From"].isna()
    outages["reason_category"] = outages["Reason"].map(classify_reason_category)
    outages["market_impact"] = outages.apply(classify_market_impact, axis=1)

    if consolidate:
        outages = consolidate_outages(outages)

    if attach_registry:
        official = parse_hydro_pdfs()
        hjks_units = load_hjks_hydro_units()
        registry = enrich_hjks_with_official(hjks_units, official)
        registry = registry.drop_duplicates(["plant_name_key", "unit_digit"])
        outages = outages.merge(
            registry[
                [
                    "plant_name_key",
                    "unit_digit",
                    "registry_lookup_key",
                    "hydro_class",
                    "dam_kind",
                    "initial_permit_date",
                    "current_permit_date",
                    "current_permit_expiry",
                    "permit_age_years",
                    "years_to_permit_expiry",
                ]
            ],
            on=["plant_name_key", "unit_digit"],
            how="left",
        )

    return outages.reset_index(drop=True)


if __name__ == "__main__":
    hydro_outages = process_hydro_outages()
    print(f"Hydro outages processed: {len(hydro_outages):,}")
    print(hydro_outages["hydro_class"].value_counts(dropna=False).to_string())
    print(hydro_outages["market_impact"].value_counts().to_string())
