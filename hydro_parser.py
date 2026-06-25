"""Parse official MLIT hydro plant PDFs and export hydro_registry.xlsx."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

_DATA_ROOT = Path(__file__).resolve().parent
_HYDRO_PDF_DIR = _DATA_ROOT / "hydro"
_HJKS_HYDRO_PATH = _DATA_ROOT / "hydro.xlsx"
_REGISTRY_EXPORT_PATH = _DATA_ROOT / "hydro_registry.xlsx"

REGION_FROM_PDF = {
    "hokkaido": "Hokkaido",
    "tohoku": "Tohoku",
    "chubu": "Chubu",
    "kansai": "Kansai",
    "chugoku": "Chugoku",
    "shikoku": "Shikoku",
    "kyushu": "Kyushu",
}

ERA_YEAR_OFFSET = {
    "M": 1867,
    "T": 1911,
    "S": 1925,
    "H": 1988,
    "R": 2018,
}

# 型式 → market class: pmp, ror, hydro, WRE
TYPE_CODE_TO_CLASS = {
    "揚": "pmp",
    "貯・揚": "pmp",
    "流込": "ror",
    "流入": "ror",
    "流込貯": "ror",
    "貯": "hydro",
    "調": "WRE",
    "調整": "WRE",
    "従属": "WRE",
    "注水": "WRE",
}

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

COLUMNS_JP_ENG = {
    "エリア": "Market Area",
    "発電事業者": "Operator",
    "発電所コード": "Plant Code",
    "発電所名": "Name",
    "発電形式": "Type",
    "ユニット名": "Unit Name",
    "認可出力": "Capacity (kW)",
    "認可出力（変更後）": "Capacity Revised (kW)",
    "適用開始日": "Effective From",
    "稼働開始日": "Commercial Operation",
    "稼働終了日": "Retired Date",
    "最終更新日時": "Last Update",
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

# HJKS plant_name_key → official registry plant_name_key
PLANT_KEY_ALIASES = {
    "奥美濃": "奥美濃水力",
    "北陸電力有峰第一": "有峰第一",
    "北陸電力有峰第二": "有峰第二",
    "四国電力本川": "本川",
    "電源開発新豊根（中部）": "新豊根",
    "電源開発池原（関西）": "池原",
    "電源開発田子倉(東北)": "田子倉",
    "電源開発長野（北陸）": "長野",
}

# HJKS-only plants missing from the 2010 official database
MANUAL_HJKS_PLANT_META: dict[str, dict[str, object]] = {
    "大河内": {
        "hydro_class": "pmp",
        "type_code": "揚",
        "dam_kind": "発電ダム",
        "operator": "関西電力",
        "region": "Kansai",
    },
    "小千谷第二": {"hydro_class": "hydro", "type_code": "貯", "region": "Hokuriku"},
    "高見": {"hydro_class": "hydro", "type_code": "貯", "region": "Kyushu"},
    "新冠": {"hydro_class": "hydro", "type_code": "貯", "region": "Hokkaido"},
}


def nkc(text: object) -> str:
    """Normalize unicode width for Japanese text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).translate(_FULLWIDTH_DIGITS).strip()


def dedupe_doubled_chars(text: object) -> str:
    """Fix pdfplumber doubled glyphs (e.g. 玉玉原原 → 玉原)."""
    value = nkc(text)
    if not value:
        return ""
    out: list[str] = []
    index = 0
    while index < len(value):
        if index + 1 < len(value) and value[index] == value[index + 1]:
            out.append(value[index])
            index += 2
        else:
            out.append(value[index])
            index += 1
    return "".join(out)


def unit_digit(unit_str: object) -> str:
    """Extract unit number(s); default to '1' when absent."""
    if unit_str is None or (isinstance(unit_str, float) and pd.isna(unit_str)):
        return "1"
    numbers = re.findall(r"\d+", nkc(unit_str))
    return "-".join(numbers) if numbers else "1"


def plant_name_key(name: object) -> str:
    """Normalize plant name for joins (strip operator prefix and spaces)."""
    value = dedupe_doubled_chars(name)
    value = re.sub(
        r"^(電源開発|北陸電力|四国電力|東京電力|関西電力|中部電力|九州電力|北海道電力|中国電力|東北電力)\s*",
        "",
        value,
    )
    value = re.sub(r"（[^）]+）", "", value)
    value = re.sub(r"\([^)]+\)", "", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"発電所$", "", value)
    return value


def resolve_plant_name_key(name: object) -> str:
    """Return registry lookup key, applying known HJKS aliases."""
    key = plant_name_key(name)
    return PLANT_KEY_ALIASES.get(key, key)


def lookup_official_plant(official: pd.DataFrame, plant_key: str) -> pd.Series | None:
    """Find best official plant row for a normalized plant key."""
    if official.empty:
        return None
    exact = official[official["plant_name_key"] == plant_key]
    if not exact.empty:
        return exact.sort_values("capacity_kw", ascending=False).iloc[0]
    alias_key = PLANT_KEY_ALIASES.get(plant_key)
    if alias_key:
        alias = official[official["plant_name_key"] == alias_key]
        if not alias.empty:
            return alias.sort_values("capacity_kw", ascending=False).iloc[0]
    contains = official[official["plant_name_key"].str.contains(plant_key, na=False, regex=False)]
    if not contains.empty:
        return contains.sort_values("capacity_kw", ascending=False).iloc[0]
    return None


def parse_numeric(value: object) -> float | None:
    """Parse numeric cells with commas and PDF artefacts."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = dedupe_doubled_chars(value)
    text = text.replace(",", "")
    text = re.sub(r"[^\d.]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_japanese_era_date(value: object) -> pd.Timestamp:
    """Parse MLIT era dates such as H04.04.01 or S41.12.15."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    text = dedupe_doubled_chars(value)
    match = re.search(r"([MTSHR])(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return pd.NaT
    era, year, month, day = match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))
    western_year = ERA_YEAR_OFFSET[era] + year
    try:
        return pd.Timestamp(year=western_year, month=month, day=day)
    except ValueError:
        return pd.NaT


def map_type_code_to_class(type_code: object) -> str:
    """Map 型式 code to pmp / ror / hydro / WRE."""
    code = dedupe_doubled_chars(type_code)
    if code in TYPE_CODE_TO_CLASS:
        return TYPE_CODE_TO_CLASS[code]
    if "揚" in code:
        return "pmp"
    if "流" in code:
        return "ror"
    if "貯" in code:
        return "hydro"
    if "調" in code:
        return "WRE"
    return "hydro"


def _region_from_pdf_name(pdf_name: str) -> str | None:
    for token, region in REGION_FROM_PDF.items():
        if token in pdf_name.lower():
            return region
    return None


def _is_data_row(row: list[object | None]) -> bool:
    if not row or not row[0]:
        return False
    first = str(row[0]).strip()
    return first.isdigit()


def parse_hydro_pdfs(pdf_dir: Path | None = None) -> pd.DataFrame:
    """Parse all official hydro PDF tables under hydro/."""
    directory = _HYDRO_PDF_DIR if pdf_dir is None else Path(pdf_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Hydro PDF directory not found: {directory}")

    records: list[dict[str, object]] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        region = _region_from_pdf_name(pdf_path.name)
        with pdfplumber.open(pdf_path) as document:
            for page in document.pages:
                for table in page.extract_tables() or []:
                    for row in table[3:]:
                        if not _is_data_row(row):
                            continue
                        plant_name = dedupe_doubled_chars(row[4])
                        if not plant_name:
                            continue
                        type_code = dedupe_doubled_chars(row[5])
                        capacity_kw = parse_numeric(row[7])
                        records.append(
                            {
                                "source_pdf": pdf_path.name,
                                "region": region,
                                "prefecture": dedupe_doubled_chars(row[1]),
                                "river_system": dedupe_doubled_chars(row[2]),
                                "river": dedupe_doubled_chars(row[3]),
                                "plant_name_jp": plant_name,
                                "plant_name_key": plant_name_key(plant_name),
                                "type_code": type_code,
                                "hydro_class": map_type_code_to_class(type_code),
                                "max_flow_m3s": parse_numeric(row[6]),
                                "capacity_kw": capacity_kw,
                                "capacity_mw": capacity_kw / 1000 if capacity_kw is not None else None,
                                "peak_power": dedupe_doubled_chars(row[8]) == "○",
                                "operator": dedupe_doubled_chars(row[9]),
                                "source_dam": dedupe_doubled_chars(row[10]),
                                "dam_kind": dedupe_doubled_chars(row[12]),
                                "initial_permit_date": parse_japanese_era_date(row[23]),
                                "current_permit_date": parse_japanese_era_date(row[24]),
                                "current_permit_expiry": parse_japanese_era_date(row[25]),
                            }
                        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    today = pd.Timestamp.today().normalize()
    frame["permit_age_years"] = (today - frame["initial_permit_date"]).dt.days / 365.25
    frame["current_permit_tenure_years"] = (today - frame["current_permit_date"]).dt.days / 365.25
    frame["years_to_permit_expiry"] = (frame["current_permit_expiry"] - today).dt.days / 365.25
    frame["database_as_of"] = pd.Timestamp("2010-03-31")
    return frame.sort_values(["region", "plant_name_jp"]).reset_index(drop=True)


def load_hjks_hydro_units(path: Path | None = None) -> pd.DataFrame:
    """Load and normalize HJKS hydro unit registry (hydro.xlsx)."""
    registry_path = _HJKS_HYDRO_PATH if path is None else Path(path)
    if not registry_path.exists():
        return pd.DataFrame()

    frame = pd.read_excel(registry_path)
    frame = frame.rename(columns=COLUMNS_JP_ENG)
    frame["Market Area"] = frame["Market Area"].map(AREA_MAP_JP_EN)
    frame["plant_name_key"] = frame["Name"].map(plant_name_key)
    frame["unit_digit"] = frame["Unit Name"].map(unit_digit)
    frame["capacity_mw"] = pd.to_numeric(frame["Capacity (kW)"], errors="coerce") / 1000
    frame["Commercial Operation"] = pd.to_datetime(frame["Commercial Operation"], errors="coerce")
    frame["Retired Date"] = pd.to_datetime(frame["Retired Date"], errors="coerce")
    frame["Last Update"] = pd.to_datetime(frame["Last Update"], errors="coerce")
    return frame


def enrich_hjks_with_official(hjks_units: pd.DataFrame, official_plants: pd.DataFrame) -> pd.DataFrame:
    """Attach official hydro metadata to HJKS unit rows."""
    if hjks_units.empty:
        return hjks_units.copy()

    enriched = hjks_units.copy()
    enriched["registry_lookup_key"] = enriched["Name"].map(resolve_plant_name_key)

    attach_columns = [
        "hydro_class",
        "type_code",
        "dam_kind",
        "initial_permit_date",
        "current_permit_date",
        "current_permit_expiry",
        "permit_age_years",
        "current_permit_tenure_years",
        "years_to_permit_expiry",
        "peak_power",
        "river_system",
        "river",
        "source_dam",
        "prefecture",
        "database_as_of",
    ]
    for column in attach_columns:
        enriched[column] = pd.NA

    for index, row in enriched.iterrows():
        lookup_key = row["registry_lookup_key"]
        official_row = lookup_official_plant(official_plants, lookup_key) if not official_plants.empty else None
        if official_row is not None:
            for column in attach_columns:
                enriched.at[index, column] = official_row.get(column, pd.NA)
            continue
        manual = MANUAL_HJKS_PLANT_META.get(lookup_key)
        if manual:
            for column, value in manual.items():
                if column in enriched.columns or column in attach_columns:
                    enriched.at[index, column] = value

    return enriched


def build_hydro_registry(
    pdf_dir: Path | None = None,
    hjks_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build official plant table and HJKS unit table enriched with permit metadata."""
    official = parse_hydro_pdfs(pdf_dir)
    hjks_units = load_hjks_hydro_units(hjks_path)
    hjks_enriched = enrich_hjks_with_official(hjks_units, official)
    return official, hjks_enriched


def export_hydro_registry(
    output_path: Path | None = None,
    pdf_dir: Path | None = None,
    hjks_path: Path | None = None,
) -> Path:
    """Export hydro_registry.xlsx with official and HJKS sheets."""
    destination = _REGISTRY_EXPORT_PATH if output_path is None else Path(output_path)
    official, hjks_enriched = build_hydro_registry(pdf_dir=pdf_dir, hjks_path=hjks_path)

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        official.to_excel(writer, sheet_name="official_plants", index=False)
        if not hjks_enriched.empty:
            hjks_enriched.to_excel(writer, sheet_name="hjks_units", index=False)

        summary = pd.DataFrame(
            {
                "metric": [
                    "official_plant_count",
                    "hjks_unit_count",
                    "hjks_matched_to_official",
                    "pmp_count",
                    "ror_count",
                    "hydro_count",
                    "WRE_count",
                ],
                "value": [
                    len(official),
                    len(hjks_enriched),
                    int(hjks_enriched["hydro_class"].notna().sum()) if not hjks_enriched.empty else 0,
                    int((official["hydro_class"] == "pmp").sum()) if not official.empty else 0,
                    int((official["hydro_class"] == "ror").sum()) if not official.empty else 0,
                    int((official["hydro_class"] == "hydro").sum()) if not official.empty else 0,
                    int((official["hydro_class"] == "WRE").sum()) if not official.empty else 0,
                ],
            }
        )
        summary.to_excel(writer, sheet_name="summary", index=False)

    return destination


if __name__ == "__main__":
    path = export_hydro_registry()
    official, hjks = build_hydro_registry()
    print(f"Exported {path}")
    print(f"  official_plants: {len(official):,}")
    print(f"  hjks_units: {len(hjks):,}")
    if not hjks.empty:
        print(f"  hjks matched: {hjks['hydro_class'].notna().sum():,} / {len(hjks):,}")
