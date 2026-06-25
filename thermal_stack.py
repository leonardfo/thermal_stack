"""Build the Japan thermal stack from MCP-catalogued datasets.

Auteur : Fourneret Leonard
Date de creation ou modification : 2026-05-12
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import data_loader

FAR_FUTURE = pd.Timestamp("2100-01-01")
EAST_JAPAN_AREAS = ("Hokkaido", "Tohoku", "Tokyo")
WEST_JAPAN_AREAS = ("Hokuriku", "Kansai", "Chugoku", "Shikoku", "Kyushu", "Chubu")
DEFAULT_PRODUCTION_TYPES = ("LNG", "Coal", "Hydro", "Oil", "Nuclear", "Biomass")
DEFAULT_STACK_FUELS = ("LNG", "Coal", "Oil", "Nuclear")

MMBTU_TO_MWH = 0.293071
DEFAULT_EFFICIENCY_BASIS = "LHV"
DEFAULT_FUEL_PRICE_BASIS = "HHV"
HHV_PER_LHV_BY_FUEL = {
    "LNG": 1.108,
    "Coal": 1.055,
    "Oil": 1.060,
}
VOM_USD_PER_MWH = {
    "LNG": 3.0,
    "Coal": 4.0,
    "Oil": 6.0,
    "Nuclear": 7.0,
}

OUTAGE_DETAIL_MAPPING = {
    "停止・定期検査等": {"severity": "full", "driver": "maintenance", "family": "Planned Maintenance"},
    "停止・長期計画停止": {"severity": "full", "driver": "maintenance", "family": "Planned Shutdown"},
    "停止・設備故障": {"severity": "full", "driver": "technical", "family": "Equipment Failure"},
    "停止・送電線等制約": {"severity": "full", "driver": "grid", "family": "Grid Constraint"},
    "停止・燃料制約": {"severity": "full", "driver": "fuel", "family": "Fuel Constraint"},
    "停止・その他": {"severity": "full", "driver": "other", "family": "Other"},
    "低下・設備故障": {"severity": "partial", "driver": "technical", "family": "Equipment Failure"},
    "低下・送電線等制約": {"severity": "partial", "driver": "grid", "family": "Grid Constraint"},
    "低下・燃料制約": {"severity": "partial", "driver": "fuel", "family": "Fuel Constraint"},
    "低下・その他": {"severity": "partial", "driver": "other", "family": "Other"},
}

MERIT_ORDER_COLORS = {
    "Nuclear": "#9467bd",
    "Coal": "#333333",
    "LNG": "#ff8c00",
    "Oil": "#8b4513",
}

REGION_SPOT_COLUMNS = {
    "Hokkaido": "jpHokk",
    "Tohoku": "jpToh",
    "Tokyo": "jpTok",
    "Chubu": "jpChub",
    "Hokuriku": "jpHoku",
    "Kansai": "jpKan",
    "Chugoku": "jpChug",
    "Shikoku": "jpShi",
    "Kyushu": "jpKyu",
}

GENERATION_COLUMNS_BY_FUEL = {
    "Coal": "gen_coa_act",
    "LNG": "gen_gas_act",
    "Oil": "gen_oil_act",
    "Nuclear": "gen_nuc_act",
    "Hydro": "gen_hyd_act",
}

DEMAND_BASIS_CONSUMPTION = "consumption"
DEMAND_BASIS_RESIDUAL_LOAD = "residual_load"
DEMAND_BASIS_THERMAL_RESIDUAL = "thermal_residual"
DEMAND_BASIS_CHOICES = (
    DEMAND_BASIS_CONSUMPTION,
    DEMAND_BASIS_RESIDUAL_LOAD,
    DEMAND_BASIS_THERMAL_RESIDUAL,
)

KANSAI_FENCE_AREAS = ("Hokuriku", "Kansai", "Chubu")

from dataclasses import dataclass

@dataclass(frozen=True)
class ThermalStackConfig:
    """Configuration for the thermal stack pipeline."""
    production_start: str | pd.Timestamp = "2026-04-01"
    outage_start: str | pd.Timestamp = "2026-03-15"
    forecast_start: str | pd.Timestamp | None = None
    delivery_month: str | pd.Timestamp | None = None
    export_excel: bool = True
    export_dataset_key: str = data_loader.THERMAL_STACK_EXPORT_DATASET_KEY
    load_unit_production: bool = True
    nuclear_efficiency_pct: float = 35.1
    outage_min_duration_days: float = 0.5
    excluded_areas: tuple[str, ...] = ("Okinawa",)
    production_types: tuple[str, ...] = DEFAULT_PRODUCTION_TYPES
    stack_fuels: tuple[str, ...] = DEFAULT_STACK_FUELS
    use_adjusted_capacity: bool = True


@dataclass(frozen=True)
class ThermalStackResult:
    """Container returned by build_thermal_stack()."""
    merit_order: pd.DataFrame
    marginal_costs: pd.DataFrame
    active_units: pd.DataFrame
    adjusted_units: pd.DataFrame
    thermal_registry: pd.DataFrame
    nuclear_registry: pd.DataFrame
    outages_thermal: pd.DataFrame
    outages_nuclear: pd.DataFrame
    outages_all: pd.DataFrame
    fuel_cocktail: pd.DataFrame
    unit_production: pd.DataFrame
    delivery_month: pd.Timestamp
    capacity_summary: dict
    export_path: Path | None


@dataclass(frozen=True)
class PriceSetterConfig:
    """One price-setter scenario: region grouping, spot reference and demand basis."""
    name: str
    areas: tuple[str, ...]
    spot_area: str
    demand_basis: str = DEMAND_BASIS_CONSUMPTION

    def __post_init__(self) -> None:
        if self.demand_basis not in DEMAND_BASIS_CHOICES:
            raise ValueError(
                f"Unsupported demand_basis {self.demand_basis!r}. "
                f"Expected one of {DEMAND_BASIS_CHOICES}."
            )
        if self.spot_area not in REGION_SPOT_COLUMNS:
            raise ValueError(f"Unknown spot_area {self.spot_area!r}.")
        for area in self.areas:
            if area not in REGION_SPOT_COLUMNS:
                raise ValueError(f"Unknown market area {area!r}.")


def default_price_setter_configs(
    include_kansai_fence: bool = True,
    include_tokyo: bool = False,
    include_chubu: bool = False,
    include_hokuriku: bool = False,
    include_kansai: bool = False,
) -> list[PriceSetterConfig]:
    """Return default parallel price-setter configurations."""
    region_groups: list[tuple[str, tuple[str, ...], str]] = [
        ("Kansai", ("Kansai",), "Kansai"),
    ]
    if include_kansai_fence:
        region_groups.append(("Kansai+Hokuriku+Chubu", KANSAI_FENCE_AREAS, "Kansai"))
    if include_tokyo:
        region_groups.append(("Tokyo", ("Tokyo",), "Tokyo"))

    configs: list[PriceSetterConfig] = []
    for label, areas, spot_area in region_groups:
        for demand_basis in DEMAND_BASIS_CHOICES:
            configs.append(
                PriceSetterConfig(
                    name=f"{label} | {demand_basis}",
                    areas=areas,
                    spot_area=spot_area,
                    demand_basis=demand_basis,
                )
            )
    return configs

def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str], frame_name: str) -> None:
    """Raise error when required columns missing."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _timestamp(value: str | pd.Timestamp | None, default: pd.Timestamp | None = None) -> pd.Timestamp:
    """Convert value to pd.Timestamp."""
    if value is None:
        if default is None:
            raise ValueError("A timestamp value is required when no default is provided.")
        return pd.Timestamp(default)
    return pd.Timestamp(value)


def _month_start(value: str | pd.Timestamp | None) -> pd.Timestamp:
    """Return first day of requested month."""
    timestamp = pd.Timestamp("today").normalize() if value is None else pd.Timestamp(value)
    return timestamp.to_period("M").to_timestamp()


def classify_region(area: object) -> str | float:
    """Classify Japan market area into East or West."""
    if area in EAST_JAPAN_AREAS:
        return "East"
    if area in WEST_JAPAN_AREAS:
        return "West"
    return np.nan


def _add_east_west(frame: pd.DataFrame, area_column: str) -> pd.DataFrame:
    """Add EastWest column from market-area column."""
    result = frame.copy()
    if area_column in result.columns:
        result["EastWest"] = result[area_column].apply(classify_region)
    return result

def load_filtered_unit_production(config: ThermalStackConfig) -> pd.DataFrame:
    """Load and filter HJKS unit production through data_loader."""
    if not config.load_unit_production:
        return pd.DataFrame()

    unit_production = data_loader.load_unit_production()
    _require_columns(unit_production, ["Date", "Type", "Market Area"], "unit_production")
    unit_production = unit_production.copy()
    unit_production["Date"] = pd.to_datetime(unit_production["Date"], errors="coerce")
    unit_production = unit_production[unit_production["Type"].isin(config.production_types)]
    unit_production = unit_production[
        unit_production["Date"] >= _timestamp(config.production_start)
    ].copy()
    return _add_east_west(unit_production, "Market Area").reset_index(drop=True)


def load_thermal_registry(config: ThermalStackConfig) -> pd.DataFrame:
    """Load and filter thermal efficiency registry."""
    thermal = data_loader.load_thermal_efficiency_registry().copy()
    _require_columns(
        thermal,
        [
            "plant_name_jp",
            "unit_digit",
            "fuel_class",
            "area_en",
            "capacity_mw",
            "plant_efficiency",
            "start_year",
            "retired_year",
        ],
        "thermal_registry",
    )
    thermal["start_year"] = pd.to_datetime(thermal["start_year"], errors="coerce")
    thermal["retired_year"] = pd.to_datetime(thermal["retired_year"], errors="coerce").fillna(FAR_FUTURE)
    thermal = thermal[thermal["retired_year"] >= _timestamp(config.outage_start)]
    if config.excluded_areas:
        thermal = thermal[~thermal["area_en"].isin(config.excluded_areas)]
    thermal["Type"] = thermal["fuel_class"]
    return _add_east_west(thermal.reset_index(drop=True), "area_en")


def prepare_nuclear_registry(
    nuclear_registry: pd.DataFrame,
    thermal_columns: Iterable[str],
    config: ThermalStackConfig,
) -> pd.DataFrame:
    """Convert nuclear registry rows to thermal-stack schema."""
    nuclear = nuclear_registry.copy()
    _require_columns(
        nuclear,
        [
            "unitAll_code",
            "unit30min_code",
            "plant_name_jp",
            "unit_digit",
            "area_en",
            "capacity_gw",
            "start_date",
            "end_date",
        ],
        "nuclear_registry",
    )
    nuclear["capacity_mw"] = pd.to_numeric(nuclear["capacity_gw"], errors="coerce") * 1000
    nuclear["fuel_class"] = "Nuclear"
    nuclear["Type"] = "Nuclear"
    nuclear["plant_efficiency"] = config.nuclear_efficiency_pct
    nuclear["start_year"] = pd.to_datetime(nuclear["start_date"], errors="coerce")
    nuclear["retired_year"] = pd.to_datetime(nuclear["end_date"], errors="coerce").fillna(FAR_FUTURE)

    columns = [column for column in thermal_columns if column in nuclear.columns]
    for column in [
        "unitAll_code",
        "unit30min_code",
        "plant_name_jp",
        "unit_digit",
        "area_en",
        "capacity_mw",
        "fuel_class",
        "Type",
        "plant_efficiency",
        "start_year",
        "retired_year",
        "owner",
    ]:
        if column in nuclear.columns and column not in columns:
            columns.append(column)
    nuclear = nuclear[columns].copy()
    return _add_east_west(nuclear.reset_index(drop=True), "area_en")

def filter_units_for_date(df_units: pd.DataFrame, target_date: str | pd.Timestamp) -> pd.DataFrame:
    """Filter units active on one date."""
    _require_columns(df_units, ["start_year", "retired_year"], "df_units")
    target = pd.Timestamp(target_date)
    mask = (df_units["start_year"] <= target) & (df_units["retired_year"] >= target)
    return df_units.loc[mask].copy()


def filter_units_for_forecast(
    df_units: pd.DataFrame,
    forecast_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter units not retired at forecast start."""
    _require_columns(df_units, ["retired_year"], "df_units")
    start = pd.Timestamp("today").normalize() if forecast_start is None else pd.Timestamp(forecast_start)
    return df_units.loc[df_units["retired_year"] >= start].copy()

def consolidate_outages(df_outages: pd.DataFrame) -> pd.DataFrame:
    """Merge HJKS update duplicates while preserving concurrent events."""
    if df_outages.empty:
        return df_outages.copy()

    _require_columns(
        df_outages,
        ["Name", "Unit Name", "OutageDetail", "From", "To", "Publish Time"],
        "df_outages",
    )
    outages = df_outages.sort_values(
        ["Name", "Unit Name", "OutageDetail", "From", "To"],
        ascending=[True, True, True, True, False],
    ).copy()

    keep_index = []
    for _, group in outages.groupby(["Name", "Unit Name", "OutageDetail"], sort=False):
        cluster_end = pd.Timestamp.min
        cluster_rows: list[int] = []
        for index, row in group.iterrows():
            if row["From"] > cluster_end:
                if cluster_rows:
                    keep_index.append(_latest_publish_index(outages, cluster_rows))
                cluster_rows = [index]
                cluster_end = row["To"]
            else:
                cluster_rows.append(index)
                cluster_end = max(cluster_end, row["To"])
        if cluster_rows:
            keep_index.append(_latest_publish_index(outages, cluster_rows))

    return outages.loc[keep_index].reset_index(drop=True)


def _latest_publish_index(df_outages: pd.DataFrame, indexes: list[int]) -> int:
    """Return index with latest publish timestamp."""
    return max(
        indexes,
        key=lambda index: (
            df_outages.loc[index, "Publish Time"]
            if pd.notna(df_outages.loc[index, "Publish Time"])
            else pd.Timestamp.min
        ),
    )


def annotate_outage_metadata(df_outages: pd.DataFrame) -> pd.DataFrame:
    """Add severity, driver, family and impact columns to outages."""
    outages = df_outages.copy()
    if outages.empty:
        return outages

    mapped = outages["OutageDetail"].map(lambda value: OUTAGE_DETAIL_MAPPING.get(value, {}))
    outages["severity"] = mapped.map(lambda value: value.get("severity", "unknown"))
    outages["driver"] = mapped.map(lambda value: value.get("driver", "unknown"))
    outages["family"] = mapped.map(lambda value: value.get("family", "Unknown"))
    if "Impact (GW)" in outages.columns:
        outages["impact_mw"] = pd.to_numeric(outages["Impact (GW)"], errors="coerce") * 1000
    return outages

def load_stack_outages(config: ThermalStackConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare thermal and nuclear outages."""
    outages = data_loader.load_outages().copy()
    _require_columns(outages, ["Type", "From", "To", "OutageDetail", "Name", "Unit Name", "Publish Time"], "outages")
    outages["From"] = pd.to_datetime(outages["From"], errors="coerce")
    outages["To"] = pd.to_datetime(outages["To"], errors="coerce")
    outages["Publish Time"] = pd.to_datetime(outages.get("Publish Time"), errors="coerce")
    outages = outages[outages["To"] >= _timestamp(config.outage_start)].copy()
    outages["duration_days"] = (outages["To"] - outages["From"]).dt.total_seconds() / 86400
    outages = outages[outages["duration_days"] >= config.outage_min_duration_days].copy()

    thermal = outages[~outages["Type"].isin(["Nuclear", "Hydro"])].copy()
    nuclear = outages[outages["Type"] == "Nuclear"].copy()
    nuclear = nuclear[~nuclear["Name"].astype(str).str.contains("柏崎刈羽", na=False)].copy()

    thermal = annotate_outage_metadata(consolidate_outages(thermal))
    nuclear = annotate_outage_metadata(consolidate_outages(nuclear))
    outages_all = pd.concat([thermal, nuclear], ignore_index=True)
    return thermal, nuclear, outages_all

def _compute_loss_mw(outage: pd.Series, capacity_mw: float) -> float:
    """Compute MW loss for one outage event."""
    severity = str(outage.get("severity", "partial")).lower()
    if severity == "full":
        return capacity_mw

    impact_mw = outage.get("impact_mw")
    if pd.notna(impact_mw) and impact_mw > 0:
        return min(float(impact_mw), capacity_mw)

    driver = str(outage.get("driver", "")).lower()
    if driver == "grid":
        return 0.3 * capacity_mw
    if driver == "fuel":
        return 0.5 * capacity_mw
    if driver == "technical":
        return 0.6 * capacity_mw
    return 0.4 * capacity_mw


def _match_unit_outages(unit: pd.Series, active_outages: pd.DataFrame) -> pd.DataFrame:
    """Match active outages to unit using available registry fields."""
    if active_outages.empty:
        return active_outages

    matched = active_outages.iloc[0:0]
    unit_code = unit.get("unitAll_code", np.nan)
    if pd.notna(unit_code) and "Plant Code" in active_outages.columns:
        matched = active_outages[active_outages["Plant Code"].astype(str) == str(unit_code)]

    if matched.empty and "Name" in active_outages.columns and "plant_name_jp" in unit.index:
        matched = active_outages[active_outages["Name"].astype(str) == str(unit["plant_name_jp"])]

    if matched.empty:
        return matched

    if "Unit Name" in matched.columns and pd.notna(unit.get("unit_digit", np.nan)):
        unit_digit = str(unit["unit_digit"])
        exact_unit = matched[matched["Unit Name"].astype(str) == unit_digit]
        return exact_unit if not exact_unit.empty else matched
    return matched

def build_adjusted_capacity(
    df_units: pd.DataFrame,
    df_outages_all: pd.DataFrame,
    target_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Adjust unit capacity by active outages for target date."""
    _require_columns(df_units, ["capacity_mw", "start_year", "retired_year"], "df_units")
    target = pd.Timestamp(target_date)
    adjusted = filter_units_for_date(df_units, target).copy()
    adjusted["Capacity_adj_GW"] = pd.to_numeric(adjusted["capacity_mw"], errors="coerce") / 1000

    if df_outages_all.empty:
        active_outages = df_outages_all.copy()
    else:
        active_outages = df_outages_all[
            (df_outages_all["From"] <= target) & (df_outages_all["To"] >= target)
        ].copy()

    full_stops = 0
    derates = 0
    for unit_index, unit in adjusted.iterrows():
        capacity_mw = pd.to_numeric(unit["capacity_mw"], errors="coerce")
        if pd.isna(capacity_mw) or capacity_mw <= 0:
            continue
        capacity_mw = float(capacity_mw)
        unit_outages = _match_unit_outages(unit, active_outages)
        if unit_outages.empty:
            continue

        total_loss_mw = 0.0
        for _, outage in unit_outages.iterrows():
            total_loss_mw += _compute_loss_mw(outage, capacity_mw)
            if str(outage.get("severity", "partial")).lower() == "full":
                full_stops += 1
            else:
                derates += 1

        adjusted.loc[unit_index, "Capacity_adj_GW"] -= min(total_loss_mw, capacity_mw) / 1000

    adjusted["Capacity_adj_GW"] = adjusted["Capacity_adj_GW"].clip(lower=0)
    gw_before = (pd.to_numeric(adjusted["capacity_mw"], errors="coerce") / 1000).sum()
    gw_after = adjusted["Capacity_adj_GW"].sum()
    adjusted = adjusted[adjusted["Capacity_adj_GW"] > 0].copy()

    summary = {
        "target_date": target,
        "n_full_stops": full_stops,
        "n_derates": derates,
        "gw_removed": gw_before - gw_after,
        "gw_before": gw_before,
        "gw_after": gw_after,
        "n_units_before": len(filter_units_for_date(df_units, target)),
        "n_units_after": len(adjusted),
    }
    return adjusted, summary


def build_adjusted_capacity_month(
    df_units: pd.DataFrame,
    df_outages_all: pd.DataFrame,
    target_month: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Adjust unit capacity for month using mid-month date."""
    mid_month = _month_start(target_month) + pd.Timedelta(days=14)
    return build_adjusted_capacity(df_units, df_outages_all, mid_month)

def _normalize_energy_basis(basis: object, default_basis: str) -> str:
    """Return canonical heating-value basis label."""
    if pd.isna(basis) or basis is None:
        return default_basis

    normalized = str(basis).strip().upper()
    alias_map = {
        "HHV": "HHV",
        "GCV": "HHV",
        "GROSS": "HHV",
        "HIGHER HEATING VALUE": "HHV",
        "LHV": "LHV",
        "NCV": "LHV",
        "NET": "LHV",
        "LOWER HEATING VALUE": "LHV",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported heating-value basis: {basis!r}")
    return alias_map[normalized]


def convert_efficiency_to_energy_basis(
    gen_type: str,
    efficiency_pct: float,
    from_basis: object = DEFAULT_EFFICIENCY_BASIS,
    to_basis: object = DEFAULT_FUEL_PRICE_BASIS,
) -> float:
    """Convert plant efficiency between LHV and HHV conventions."""
    if pd.isna(efficiency_pct) or efficiency_pct <= 0:
        return np.nan

    source_basis = _normalize_energy_basis(from_basis, DEFAULT_EFFICIENCY_BASIS)
    target_basis = _normalize_energy_basis(to_basis, DEFAULT_FUEL_PRICE_BASIS)
    if source_basis == target_basis or gen_type == "Nuclear":
        return float(efficiency_pct)

    hhv_per_lhv = HHV_PER_LHV_BY_FUEL.get(gen_type)
    if hhv_per_lhv is None or hhv_per_lhv <= 0:
        return np.nan

    efficiency = float(efficiency_pct)
    if source_basis == "LHV" and target_basis == "HHV":
        return efficiency / hhv_per_lhv
    if source_basis == "HHV" and target_basis == "LHV":
        return efficiency * hhv_per_lhv
    raise ValueError(f"Unsupported energy-basis conversion: {source_basis!r} -> {target_basis!r}")

def compute_mc_eur(
    gen_type: str,
    efficiency_pct: float,
    fuel_row: pd.Series | dict,
    efficiency_basis: object = DEFAULT_EFFICIENCY_BASIS,
    fuel_basis: object = DEFAULT_FUEL_PRICE_BASIS,
) -> float:
    """Compute marginal cost in EUR/MWh for one fuel and efficiency."""
    if pd.isna(efficiency_pct) or efficiency_pct <= 0:
        return np.nan

    eurusd = fuel_row.get("EURUSD", fuel_row.get("eurusd", np.nan))
    eurjpy = fuel_row.get("EURJPY", fuel_row.get("eurjpy", np.nan))
    
    if pd.isna(eurusd) or eurusd <= 0:
        return np.nan

    if gen_type == "Nuclear":
        return VOM_USD_PER_MWH["Nuclear"] / eurusd

    if gen_type == "LNG":
        fuel = fuel_row.get("jlc_eurmwh", np.nan)
    elif gen_type == "Coal":
        fuel = fuel_row.get("coal_cif_eurmwh", fuel_row.get("newc_eurmwh", np.nan))
    elif gen_type == "Oil":
        fuel = fuel_row.get("jcc_eurmwh", np.nan)
    else:
        return np.nan

    if pd.isna(fuel):
        return np.nan

    aligned_efficiency_pct = convert_efficiency_to_energy_basis(
        gen_type,
        efficiency_pct,
        from_basis=efficiency_basis,
        to_basis=fuel_basis,
    )
    if pd.isna(aligned_efficiency_pct) or aligned_efficiency_pct <= 0:
        return np.nan

    efficiency = aligned_efficiency_pct / 100.0
    variable_om = VOM_USD_PER_MWH.get(gen_type, 0.0) / eurusd
    return fuel / efficiency + variable_om


def build_marginal_costs(df_active_units: pd.DataFrame, df_cocktail: pd.DataFrame) -> pd.DataFrame:
    """Build monthly marginal costs for every active stack unit."""
    _require_columns(
        df_active_units,
        ["plant_name_jp", "unit_digit", "fuel_class", "plant_efficiency"],
        "df_active_units",
    )
    if df_cocktail.empty:
        raise ValueError("fuel_cocktail is empty; marginal costs cannot be computed.")

    monthly_fuels = df_cocktail.sort_index().groupby(df_cocktail.index.to_period("M")).last().copy()
    monthly_fuels["delivery"] = monthly_fuels.index.to_timestamp(how="start")

    records = []
    for _, fuel_row in monthly_fuels.iterrows():
        delivery = pd.Timestamp(fuel_row["delivery"])
        for _, plant in df_active_units.iterrows():
            mc_eur_mwh = compute_mc_eur(
                plant["fuel_class"],
                plant["plant_efficiency"],
                fuel_row,
                efficiency_basis=plant.get("plant_efficiency_basis", plant.get("efficiency_basis", DEFAULT_EFFICIENCY_BASIS)),
                fuel_basis=plant.get("fuel_price_basis", DEFAULT_FUEL_PRICE_BASIS),
            )
            aligned_efficiency_pct = convert_efficiency_to_energy_basis(
                plant["fuel_class"],
                plant["plant_efficiency"],
                from_basis=plant.get("plant_efficiency_basis", plant.get("efficiency_basis", DEFAULT_EFFICIENCY_BASIS)),
                to_basis=plant.get("fuel_price_basis", DEFAULT_FUEL_PRICE_BASIS),
            )
            records.append(
                {
                    "delivery": delivery,
                    "plant_name_jp": plant["plant_name_jp"],
                    "unit_digit": plant["unit_digit"],
                    "fuel_class": plant["fuel_class"],
                    "mc_eur_mwh": mc_eur_mwh,
                    "efficiency_pct": plant["plant_efficiency"],
                    "efficiency_pct_mc_basis": aligned_efficiency_pct,
                    "efficiency_basis": plant.get("plant_efficiency_basis", plant.get("efficiency_basis", DEFAULT_EFFICIENCY_BASIS)),
                    "fuel_price_basis": plant.get("fuel_price_basis", DEFAULT_FUEL_PRICE_BASIS),
                }
            )

    return pd.DataFrame.from_records(records)

def build_merit_order(
    delivery_month: str | pd.Timestamp,
    df_mc: pd.DataFrame,
    df_units: pd.DataFrame,
    region: str | None = None,
    areas: str | list[str] | None = None,
    capacity_column: str = "capacity_mw",
) -> pd.DataFrame:
    """Build sorted merit order for one delivery month."""
    _require_columns(df_units, ["plant_name_jp", "unit_digit", "fuel_class", capacity_column], "df_units")
    _require_columns(df_mc, ["delivery", "plant_name_jp", "unit_digit", "fuel_class", "mc_eur_mwh"], "df_mc")

    delivery = _month_start(delivery_month)
    mc_month = df_mc[df_mc["delivery"] == delivery].copy()
    mc_columns = ["plant_name_jp", "unit_digit", "fuel_class", "mc_eur_mwh", "efficiency_pct"]
    mc_columns = [column for column in mc_columns if column in mc_month.columns]

    merged = df_units.merge(
        mc_month[mc_columns],
        on=["plant_name_jp", "unit_digit", "fuel_class"],
        how="left",
    )
    if areas is not None:
        _require_columns(merged, ["area_en"], "df_units")
        area_list = [areas] if isinstance(areas, str) else list(areas)
        merged = merged[merged["area_en"].isin(area_list)]
    elif region is not None:
        if "EastWest" not in merged.columns:
            merged = _add_east_west(merged, "area_en")
        merged = merged[merged["EastWest"] == region]

    merged["capacity_mw"] = pd.to_numeric(merged[capacity_column], errors="coerce")
    merged = merged.dropna(subset=["mc_eur_mwh"]).copy()
    merged = merged.dropna(subset=["capacity_mw"]).copy()
    merged = merged[merged["capacity_mw"] > 0].copy()
    merged = merged.sort_values("mc_eur_mwh").reset_index(drop=True)
    merged["cum_GW"] = merged["capacity_mw"].cumsum() / 1000
    return merged


def merit_order_for_areas(
    result: ThermalStackResult,
    areas: str | list[str] | None = None,
    use_adjusted_capacity: bool | None = None,
) -> pd.DataFrame:
    """Build merit order for one or more market areas."""
    use_adjusted = (
        result.merit_order.attrs.get("use_adjusted_capacity", True)
        if use_adjusted_capacity is None
        else use_adjusted_capacity
    )
    units = result.adjusted_units.copy() if use_adjusted else result.active_units.copy()
    if use_adjusted:
        units["capacity_mw"] = pd.to_numeric(units["Capacity_adj_GW"], errors="coerce") * 1000

    return build_merit_order(
        result.delivery_month,
        result.marginal_costs,
        units,
        areas=areas,
    )


def _select_merit_order_month(
    requested_month: pd.Timestamp,
    df_mc: pd.DataFrame,
    df_units: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return non-empty merit order for requested or nearest month."""
    merit_order = build_merit_order(requested_month, df_mc, df_units, region=None)
    if not merit_order.empty:
        return requested_month, merit_order

    valid_months = sorted(df_mc.loc[df_mc["mc_eur_mwh"].notna(), "delivery"].dropna().unique())
    if not valid_months:
        raise ValueError("No available month with valid marginal-cost data.")

    fallback = max([month for month in valid_months if month <= requested_month], default=valid_months[0])
    fallback = pd.Timestamp(fallback)
    return fallback, build_merit_order(fallback, df_mc, df_units, region=None)

def export_thermal_stack(merit_order: pd.DataFrame, dataset_key: str) -> Path:
    """Export thermal stack to MCP-catalogued Excel path."""
    output_path = data_loader.resolve_dataset_output_path(dataset_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        merit_order.to_excel(writer, index=True)
    return output_path

def build_thermal_stack(config: ThermalStackConfig | None = None) -> ThermalStackResult:
    """Build and optionally export Japan thermal stack."""
    config = config or ThermalStackConfig()
    unit_production = load_filtered_unit_production(config)
    thermal_registry = load_thermal_registry(config)
    nuclear_source = data_loader.load_nuclear_registry()
    nuclear_registry = prepare_nuclear_registry(nuclear_source, thermal_registry.columns, config)

    stack_units = thermal_registry[thermal_registry["fuel_class"] != "Nuclear"].copy()
    stack_units = pd.concat([stack_units, nuclear_registry], ignore_index=True)

    outages_thermal, outages_nuclear, outages_all = load_stack_outages(config)
    forecast_start = pd.Timestamp("today").normalize() if config.forecast_start is None else pd.Timestamp(config.forecast_start)
    active_units = filter_units_for_forecast(stack_units, forecast_start=forecast_start)
    active_units = active_units[active_units["fuel_class"].isin(config.stack_fuels)].reset_index(drop=True)

    fuel_cocktail = data_loader.load_japan_fuel_cocktail()
    marginal_costs = build_marginal_costs(active_units, fuel_cocktail)

    requested_month = _month_start(config.delivery_month)
    adjusted_units, capacity_summary = build_adjusted_capacity_month(stack_units, outages_all, requested_month)
    units_for_merit = adjusted_units.copy() if config.use_adjusted_capacity else active_units.copy()
    if config.use_adjusted_capacity:
        units_for_merit["capacity_mw"] = pd.to_numeric(units_for_merit["Capacity_adj_GW"], errors="coerce") * 1000

    delivery_month, merit_order = _select_merit_order_month(requested_month, marginal_costs, units_for_merit)
    merit_order.attrs["use_adjusted_capacity"] = config.use_adjusted_capacity

    export_path = export_thermal_stack(merit_order, config.export_dataset_key) if config.export_excel else None
    return ThermalStackResult(
        merit_order=merit_order,
        marginal_costs=marginal_costs,
        active_units=active_units,
        adjusted_units=adjusted_units,
        thermal_registry=thermal_registry,
        nuclear_registry=nuclear_registry,
        outages_thermal=outages_thermal,
        outages_nuclear=outages_nuclear,
        outages_all=outages_all,
        fuel_cocktail=fuel_cocktail,
        unit_production=unit_production,
        delivery_month=delivery_month,
        capacity_summary=capacity_summary,
        export_path=export_path,
    )

def plot_merit_order_plotly(
    merit_order: pd.DataFrame,
    region: str | list[str] | None = None,
    period: str | pd.Timestamp | None = None,
    title: str | None = None,
    height: int = 650,
    demand_gw: float | None = None,
    spot_price_eur_mwh: float | None = None,
    marginal_unit: pd.Series | None = None,
):
    """Create interactive merit order chart with optional demand and price overlays."""
    import plotly.graph_objects as go

    mo = merit_order.copy()
    _require_columns(mo, ["cum_GW", "capacity_mw", "mc_eur_mwh", "fuel_class"], "merit_order")

    if region is not None:
        _require_columns(mo, ["area_en"], "merit_order")
        regions = region if isinstance(region, list) else [region]
        mo = mo[mo["area_en"].isin(regions)].copy()
        mo = mo.sort_values("mc_eur_mwh").reset_index(drop=True)
        mo["cum_GW"] = mo["capacity_mw"].cumsum() / 1000

    if period is None:
        period = _month_start(None)
    else:
        period = pd.Timestamp(period)

    if title is None:
        region_label = " + ".join(region if isinstance(region, list) else [region]) if region else "All Japan"
        period_label = period.strftime("%B %Y")
        title = f"Merit Order ({region_label}) - {period_label}"

    fig = go.Figure()
    for fuel in ["Nuclear", "Coal", "LNG", "Oil"]:
        fuel_rows = mo[mo["fuel_class"] == fuel]
        for _, row in fuel_rows.iterrows():
            x0 = row["cum_GW"] - row["capacity_mw"] / 1000
            x1 = row["cum_GW"]
            y = row["mc_eur_mwh"]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[0, 0, y, y, 0],
                    fill="toself",
                    fillcolor=MERIT_ORDER_COLORS.get(fuel, "gray"),
                    opacity=0.7,
                    line=dict(color=MERIT_ORDER_COLORS.get(fuel, "gray"), width=0.5),
                    name=fuel,
                    legendgroup=fuel,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    for fuel in ["Nuclear", "Coal", "LNG", "Oil"]:
        if fuel in set(mo["fuel_class"]):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=MERIT_ORDER_COLORS.get(fuel, "gray"), symbol="square"),
                    name=fuel,
                    legendgroup=fuel,
                    showlegend=True,
                )
            )

    if marginal_unit is None and demand_gw is not None:
        marginal_unit = find_marginal_unit_by_demand(mo, demand_gw)
    if marginal_unit is None and spot_price_eur_mwh is not None:
        marginal_unit = find_marginal_unit_by_price(mo, spot_price_eur_mwh)

    x_max = mo["cum_GW"].max() * 1.02
    y_max = mo["mc_eur_mwh"].quantile(0.98) * 1.15

    if demand_gw is not None:
        fig.add_vline(
            x=demand_gw,
            line_dash="dash",
            line_color="#1f77b4",
            annotation_text=f"Demand {demand_gw:.1f} GW",
            annotation_position="top",
        )
        x_max = max(x_max, demand_gw * 1.05)

    if spot_price_eur_mwh is not None:
        fig.add_hline(
            y=spot_price_eur_mwh,
            line_dash="dot",
            line_color="#d62728",
            annotation_text=f"Spot {spot_price_eur_mwh:.0f} EUR/MWh",
            annotation_position="right",
        )
        y_max = max(y_max, spot_price_eur_mwh * 1.1)

    if marginal_unit is not None and not pd.isna(marginal_unit.get("cum_GW", np.nan)):
        marginal_x = marginal_unit["cum_GW"] - marginal_unit["capacity_mw"] / 2000
        marginal_y = marginal_unit["mc_eur_mwh"]
        fuel = marginal_unit.get("fuel_class", "unknown")
        fig.add_trace(
            go.Scatter(
                x=[marginal_x],
                y=[marginal_y],
                mode="markers+text",
                marker=dict(size=14, color="red", symbol="diamond", line=dict(width=2, color="white")),
                text=[f"Marginal: {fuel}"],
                textposition="top center",
                name="Marginal unit",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Cumulative GW",
        yaxis_title="Marginal Cost (EUR/MWh)",
        height=height,
        hovermode="closest",
        template="plotly_white",
        yaxis=dict(range=[0, y_max]),
        xaxis=dict(range=[0, x_max]),
    )
    return fig


def plot_regional_merit_order_plotly(result: ThermalStackResult, height: int = 600):
    """Create East-vs-West merit order charts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    west = merit_order_for_areas(result, areas=list(WEST_JAPAN_AREAS))
    east = merit_order_for_areas(result, areas=list(EAST_JAPAN_AREAS))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["West Japan", "East Japan"],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    for column_index, merit_order in enumerate([west, east], start=1):
        shown_legends: set[str] = set()
        for _, row in merit_order.iterrows():
            fuel = row["fuel_class"]
            x0 = row["cum_GW"] - row["capacity_mw"] / 1000
            x1 = row["cum_GW"]
            show_legend = fuel not in shown_legends and column_index == 1
            shown_legends.add(fuel)
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[0, 0, row["mc_eur_mwh"], row["mc_eur_mwh"], 0],
                    fill="toself",
                    fillcolor=MERIT_ORDER_COLORS.get(fuel, "gray"),
                    opacity=0.7,
                    line=dict(color=MERIT_ORDER_COLORS.get(fuel, "gray"), width=0.5),
                    name=fuel,
                    legendgroup=fuel,
                    showlegend=show_legend,
                    hoverinfo="skip",
                ),
                row=1,
                col=column_index,
            )

    fig.update_xaxes(title_text="Cumulative GW", row=1, col=1)
    fig.update_xaxes(title_text="Cumulative GW", row=1, col=2)
    fig.update_yaxes(title_text="Marginal Cost (EUR/MWh)", row=1, col=1)
    fig.update_layout(
        title=f"Merit Order - West vs East Japan - {result.delivery_month.strftime('%B %Y')}",
        height=height,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def jpy_kwh_to_eur_mwh(price_jpy_kwh: float, eurjpy: float) -> float:
    """Convert JEPX spot price from JPY/kWh to EUR/MWh."""
    if pd.isna(price_jpy_kwh) or pd.isna(eurjpy) or eurjpy <= 0:
        return np.nan
    return float(price_jpy_kwh) * 1000.0 / float(eurjpy)


def _daily_eurjpy_series(fuel_cocktail: pd.DataFrame) -> pd.Series:
    """Return daily EURJPY series indexed by calendar date."""
    if "EURJPY" in fuel_cocktail.columns:
        return fuel_cocktail["EURJPY"]
    if "EURUSD" in fuel_cocktail.columns:
        return fuel_cocktail["EURUSD"]
    raise ValueError("fuel_cocktail must contain EURJPY or EURUSD for spot conversion.")


def convert_spot_to_eur_mwh(spot: pd.DataFrame, fuel_cocktail: pd.DataFrame, region: str) -> pd.Series:
    """Convert one regional JEPX spot column to EUR/MWh."""
    spot_column = REGION_SPOT_COLUMNS.get(region)
    if spot_column is None:
        raise ValueError(f"Unknown region {region!r}. Expected one of {sorted(REGION_SPOT_COLUMNS)}.")
    if spot_column not in spot.columns:
        raise ValueError(f"Spot data is missing column {spot_column!r}.")

    eurjpy = _daily_eurjpy_series(fuel_cocktail)
    daily_eurjpy = pd.to_numeric(eurjpy, errors="coerce")
    daily_eurjpy.index = pd.to_datetime(daily_eurjpy.index).normalize()

    spot_prices = spot[spot_column].astype(float)
    spot_dates = pd.to_datetime(spot_prices.index)
    if spot_dates.tz is not None:
        spot_dates = spot_dates.tz_convert(None)
    spot_dates = spot_dates.normalize()
    mapped_eurjpy = spot_dates.map(daily_eurjpy)
    converted = spot_prices.mul(1000.0).div(mapped_eurjpy)
    converted.index = spot_prices.index
    return converted


def _align_series_index(series: pd.Series, spot_index: pd.DatetimeIndex) -> pd.Series:
    """Align one series index to the spot index timezone."""
    aligned = series.copy()
    aligned.index = pd.to_datetime(aligned.index)
    if aligned.index.tz is None and spot_index.tz is not None:
        aligned.index = aligned.index.tz_localize(spot_index.tz)
    return aligned


def _aggregate_regional_series_mw(
    df_30min: pd.DataFrame,
    areas: Iterable[str],
    column: str,
) -> pd.Series:
    """Sum one 30-minute column across multiple market areas."""
    total: pd.Series | None = None
    for area in areas:
        key = (area, column)
        if key not in df_30min.columns:
            raise ValueError(f"30-minute data has no {column!r} column for area {area!r}.")
        series = pd.to_numeric(df_30min[key], errors="coerce")
        total = series if total is None else total.add(series, fill_value=0)
    if total is None:
        raise ValueError("At least one market area is required.")
    return total


def compute_demand_mw(
    df_30min: pd.DataFrame,
    areas: str | Iterable[str],
    demand_basis: str = DEMAND_BASIS_CONSUMPTION,
) -> pd.Series:
    """Compute demand MW for one area or a combined regional grouping."""
    area_list = [areas] if isinstance(areas, str) else list(areas)
    consumption = _aggregate_regional_series_mw(df_30min, area_list, "cons_act")

    if demand_basis == DEMAND_BASIS_CONSUMPTION:
        return consumption

    solar = _aggregate_regional_series_mw(df_30min, area_list, "gen_sol_act")
    wind = _aggregate_regional_series_mw(df_30min, area_list, "gen_win_act")

    if demand_basis == DEMAND_BASIS_RESIDUAL_LOAD:
        return consumption - solar - wind

    if demand_basis == DEMAND_BASIS_THERMAL_RESIDUAL:
        flows = _aggregate_regional_series_mw(df_30min, area_list, "inter_flows")
        hydro = _aggregate_regional_series_mw(df_30min, area_list, "gen_hyd_tot_act")
        nuclear = _aggregate_regional_series_mw(df_30min, area_list, "gen_nuc_act")
        return consumption - solar - wind - flows - hydro - nuclear

    raise ValueError(
        f"Unsupported demand_basis {demand_basis!r}. Expected one of {DEMAND_BASIS_CHOICES}."
    )


def load_regional_demand_mw(df_30min: pd.DataFrame, region: str) -> pd.Series:
    """Return regional consumption in MW from the 30-minute wide table."""
    return compute_demand_mw(df_30min, region, demand_basis=DEMAND_BASIS_CONSUMPTION)


def find_marginal_unit_by_demand(merit_order: pd.DataFrame, demand_gw: float) -> pd.Series | None:
    """Return the marginal stack unit for a given demand level."""
    if pd.isna(demand_gw):
        return None

    mo = merit_order.sort_values("mc_eur_mwh").reset_index(drop=True)
    eligible = mo.index[mo["cum_GW"] >= demand_gw]
    if len(eligible) == 0:
        return mo.iloc[-1]
    return mo.loc[eligible[0]]


def find_marginal_unit_by_price(merit_order: pd.DataFrame, price_eur_mwh: float) -> pd.Series | None:
    """Return the marginal stack unit closest to the observed clearing price."""
    if pd.isna(price_eur_mwh):
        return None

    mo = merit_order.sort_values("mc_eur_mwh").reset_index(drop=True)
    eligible = mo[mo["mc_eur_mwh"] <= price_eur_mwh]
    if eligible.empty:
        return mo.iloc[0]
    return eligible.iloc[-1]


def identify_price_setters(
    merit_order: pd.DataFrame,
    region: str | None = None,
    spot: pd.DataFrame | None = None,
    df_30min: pd.DataFrame | None = None,
    fuel_cocktail: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    config: PriceSetterConfig | None = None,
    areas: str | list[str] | None = None,
    spot_area: str | None = None,
    demand_basis: str = DEMAND_BASIS_CONSUMPTION,
) -> pd.DataFrame:
    """Identify marginal fuel for each 30-minute interval."""
    if config is not None:
        area_list = list(config.areas)
        spot_reference = config.spot_area
        demand_mode = config.demand_basis
        config_name = config.name
    else:
        if region is None and areas is None:
            raise ValueError("Provide either config, region, or areas.")
        area_list = [region] if areas is None else ([areas] if isinstance(areas, str) else list(areas))
        spot_reference = spot_area or area_list[0]
        demand_mode = demand_basis
        config_name = "+".join(area_list) + f" | {demand_mode}"

    if spot is None:
        spot = data_loader.load_jepx_spot()
    if df_30min is None:
        df_30min = data_loader.load_df30min()
    if fuel_cocktail is None:
        fuel_cocktail = data_loader.load_japan_fuel_cocktail()

    demand_mw = compute_demand_mw(df_30min, area_list, demand_basis=demand_mode)
    spot_eur_mwh = convert_spot_to_eur_mwh(spot, fuel_cocktail, spot_reference)
    spot_index = pd.to_datetime(spot.index)

    demand_mw = _align_series_index(demand_mw, spot_index)
    spot_eur_mwh = _align_series_index(spot_eur_mwh, spot_index)

    common_index = demand_mw.index.intersection(spot_eur_mwh.index).intersection(spot_index)
    frame = pd.DataFrame(
        {
            "config": config_name,
            "areas": ", ".join(area_list),
            "spot_area": spot_reference,
            "demand_basis": demand_mode,
            "demand_gw": demand_mw.reindex(common_index) / 1000.0,
            "spot_jpy_kwh": spot[REGION_SPOT_COLUMNS[spot_reference]].astype(float).reindex(common_index),
            "spot_eur_mwh": spot_eur_mwh.reindex(common_index),
        },
        index=common_index,
    )
    for generation_column, fuel in GENERATION_COLUMNS_BY_FUEL.items():
        try:
            generation = _aggregate_regional_series_mw(df_30min, area_list, generation_column)
        except ValueError:
            continue
        generation = _align_series_index(generation, spot_index)
        frame[f"gen_{fuel.lower()}_mw"] = generation.reindex(common_index)

    if start is not None:
        start_ts = pd.Timestamp(start)
        if frame.index.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(frame.index.tz)
        frame = frame.loc[start_ts:]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if frame.index.tz is not None and end_ts.tz is None:
            end_ts = end_ts.tz_localize(frame.index.tz)
        frame = frame.loc[:end_ts]

    demand_setters = []
    price_setters = []
    demand_units = []
    price_units = []
    for _, row in frame.iterrows():
        demand_gw = row["demand_gw"]
        if pd.notna(demand_gw) and demand_gw < 0:
            demand_gw = 0.0
        demand_unit = find_marginal_unit_by_demand(merit_order, demand_gw)
        price_unit = find_marginal_unit_by_price(merit_order, row["spot_eur_mwh"])
        demand_setters.append(None if demand_unit is None else demand_unit.get("fuel_class"))
        price_setters.append(None if price_unit is None else price_unit.get("fuel_class"))
        demand_units.append(None if demand_unit is None else demand_unit.get("plant_name_jp"))
        price_units.append(None if price_unit is None else price_unit.get("plant_name_jp"))

    frame["marginal_fuel_by_demand"] = demand_setters
    frame["marginal_fuel_by_price"] = price_setters
    frame["marginal_plant_by_demand"] = demand_units
    frame["marginal_plant_by_price"] = price_units
    return frame


def run_price_setter_configs(
    result: ThermalStackResult,
    configs: list[PriceSetterConfig] | None = None,
    spot: pd.DataFrame | None = None,
    df_30min: pd.DataFrame | None = None,
    fuel_cocktail: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Run multiple price-setter configurations in parallel for comparison."""
    configs = configs or default_price_setter_configs()
    spot = spot or data_loader.load_jepx_spot()
    df_30min = df_30min or data_loader.load_df30min()
    fuel_cocktail = fuel_cocktail or data_loader.load_japan_fuel_cocktail()

    outputs: dict[str, pd.DataFrame] = {}
    for config in configs:
        merit_order = merit_order_for_areas(result, areas=list(config.areas))
        outputs[config.name] = identify_price_setters(
            merit_order,
            config=config,
            spot=spot,
            df_30min=df_30min,
            fuel_cocktail=fuel_cocktail,
            start=start,
            end=end,
        )
    return outputs


def summarize_price_setter_configs(config_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize marginal-fuel shares across multiple configurations."""
    summaries = []
    for config_name, setters in config_results.items():
        summary = summarize_price_setters(setters)
        if summary.empty:
            continue
        summary["config"] = config_name
        if "demand_basis" in setters.columns:
            summary["demand_basis"] = setters["demand_basis"].iloc[0]
        if "areas" in setters.columns:
            summary["areas"] = setters["areas"].iloc[0]
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def compare_price_setter_demand_levels(config_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare demand levels used by each configuration."""
    rows = []
    for config_name, setters in config_results.items():
        rows.append(
            {
                "config": config_name,
                "areas": setters["areas"].iloc[0] if "areas" in setters.columns else None,
                "demand_basis": setters["demand_basis"].iloc[0] if "demand_basis" in setters.columns else None,
                "demand_gw_min": setters["demand_gw"].min(),
                "demand_gw_median": setters["demand_gw"].median(),
                "demand_gw_max": setters["demand_gw"].max(),
                "spot_eur_mwh_median": setters["spot_eur_mwh"].median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_price_setters(setters: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each fuel is marginal."""
    summaries = []
    for column in ("marginal_fuel_by_demand", "marginal_fuel_by_price"):
        if column not in setters.columns:
            continue
        counts = setters[column].value_counts(dropna=False).rename("intervals")
        summary = (counts / counts.sum()).rename("share").to_frame()
        summary["method"] = column.replace("marginal_fuel_by_", "")
        summary["fuel_class"] = summary.index
        summaries.append(summary.reset_index(drop=True))
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def plot_price_setter_share_plotly(summary: pd.DataFrame, region: str, title: str | None = None):
    """Plot marginal-fuel shares from summarize_price_setters()."""
    import plotly.express as px

    label = region
    if "config" in summary.columns and summary["config"].nunique() == 1:
        label = str(summary["config"].iloc[0])

    if title is None:
        title = f"Marginal fuel share - {label}"
    chart = px.bar(
        summary,
        x="method",
        y="share",
        color="fuel_class",
        color_discrete_map=MERIT_ORDER_COLORS,
        title=title,
        labels={"method": "Identification method", "share": "Share of intervals", "fuel_class": "Fuel"},
    )
    chart.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    return chart


def plot_price_setter_config_comparison(
    summary: pd.DataFrame,
    title: str = "Marginal fuel share by configuration",
):
    """Plot side-by-side comparison across price-setter configurations."""
    import plotly.express as px

    chart = px.bar(
        summary,
        x="config",
        y="share",
        color="fuel_class",
        facet_col="method",
        color_discrete_map=MERIT_ORDER_COLORS,
        title=title,
        labels={"config": "Configuration", "share": "Share of intervals", "fuel_class": "Fuel"},
        category_orders={"config": list(summary["config"].drop_duplicates())},
    )
    chart.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    chart.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return chart


def main() -> None:
    """Run thermal stack pipeline from command line."""
    result = build_thermal_stack()
    print(f"Thermal stack rows: {len(result.merit_order):,}")
    print(f"Delivery month: {result.delivery_month:%Y-%m}")
    if result.export_path is not None:
        print(f"Exported to: {result.export_path}")


if __name__ == "__main__":
    main()