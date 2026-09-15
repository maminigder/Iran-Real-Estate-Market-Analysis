"""Generate the first validated nationwide market-analysis outputs.

This script intentionally uses medians and minimum sample thresholds because
real-estate asking prices are heavily skewed and city sample sizes differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "residential_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MIN_CITY_SAMPLE = 100


def to_json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_core_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    core = df.loc[df["analysis_ready"].fillna(False)].copy()

    # Winsor-style exclusion for the primary descriptive sample. The raw
    # observations stay in the processed file; only the extreme tails of
    # price-per-m² are excluded from national/city comparisons.
    lower = float(core["asking_price_per_sqm"].quantile(0.005))
    upper = float(core["asking_price_per_sqm"].quantile(0.995))
    core = core.loc[core["asking_price_per_sqm"].between(lower, upper)].copy()

    return core, {"lower_ppsqm_cutoff": lower, "upper_ppsqm_cutoff": upper}


def city_summary(core: pd.DataFrame) -> pd.DataFrame:
    summary = (
        core.groupby("city_slug", dropna=False)
        .agg(
            listings=("asking_price", "size"),
            median_asking_price=("asking_price", "median"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            mean_price_per_sqm=("asking_price_per_sqm", "mean"),
            median_building_size_sqm=("building_size_sqm", "median"),
            median_rooms=("rooms_numeric", "median"),
        )
        .reset_index()
    )
    return summary.sort_values("listings", ascending=False)


def amenity_summary(core: pd.DataFrame) -> pd.DataFrame:
    amenities = ["has_elevator", "has_parking", "has_warehouse", "is_rebuilt"]
    records: list[dict] = []

    for amenity in amenities:
        subset = core.loc[core[amenity].notna()].copy()
        for value, group in subset.groupby(amenity):
            records.append(
                {
                    "amenity": amenity,
                    "value": bool(value),
                    "listings": len(group),
                    "median_asking_price": group["asking_price"].median(),
                    "median_price_per_sqm": group["asking_price_per_sqm"].median(),
                    "median_building_size_sqm": group["building_size_sqm"].median(),
                }
            )

    return pd.DataFrame(records)


def construction_year_summary(core: pd.DataFrame) -> pd.DataFrame:
    valid = core.loc[core["construction_year_jalali"].between(1300, 1450)].copy()
    bins = [1299, 1379, 1389, 1399, 1450]
    labels = ["<=1379", "1380-1389", "1390-1399", "1400+"]
    valid["construction_period_jalali"] = pd.cut(
        valid["construction_year_jalali"],
        bins=bins,
        labels=labels,
    )

    result = (
        valid.groupby("construction_period_jalali", observed=True)
        .agg(
            listings=("asking_price", "size"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            median_asking_price=("asking_price", "median"),
            median_building_size_sqm=("building_size_sqm", "median"),
        )
        .reset_index()
    )
    return result


def save_charts(core: pd.DataFrame, cities: pd.DataFrame, years: pd.DataFrame) -> None:
    top_counts = cities.head(15).sort_values("listings")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_counts["city_slug"], top_counts["listings"])
    ax.set_title("Residential Sale Listings: Top 15 Cities in the Analytical Sample")
    ax.set_xlabel("Number of listings")
    ax.set_ylabel("City")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_cities_by_listing_count.png", dpi=180)
    plt.close(fig)

    eligible = cities.loc[cities["listings"] >= MIN_CITY_SAMPLE].copy()
    top_price = eligible.nlargest(20, "median_price_per_sqm").sort_values(
        "median_price_per_sqm"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_price["city_slug"], top_price["median_price_per_sqm"] / 1_000_000)
    ax.set_title(
        f"Highest Median Asking Price per m² by City (minimum {MIN_CITY_SAMPLE} listings)"
    )
    ax.set_xlabel("Median asking price per m² (millions of source units)")
    ax.set_ylabel("City")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "median_price_per_sqm_by_city.png", dpi=180)
    plt.close(fig)

    if not years.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            years["construction_period_jalali"].astype(str),
            years["median_price_per_sqm"] / 1_000_000,
        )
        ax.set_title("Median Asking Price per m² by Construction Period")
        ax.set_xlabel("Construction year (Jalali)")
        ax.set_ylabel("Median asking price per m² (millions of source units)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "price_per_sqm_by_construction_period.png", dpi=180)
        plt.close(fig)

    # Distribution chart limited to the central 99% for readability.
    upper_size = core["building_size_sqm"].quantile(0.99)
    sizes = core.loc[core["building_size_sqm"] <= upper_size, "building_size_sqm"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(sizes, bins=50)
    ax.set_title("Distribution of Residential Building Size")
    ax.set_xlabel("Building size (m²)")
    ax.set_ylabel("Listings")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "building_size_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared file not found: {DATA_FILE}\n"
            "Run `python src/prepare_sales_data.py` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_FILE)
    core, cutoffs = build_core_sample(df)

    cities = city_summary(core)
    amenities = amenity_summary(core)
    years = construction_year_summary(core)

    cities.to_csv(OUTPUT_DIR / "city_summary.csv", index=False)
    amenities.to_csv(OUTPUT_DIR / "amenity_summary.csv", index=False)
    years.to_csv(OUTPUT_DIR / "construction_year_summary.csv", index=False)

    date_min = core["listing_month"].min()
    date_max = core["listing_month"].max()

    summary = {
        "processed_residential_sale_rows": len(df),
        "analysis_ready_before_ppsqm_tail_filter": int(df["analysis_ready"].sum()),
        "core_analysis_rows": len(core),
        "cities_in_core_sample": int(core["city_slug"].nunique()),
        "observed_listing_month_min": to_json_safe(date_min) if pd.notna(date_min) else None,
        "observed_listing_month_max": to_json_safe(date_max) if pd.notna(date_max) else None,
        "median_building_size_sqm": to_json_safe(core["building_size_sqm"].median()),
        "median_asking_price_source_units": to_json_safe(core["asking_price"].median()),
        "median_asking_price_per_sqm_source_units": to_json_safe(
            core["asking_price_per_sqm"].median()
        ),
        "minimum_city_sample_for_price_ranking": MIN_CITY_SAMPLE,
        **{key: to_json_safe(value) for key, value in cutoffs.items()},
        "interpretation_note": (
            "Prices are listing/asking values. Monetary denomination is preserved as "
            "source units because the official dataset documentation does not clearly "
            "specify the unit."
        ),
    }

    with (OUTPUT_DIR / "market_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_charts(core, cities, years)

    print("Analysis complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
