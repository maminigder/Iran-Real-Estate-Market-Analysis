"""Build a Tehran / major-cities deep dive with neighborhood and time analysis.

The outputs are designed for a portfolio: robust neighborhood rankings,
approximate geographic visualization, monthly market trends, and a
composition-adjusted price index that controls for observable listing mix.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "residential_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

TARGET_CITY = "tehran"
N_MAJOR_CITIES = 5
MIN_NEIGHBORHOOD_SAMPLE = 100
MIN_RANKING_SAMPLE = 250
MIN_MONTHLY_SAMPLE = 150
MAX_ADJUSTMENT_TRAIN_ROWS_PER_CITY = 80_000
RANDOM_STATE = 42
SHRINKAGE_STRENGTH = 200

PERSIAN_ARABIC_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def build_core_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Match the nationwide analysis-ready sample and tail treatment."""
    core = df.loc[df["analysis_ready"].fillna(False)].copy()
    lower = float(core["asking_price_per_sqm"].quantile(0.005))
    upper = float(core["asking_price_per_sqm"].quantile(0.995))
    core = core.loc[core["asking_price_per_sqm"].between(lower, upper)].copy()
    return core, {"lower_ppsqm_cutoff": lower, "upper_ppsqm_cutoff": upper}


def extract_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.translate(PERSIAN_ARABIC_DIGITS)
    extracted = text.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def share_true(series: pd.Series) -> float:
    """Return a robust true-share for bool-like fields, preserving missingness."""
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )
    mapped = mapped.dropna()
    if mapped.empty:
        return float("nan")
    return float(mapped.mean())


def major_cities(core: pd.DataFrame) -> list[str]:
    counts = core["city_slug"].value_counts(dropna=True)
    return counts.head(N_MAJOR_CITIES).index.astype(str).tolist()


def neighborhood_summary(core: pd.DataFrame, city: str) -> pd.DataFrame:
    city_data = core.loc[core["city_slug"].eq(city)].copy()
    city_data = city_data.loc[city_data["neighborhood_slug"].notna()].copy()
    city_data["neighborhood_slug"] = city_data["neighborhood_slug"].astype(str).str.strip()
    city_data = city_data.loc[city_data["neighborhood_slug"].ne("")].copy()

    if city_data.empty:
        return pd.DataFrame()

    summary = (
        city_data.groupby("neighborhood_slug", dropna=False)
        .agg(
            listings=("asking_price", "size"),
            median_asking_price=("asking_price", "median"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            p25_price_per_sqm=("asking_price_per_sqm", lambda s: s.quantile(0.25)),
            p75_price_per_sqm=("asking_price_per_sqm", lambda s: s.quantile(0.75)),
            median_building_size_sqm=("building_size_sqm", "median"),
            median_rooms=("rooms_numeric", "median"),
            median_construction_year_jalali=("construction_year_jalali", "median"),
            median_latitude=("location_latitude", "median"),
            median_longitude=("location_longitude", "median"),
            median_location_radius=("location_radius", "median"),
            elevator_share=("has_elevator", share_true),
            parking_share=("has_parking", share_true),
            warehouse_share=("has_warehouse", share_true),
            rebuilt_share=("is_rebuilt", share_true),
            balcony_share=("has_balcony", share_true),
        )
        .reset_index()
    )

    city_median = float(city_data["asking_price_per_sqm"].median())
    summary["price_vs_city_median_pct"] = (
        summary["median_price_per_sqm"] / city_median - 1.0
    ) * 100.0

    # A transparent sample-size stabilization: small neighborhood medians are
    # pulled toward the city median instead of being allowed to dominate a rank.
    credibility = summary["listings"] / (summary["listings"] + SHRINKAGE_STRENGTH)
    summary["stabilized_price_per_sqm"] = np.exp(
        credibility * np.log(summary["median_price_per_sqm"])
        + (1.0 - credibility) * np.log(city_median)
    )
    summary["stabilization_weight"] = credibility
    summary["iqr_price_per_sqm"] = summary["p75_price_per_sqm"] - summary["p25_price_per_sqm"]

    return summary.sort_values("listings", ascending=False)


def monthly_raw_summary(core: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    data = core.loc[
        core["city_slug"].isin(cities) & core["listing_month"].notna()
    ].copy()
    data["listing_month"] = pd.to_datetime(data["listing_month"]).dt.to_period("M").dt.to_timestamp()

    monthly = (
        data.groupby(["city_slug", "listing_month"])
        .agg(
            listings=("asking_price", "size"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            median_asking_price=("asking_price", "median"),
            median_building_size_sqm=("building_size_sqm", "median"),
            median_construction_year_jalali=("construction_year_jalali", "median"),
        )
        .reset_index()
    )
    monthly["eligible_month"] = monthly["listings"] >= MIN_MONTHLY_SAMPLE
    return monthly


def prepare_adjustment_features(city_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = city_data.copy()
    work["log_building_size"] = np.log(work["building_size_sqm"].clip(lower=1))
    work["floor_numeric"] = extract_number(work["floor"])
    work["total_floors_numeric"] = extract_number(work["total_floors_count"])
    work["units_per_floor_numeric"] = extract_number(work["unit_per_floor"])

    numeric_features = [
        "log_building_size",
        "rooms_numeric",
        "construction_year_jalali",
        "floor_numeric",
        "total_floors_numeric",
        "units_per_floor_numeric",
    ]
    categorical_features = [
        "neighborhood_slug",
        "cat3_slug",
        "property_type",
        "user_type",
        "has_elevator",
        "has_parking",
        "has_warehouse",
        "is_rebuilt",
        "has_balcony",
    ]

    for column in categorical_features:
        work[column] = work[column].astype("string").fillna("unknown")

    return work, numeric_features, categorical_features


def composition_adjusted_city_trend(
    core: pd.DataFrame,
    city: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    city_data = core.loc[
        core["city_slug"].eq(city) & core["listing_month"].notna()
    ].copy()
    if city_data.empty:
        return pd.DataFrame(), {"city_slug": city, "status": "no_data"}

    city_data["listing_month"] = (
        pd.to_datetime(city_data["listing_month"]).dt.to_period("M").dt.to_timestamp()
    )
    city_data, numeric_features, categorical_features = prepare_adjustment_features(city_data)
    city_data["log_price_per_sqm"] = np.log(city_data["asking_price_per_sqm"])

    if len(city_data) > MAX_ADJUSTMENT_TRAIN_ROWS_PER_CITY:
        train = city_data.sample(
            n=MAX_ADJUSTMENT_TRAIN_ROWS_PER_CITY,
            random_state=RANDOM_STATE,
        ).copy()
    else:
        train = city_data.copy()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=10.0, solver="lsqr")),
        ]
    )

    features = numeric_features + categorical_features
    model.fit(train[features], train["log_price_per_sqm"])
    fitted_train = model.predict(train[features])
    in_sample_r2 = float(r2_score(train["log_price_per_sqm"], fitted_train))

    expected = model.predict(city_data[features])
    city_data["mix_adjustment_residual"] = city_data["log_price_per_sqm"] - expected

    adjusted = (
        city_data.groupby("listing_month")
        .agg(
            listings=("asking_price", "size"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            median_mix_residual=("mix_adjustment_residual", "median"),
        )
        .reset_index()
    )
    adjusted = adjusted.loc[adjusted["listings"] >= MIN_MONTHLY_SAMPLE].copy()
    adjusted["city_slug"] = city

    if adjusted.empty:
        return adjusted, {
            "city_slug": city,
            "status": "insufficient_monthly_sample",
            "training_rows": len(train),
            "in_sample_r2": in_sample_r2,
        }

    adjusted = adjusted.sort_values("listing_month")
    baseline_raw = float(adjusted.iloc[0]["median_price_per_sqm"])
    baseline_residual = float(adjusted.iloc[0]["median_mix_residual"])
    adjusted["raw_price_index_first_eligible_100"] = (
        adjusted["median_price_per_sqm"] / baseline_raw * 100.0
    )
    adjusted["composition_adjusted_index_first_eligible_100"] = np.exp(
        adjusted["median_mix_residual"] - baseline_residual
    ) * 100.0

    diagnostics: dict[str, float | int | str] = {
        "city_slug": city,
        "status": "ok",
        "rows_available": len(city_data),
        "training_rows": len(train),
        "eligible_months": len(adjusted),
        "first_eligible_month": adjusted["listing_month"].min().date().isoformat(),
        "last_eligible_month": adjusted["listing_month"].max().date().isoformat(),
        "in_sample_r2_mix_model": in_sample_r2,
    }
    return adjusted, diagnostics


def common_base_reindex(trends: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Rebase every major city to the first month available for all cities."""
    if trends.empty:
        return trends, None

    month_sets = [
        set(group["listing_month"].tolist())
        for _, group in trends.groupby("city_slug")
        if not group.empty
    ]
    if not month_sets:
        return trends, None
    common = sorted(set.intersection(*month_sets))
    if not common:
        trends["raw_price_index_common_100"] = trends["raw_price_index_first_eligible_100"]
        trends["composition_adjusted_index_common_100"] = trends[
            "composition_adjusted_index_first_eligible_100"
        ]
        return trends, None

    base_month = common[0]
    pieces: list[pd.DataFrame] = []
    for city, group in trends.groupby("city_slug"):
        group = group.sort_values("listing_month").copy()
        base_row = group.loc[group["listing_month"].eq(base_month)].iloc[0]
        raw_base = float(base_row["median_price_per_sqm"])
        residual_base = float(base_row["median_mix_residual"])
        group["raw_price_index_common_100"] = group["median_price_per_sqm"] / raw_base * 100.0
        group["composition_adjusted_index_common_100"] = np.exp(
            group["median_mix_residual"] - residual_base
        ) * 100.0
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True), base_month.date().isoformat()


def major_city_summary(trends: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for city, group in trends.groupby("city_slug"):
        group = group.sort_values("listing_month")
        first = group.iloc[0]
        last = group.iloc[-1]
        records.append(
            {
                "city_slug": city,
                "eligible_months": len(group),
                "first_eligible_month": first["listing_month"],
                "last_eligible_month": last["listing_month"],
                "latest_month_listings": int(last["listings"]),
                "latest_median_price_per_sqm": float(last["median_price_per_sqm"]),
                "raw_index_change_pct": float(
                    last["raw_price_index_first_eligible_100"] - 100.0
                ),
                "composition_adjusted_index_change_pct": float(
                    last["composition_adjusted_index_first_eligible_100"] - 100.0
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        "latest_median_price_per_sqm", ascending=False
    )


def tehran_map(neighborhoods: pd.DataFrame) -> None:
    eligible = neighborhoods.loc[
        (neighborhoods["listings"] >= MIN_NEIGHBORHOOD_SAMPLE)
        & neighborhoods["median_latitude"].notna()
        & neighborhoods["median_longitude"].notna()
    ].copy()
    if eligible.empty:
        return

    eligible = eligible.loc[
        eligible["median_latitude"].between(20, 45)
        & eligible["median_longitude"].between(40, 70)
    ].copy()
    if len(eligible) >= 20:
        lat_lo, lat_hi = eligible["median_latitude"].quantile([0.01, 0.99])
        lon_lo, lon_hi = eligible["median_longitude"].quantile([0.01, 0.99])
        eligible = eligible.loc[
            eligible["median_latitude"].between(lat_lo, lat_hi)
            & eligible["median_longitude"].between(lon_lo, lon_hi)
        ].copy()
    if eligible.empty:
        return

    marker_sizes = 25 + 160 * np.sqrt(
        eligible["listings"] / eligible["listings"].max()
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        eligible["median_longitude"],
        eligible["median_latitude"],
        s=marker_sizes,
        c=eligible["stabilized_price_per_sqm"] / 1_000_000,
        alpha=0.72,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Stabilized median asking price / m² (millions of source units)")

    for _, row in eligible.nlargest(12, "listings").iterrows():
        ax.annotate(
            str(row["neighborhood_slug"]),
            (row["median_longitude"], row["median_latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Tehran Neighborhood Asking-Price Map (Approximate Listing Coordinates)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tehran_neighborhood_price_map.png", dpi=190)
    plt.close(fig)


def tehran_ranking_chart(neighborhoods: pd.DataFrame) -> None:
    ranked = neighborhoods.loc[neighborhoods["listings"] >= MIN_RANKING_SAMPLE].copy()
    if ranked.empty:
        return
    ranked = ranked.nlargest(20, "stabilized_price_per_sqm").sort_values(
        "stabilized_price_per_sqm"
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(
        ranked["neighborhood_slug"],
        ranked["stabilized_price_per_sqm"] / 1_000_000,
    )
    ax.set_title(
        f"Tehran: Highest Stabilized Neighborhood Asking Price / m² "
        f"(minimum {MIN_RANKING_SAMPLE} listings)"
    )
    ax.set_xlabel("Stabilized median asking price / m² (millions of source units)")
    ax.set_ylabel("Neighborhood")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tehran_neighborhood_price_ranking.png", dpi=190)
    plt.close(fig)


def tehran_amenity_chart(neighborhoods: pd.DataFrame) -> None:
    ranked = neighborhoods.loc[neighborhoods["listings"] >= MIN_RANKING_SAMPLE].copy()
    ranked = ranked.nlargest(15, "listings")
    if ranked.empty:
        return

    features = ["elevator_share", "parking_share", "warehouse_share", "rebuilt_share"]
    matrix = ranked[features].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ranked)), labels=ranked["neighborhood_slug"])
    ax.set_xticks(
        np.arange(len(features)),
        labels=["Elevator", "Parking", "Storage", "Rebuilt"],
    )
    ax.set_title("Tehran: Amenity Share in the 15 Largest Neighborhood Samples")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Share of known listings")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tehran_neighborhood_amenity_profile.png", dpi=190)
    plt.close(fig)


def price_trend_chart(trends: pd.DataFrame, common_base: str | None) -> None:
    if trends.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for city, group in trends.groupby("city_slug"):
        group = group.sort_values("listing_month")
        ax.plot(
            group["listing_month"],
            group["composition_adjusted_index_common_100"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=city,
        )
    base_text = common_base or "city-specific first eligible month"
    ax.set_title("Major Cities: Composition-Adjusted Asking-Price Index")
    ax.set_xlabel("Listing month")
    ax.set_ylabel(f"Index ({base_text} = 100)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "major_cities_composition_adjusted_price_index.png", dpi=190)
    plt.close(fig)


def listing_volume_chart(raw_monthly: pd.DataFrame, cities: list[str]) -> None:
    data = raw_monthly.loc[raw_monthly["city_slug"].isin(cities)].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for city, group in data.groupby("city_slug"):
        group = group.sort_values("listing_month")
        ax.plot(group["listing_month"], group["listings"], linewidth=1.7, label=city)
    ax.set_title("Major Cities: Monthly Residential-Sale Listing Volume")
    ax.set_xlabel("Listing month")
    ax.set_ylabel("Listings")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "major_cities_monthly_listing_volume.png", dpi=190)
    plt.close(fig)


def city_positioning_chart(core: pd.DataFrame, cities: list[str]) -> None:
    data = core.loc[core["city_slug"].isin(cities)].copy()
    summary = (
        data.groupby("city_slug")
        .agg(
            listings=("asking_price", "size"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            median_building_size_sqm=("building_size_sqm", "median"),
        )
        .reset_index()
    )
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = 70 + 280 * np.sqrt(summary["listings"] / summary["listings"].max())
    ax.scatter(
        summary["median_building_size_sqm"],
        summary["median_price_per_sqm"] / 1_000_000,
        s=sizes,
        alpha=0.75,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["city_slug"],
            (row["median_building_size_sqm"], row["median_price_per_sqm"] / 1_000_000),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_title("Major-City Market Positioning")
    ax.set_xlabel("Median building size (m²)")
    ax.set_ylabel("Median asking price / m² (millions of source units)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "major_cities_market_positioning.png", dpi=190)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    if df.empty:
        return "_No eligible observations._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("—")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(
    core: pd.DataFrame,
    cities: list[str],
    neighborhoods: pd.DataFrame,
    city_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    common_base: str | None,
) -> None:
    tehran_rows = int(core["city_slug"].eq(TARGET_CITY).sum())
    eligible_neighborhoods = neighborhoods.loc[
        neighborhoods["listings"] >= MIN_RANKING_SAMPLE
    ].copy()
    top_neighborhoods = eligible_neighborhoods.nlargest(10, "stabilized_price_per_sqm")

    city_table = markdown_table(
        city_summary,
        [
            "city_slug",
            "eligible_months",
            "latest_median_price_per_sqm",
            "raw_index_change_pct",
            "composition_adjusted_index_change_pct",
        ],
        {
            "eligible_months": "{:.0f}",
            "latest_median_price_per_sqm": "{:.0f}",
            "raw_index_change_pct": "{:+.1f}%",
            "composition_adjusted_index_change_pct": "{:+.1f}%",
        },
    )
    neighborhood_table = markdown_table(
        top_neighborhoods,
        [
            "neighborhood_slug",
            "listings",
            "stabilized_price_per_sqm",
            "price_vs_city_median_pct",
            "elevator_share",
            "parking_share",
        ],
        {
            "listings": "{:.0f}",
            "stabilized_price_per_sqm": "{:.0f}",
            "price_vs_city_median_pct": "{:+.1f}%",
            "elevator_share": "{:.1%}",
            "parking_share": "{:.1%}",
        },
    )

    successful = diagnostics.loc[diagnostics["status"].eq("ok")]
    median_r2 = (
        float(successful["in_sample_r2_mix_model"].median()) if not successful.empty else np.nan
    )
    report = f"""# Tehran & Major Cities Deep Dive

This extension moves the project from nationwide descriptive statistics into **local market segmentation, neighborhood analytics, approximate spatial visualization, and time-series market tracking**.

## Scope

- Tehran core listings: **{tehran_rows:,}**
- Major cities selected by nationwide listing count: **{', '.join(cities)}**
- Neighborhood ranking minimum: **{MIN_RANKING_SAMPLE:,} listings**
- Monthly trend minimum: **{MIN_MONTHLY_SAMPLE:,} listings per city-month**
- Common trend base month: **{common_base or 'not available; city-specific bases used'}**

## Tehran neighborhood analysis

Neighborhood medians are supplemented with a transparent sample-size stabilization toward the Tehran-wide median. This prevents a relatively small neighborhood sample from ranking above a much larger neighborhood solely because of sampling noise. The stabilized statistic is a benchmark, not a transaction-price estimate.

{neighborhood_table}

![Tehran neighborhood asking-price map](tehran_neighborhood_price_map.png)

![Tehran neighborhood price ranking](tehran_neighborhood_price_ranking.png)

![Tehran neighborhood amenity profile](tehran_neighborhood_amenity_profile.png)

## Major-city time trends

Raw median asking prices can move because the mix of listed properties changes. The deep-dive therefore adds a **composition-adjusted index**. Within each city, a regularized model controls for neighborhood, property category/type, building size, rooms, construction year, floor structure, advertiser type, and available amenities. The monthly median residual is then converted to an index.

The adjustment model intentionally excludes listing month, allowing the monthly residual pattern to capture time variation after observable listing mix is controlled. The median in-sample R² across successful city mix models is **{median_r2:.3f}**. This is a descriptive adjustment, not a repeat-sales or official house-price index.

{city_table}

![Major cities composition-adjusted price index](major_cities_composition_adjusted_price_index.png)

![Major cities monthly listing volume](major_cities_monthly_listing_volume.png)

![Major cities market positioning](major_cities_market_positioning.png)

## Interpretation guardrails

- The source contains **asking/listing prices**, not verified completed transaction prices.
- Neighborhood coordinates are approximate listing-location fields; the map shows neighborhood median coordinates, not legal boundaries or exact addresses.
- The time index controls for observed listing mix but cannot remove all changes in unobserved quality, seller behavior, duplicate listings, or platform coverage.
- A change in the index should be read as a change in the listed market represented by the dataset, not as an official Iranian property-price index.
- Neighborhood rankings are only shown above explicit sample thresholds and should not be used as standalone investment advice.
"""
    (OUTPUT_DIR / "CITY_DEEP_DIVE.md").write_text(report, encoding="utf-8")


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Tehran & Major Cities Deep Dive\n\n"
            "Portfolio companion notebook for neighborhood analytics, approximate spatial "
            "visualization, and composition-adjusted monthly asking-price trends."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path('..') if Path.cwd().name == 'notebooks' else Path('.')\n"
            "OUT = ROOT / 'outputs'\n"
            "neighborhoods = pd.read_csv(OUT / 'tehran_neighborhood_summary.csv')\n"
            "trends = pd.read_csv(OUT / 'major_cities_monthly_trends.csv', parse_dates=['listing_month'])\n"
            "city_summary = pd.read_csv(OUT / 'major_cities_deep_dive_summary.csv')"
        ),
        nbf.v4.new_markdown_cell("## Tehran neighborhood leaders"),
        nbf.v4.new_code_cell(
            "neighborhoods.query('listings >= 250')\n"
            "    .nlargest(15, 'stabilized_price_per_sqm')\n"
            "    [['neighborhood_slug','listings','stabilized_price_per_sqm',"
            "'price_vs_city_median_pct','elevator_share','parking_share']]"
        ),
        nbf.v4.new_markdown_cell("## Approximate Tehran neighborhood map"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(OUT / 'tehran_neighborhood_price_map.png')))"
        ),
        nbf.v4.new_markdown_cell("## Major-city composition-adjusted price index"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(OUT / 'major_cities_composition_adjusted_price_index.png')))"
        ),
        nbf.v4.new_markdown_cell("## Latest major-city trend summary"),
        nbf.v4.new_code_cell("city_summary"),
        nbf.v4.new_markdown_cell(
            "### Interpretation\n"
            "The index is a descriptive listing-market measure. It adjusts for observed "
            "property mix but is not a repeat-sales index and does not represent verified "
            "transaction-price appreciation."
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with (NOTEBOOK_DIR / "02_city_deep_dive.ipynb").open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared file not found: {DATA_FILE}\n"
            "Run `python src/prepare_sales_data.py` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    core, cutoffs = build_core_sample(df)
    cities = major_cities(core)

    neighborhoods = neighborhood_summary(core, TARGET_CITY)
    neighborhoods.to_csv(OUTPUT_DIR / "tehran_neighborhood_summary.csv", index=False)

    raw_monthly = monthly_raw_summary(core, cities)
    raw_monthly.to_csv(OUTPUT_DIR / "major_cities_raw_monthly.csv", index=False)

    trend_parts: list[pd.DataFrame] = []
    diagnostic_records: list[dict] = []
    for city in cities:
        trend, diagnostic = composition_adjusted_city_trend(core, city)
        diagnostic_records.append(diagnostic)
        if not trend.empty:
            trend_parts.append(trend)
        print(f"City trend complete: {city} ({diagnostic.get('status')})")

    if trend_parts:
        trends = pd.concat(trend_parts, ignore_index=True)
        trends, common_base = common_base_reindex(trends)
    else:
        trends = pd.DataFrame()
        common_base = None

    diagnostics = pd.DataFrame(diagnostic_records)
    city_summary = major_city_summary(trends) if not trends.empty else pd.DataFrame()

    trends.to_csv(OUTPUT_DIR / "major_cities_monthly_trends.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "major_cities_trend_model_diagnostics.csv", index=False)
    city_summary.to_csv(OUTPUT_DIR / "major_cities_deep_dive_summary.csv", index=False)

    tehran_map(neighborhoods)
    tehran_ranking_chart(neighborhoods)
    tehran_amenity_chart(neighborhoods)
    price_trend_chart(trends, common_base)
    listing_volume_chart(raw_monthly, cities)
    city_positioning_chart(core, cities)

    build_report(core, cities, neighborhoods, city_summary, diagnostics, common_base)
    build_notebook()

    metadata = {
        "target_city": TARGET_CITY,
        "major_cities": cities,
        "minimum_neighborhood_sample": MIN_NEIGHBORHOOD_SAMPLE,
        "minimum_neighborhood_ranking_sample": MIN_RANKING_SAMPLE,
        "minimum_monthly_sample": MIN_MONTHLY_SAMPLE,
        "common_trend_base_month": common_base,
        "core_rows": len(core),
        "tehran_rows": int(core["city_slug"].eq(TARGET_CITY).sum()),
        "tehran_neighborhoods_total": int(len(neighborhoods)),
        "tehran_neighborhoods_ranking_eligible": int(
            (neighborhoods["listings"] >= MIN_RANKING_SAMPLE).sum()
        ) if not neighborhoods.empty else 0,
        **cutoffs,
        "interpretation_note": (
            "Deep-dive metrics use asking prices from listings. The neighborhood map uses "
            "approximate coordinates, and the adjusted time series is a descriptive "
            "composition-adjusted listing index rather than an official transaction-price index."
        ),
    }
    with (OUTPUT_DIR / "city_deep_dive_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("City deep dive complete")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
