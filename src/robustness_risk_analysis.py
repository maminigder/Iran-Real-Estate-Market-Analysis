"""Add robustness, uncertainty, spatial-generalization, and market-tier analysis.

This module intentionally treats several outputs as *risk diagnostics* rather than
new point estimates. It tests how sensitive the portfolio conclusions are to
possible duplicate-like records, quantifies sampling uncertainty for Tehran
neighborhood medians, evaluates prediction on completely unseen neighborhoods,
and creates transparent Affordable / Mid-market / Premium neighborhood tiers.

Important caveat: the published analytical file does not include a verified
property-level identity key or ad text. Duplicate detection therefore uses
feature fingerprints and must be interpreted as duplicate-like / repeat-like
candidate detection, not proof that two ads represent the same dwelling.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "residential_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
TEHRAN_SUMMARY_FILE = OUTPUT_DIR / "tehran_neighborhood_summary.csv"

TARGET_CITY = "tehran"
MIN_NEIGHBORHOOD_SAMPLE = 250
BOOTSTRAP_REPS = 300
SPATIAL_CV_FOLDS = 3
MAX_TRAIN_ROWS_PER_FOLD = 60_000
RANDOM_STATE = 42

PERSIAN_ARABIC_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def build_core_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    core = df.loc[df["analysis_ready"].fillna(False)].copy()
    lower = float(core["asking_price_per_sqm"].quantile(0.005))
    upper = float(core["asking_price_per_sqm"].quantile(0.995))
    core = core.loc[core["asking_price_per_sqm"].between(lower, upper)].copy()
    return core, {"lower_ppsqm_cutoff": lower, "upper_ppsqm_cutoff": upper}


def extract_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.translate(PERSIAN_ARABIC_DIGITS)
    extracted = text.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def bool_numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )


def safe_pct_change(new: float, old: float) -> float:
    if not np.isfinite(old) or old == 0:
        return float("nan")
    return (new / old - 1.0) * 100.0


def hash_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
        else:
            normalized[column] = normalized[column].astype("string")
        normalized[column] = normalized[column].fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False)


def duplicate_risk_analysis(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Detect strict same-month fingerprints and broader cross-month repeats.

    Strict fingerprints include price, month, approximate location, size, room
    count, year, floor, property type, and selected amenities. Removing all but
    one row per strict fingerprint is used only as a sensitivity test.

    Repeat-like fingerprints omit month and price. Clusters spanning multiple
    months are reported as an *upper-bound risk signal* because identical units
    in the same building can legitimately share these characteristics.
    """
    work = core.copy()
    work["lat_round4"] = pd.to_numeric(work["location_latitude"], errors="coerce").round(4)
    work["lon_round4"] = pd.to_numeric(work["location_longitude"], errors="coerce").round(4)
    work["floor_numeric_fp"] = extract_number(work["floor"])

    strict_columns = [
        "city_slug",
        "neighborhood_slug",
        "listing_month",
        "asking_price",
        "building_size_sqm",
        "rooms_numeric",
        "construction_year_jalali",
        "floor_numeric_fp",
        "property_type",
        "lat_round4",
        "lon_round4",
        "has_elevator",
        "has_parking",
        "has_warehouse",
        "is_rebuilt",
    ]
    repeat_columns = [
        "city_slug",
        "neighborhood_slug",
        "building_size_sqm",
        "rooms_numeric",
        "construction_year_jalali",
        "floor_numeric_fp",
        "property_type",
        "lat_round4",
        "lon_round4",
        "has_elevator",
        "has_parking",
        "has_warehouse",
        "is_rebuilt",
    ]

    strict_hash = hash_rows(work, strict_columns)
    strict_counts = strict_hash.value_counts()
    strict_cluster_sizes = strict_hash.map(strict_counts)
    strict_candidate = strict_cluster_sizes.gt(1)
    strict_duplicate_extra = strict_hash.duplicated(keep="first")

    repeat_hash = hash_rows(work, repeat_columns)
    repeat_frame = pd.DataFrame(
        {
            "fingerprint": repeat_hash,
            "listing_month": pd.to_datetime(work["listing_month"], errors="coerce"),
        }
    )
    repeat_stats = (
        repeat_frame.groupby("fingerprint", dropna=False)
        .agg(rows=("fingerprint", "size"), months=("listing_month", "nunique"))
        .reset_index()
    )
    repeat_clusters = repeat_stats.loc[
        repeat_stats["rows"].gt(1) & repeat_stats["months"].gt(1)
    ].copy()
    repeat_keys = set(repeat_clusters["fingerprint"].tolist())
    repeat_candidate = repeat_hash.isin(repeat_keys)

    dedup = work.loc[~strict_duplicate_extra].copy()
    largest_cities = work["city_slug"].value_counts().head(5).index.tolist()
    scopes = ["national", TARGET_CITY, *[c for c in largest_cities if c != TARGET_CITY]]
    sensitivity_records: list[dict] = []

    for scope in scopes:
        before = work if scope == "national" else work.loc[work["city_slug"].eq(scope)]
        after = dedup if scope == "national" else dedup.loc[dedup["city_slug"].eq(scope)]
        if before.empty or after.empty:
            continue
        before_median = float(before["asking_price_per_sqm"].median())
        after_median = float(after["asking_price_per_sqm"].median())
        sensitivity_records.append(
            {
                "scope": scope,
                "rows_before": len(before),
                "rows_after_strict_dedup_sensitivity": len(after),
                "rows_removed": len(before) - len(after),
                "rows_removed_pct": (len(before) - len(after)) / len(before) * 100.0,
                "median_ppsqm_before": before_median,
                "median_ppsqm_after": after_median,
                "median_ppsqm_change_pct": safe_pct_change(after_median, before_median),
            }
        )

    cluster_distribution = (
        pd.DataFrame(
            {
                "cluster_size": strict_cluster_sizes.loc[strict_candidate].astype(int),
                "scope": np.where(work.loc[strict_candidate, "city_slug"].eq(TARGET_CITY), "tehran", "other"),
            }
        )
        .groupby(["scope", "cluster_size"])
        .size()
        .reset_index(name="rows_in_clusters")
    )

    metadata: dict[str, float | int] = {
        "core_rows": len(work),
        "strict_duplicate_like_candidate_rows": int(strict_candidate.sum()),
        "strict_duplicate_like_extra_rows": int(strict_duplicate_extra.sum()),
        "strict_duplicate_like_extra_share_pct": float(strict_duplicate_extra.mean() * 100.0),
        "strict_duplicate_like_clusters": int((strict_counts > 1).sum()),
        "cross_month_repeat_like_clusters": int(len(repeat_clusters)),
        "cross_month_repeat_like_rows": int(repeat_candidate.sum()),
        "cross_month_repeat_like_row_share_pct": float(repeat_candidate.mean() * 100.0),
    }

    return pd.DataFrame(sensitivity_records), cluster_distribution, metadata


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.median(values))
    if len(values) < 2:
        return median, median, median
    rng = np.random.default_rng(seed)
    draw_idx = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    medians = np.median(values[draw_idx], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return median, float(low), float(high)


def neighborhood_uncertainty(core: pd.DataFrame) -> pd.DataFrame:
    tehran = core.loc[
        core["city_slug"].eq(TARGET_CITY) & core["neighborhood_slug"].notna()
    ].copy()
    tehran["neighborhood_slug"] = tehran["neighborhood_slug"].astype(str).str.strip()
    counts = tehran["neighborhood_slug"].value_counts()
    eligible_names = counts.loc[counts >= MIN_NEIGHBORHOOD_SAMPLE].index
    tehran = tehran.loc[tehran["neighborhood_slug"].isin(eligible_names)].copy()

    existing = pd.read_csv(TEHRAN_SUMMARY_FILE) if TEHRAN_SUMMARY_FILE.exists() else pd.DataFrame()
    stable_lookup = {}
    if not existing.empty and "stabilized_price_per_sqm" in existing.columns:
        stable_lookup = existing.set_index("neighborhood_slug")["stabilized_price_per_sqm"].to_dict()

    records: list[dict] = []
    for idx, (name, group) in enumerate(tehran.groupby("neighborhood_slug")):
        values = group["asking_price_per_sqm"].to_numpy(dtype=float)
        median, low, high = bootstrap_median_ci(values, RANDOM_STATE + idx)
        width = high - low
        records.append(
            {
                "neighborhood_slug": name,
                "listings": len(group),
                "median_price_per_sqm": median,
                "ci95_low": low,
                "ci95_high": high,
                "ci95_width": width,
                "relative_ci_width_pct": width / median * 100.0 if median > 0 else np.nan,
                "stabilized_price_per_sqm": stable_lookup.get(name, median),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["uncertainty_rank"] = result["relative_ci_width_pct"].rank(method="min")
    result["price_rank_stabilized"] = result["stabilized_price_per_sqm"].rank(
        ascending=False, method="min"
    )
    return result.sort_values("stabilized_price_per_sqm", ascending=False)


def prepare_spatial_features(tehran: pd.DataFrame) -> pd.DataFrame:
    work = tehran.copy()
    work["log_building_size"] = np.log(work["building_size_sqm"].clip(lower=1))
    work["floor_numeric"] = extract_number(work["floor"])
    work["total_floors_numeric"] = extract_number(work["total_floors_count"])
    work["units_per_floor_numeric"] = extract_number(work["unit_per_floor"])
    for amenity in [
        "has_elevator",
        "has_parking",
        "has_warehouse",
        "is_rebuilt",
        "has_balcony",
    ]:
        work[f"{amenity}_numeric"] = bool_numeric(work[amenity])
    work["target_log_ppsqm"] = np.log(work["asking_price_per_sqm"])
    return work


def spatial_model(feature_names: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_features = [f for f in feature_names if f not in categorical_features]
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        sparse_threshold=0.0,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", regressor)])


def prediction_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    actual = np.exp(y_true_log)
    predicted = np.exp(y_pred_log)
    pct_error = np.abs(predicted - actual) / actual
    return {
        "r2_log_ppsqm": float(r2_score(y_true_log, y_pred_log)),
        "mae_log_ppsqm": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log_ppsqm": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "median_absolute_percentage_error": float(np.median(pct_error)),
        "share_predictions_within_20_percent": float(np.mean(pct_error <= 0.20)),
        "share_predictions_within_30_percent": float(np.mean(pct_error <= 0.30)),
    }


def spatial_holdout_analysis(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    tehran = core.loc[
        core["city_slug"].eq(TARGET_CITY) & core["neighborhood_slug"].notna()
    ].copy()
    tehran["neighborhood_slug"] = tehran["neighborhood_slug"].astype(str).str.strip()
    counts = tehran["neighborhood_slug"].value_counts()
    eligible = counts.loc[counts >= 100].index
    tehran = tehran.loc[tehran["neighborhood_slug"].isin(eligible)].copy()
    work = prepare_spatial_features(tehran)

    categorical_features = ["cat3_slug", "property_type", "user_type"]
    base_numeric = [
        "log_building_size",
        "rooms_numeric",
        "construction_year_jalali",
        "floor_numeric",
        "total_floors_numeric",
        "units_per_floor_numeric",
        "has_elevator_numeric",
        "has_parking_numeric",
        "has_warehouse_numeric",
        "is_rebuilt_numeric",
        "has_balcony_numeric",
    ]
    property_features = base_numeric + categorical_features
    spatial_features = property_features + [
        "location_latitude",
        "location_longitude",
        "location_radius",
    ]

    groups = work["neighborhood_slug"].astype(str).to_numpy()
    target = work["target_log_ppsqm"].to_numpy(dtype=float)
    gkf = GroupKFold(n_splits=SPATIAL_CV_FOLDS)

    fold_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(work, target, groups=groups), start=1):
        train = work.iloc[train_idx].copy()
        test = work.iloc[test_idx].copy()
        if len(train) > MAX_TRAIN_ROWS_PER_FOLD:
            train = train.sample(n=MAX_TRAIN_ROWS_PER_FOLD, random_state=RANDOM_STATE + fold)

        y_train = train["target_log_ppsqm"].to_numpy(dtype=float)
        y_test = test["target_log_ppsqm"].to_numpy(dtype=float)
        baseline_pred = np.full(len(test), np.median(y_train))

        property_model = spatial_model(property_features, categorical_features)
        property_model.fit(train[property_features], y_train)
        property_pred = property_model.predict(test[property_features])

        geo_model = spatial_model(spatial_features, categorical_features)
        geo_model.fit(train[spatial_features], y_train)
        geo_pred = geo_model.predict(test[spatial_features])

        for model_name, pred in [
            ("Tehran median baseline", baseline_pred),
            ("Property-only gradient boosting", property_pred),
            ("Spatial gradient boosting", geo_pred),
        ]:
            metrics = prediction_metrics(y_test, pred)
            fold_records.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "training_rows": len(train),
                    "test_rows": len(test),
                    "training_neighborhoods": train["neighborhood_slug"].nunique(),
                    "held_out_neighborhoods": test["neighborhood_slug"].nunique(),
                    **metrics,
                }
            )

        prediction_records.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "neighborhood_slug": test["neighborhood_slug"].to_numpy(),
                    "actual_log_ppsqm": y_test,
                    "baseline_pred_log_ppsqm": baseline_pred,
                    "property_pred_log_ppsqm": property_pred,
                    "spatial_pred_log_ppsqm": geo_pred,
                }
            )
        )

    folds = pd.DataFrame(fold_records)
    preds = pd.concat(prediction_records, ignore_index=True)
    aggregate_records: list[dict] = []
    mapping = {
        "Tehran median baseline": "baseline_pred_log_ppsqm",
        "Property-only gradient boosting": "property_pred_log_ppsqm",
        "Spatial gradient boosting": "spatial_pred_log_ppsqm",
    }
    y_all = preds["actual_log_ppsqm"].to_numpy(dtype=float)
    for model_name, column in mapping.items():
        aggregate_records.append(
            {"model": model_name, **prediction_metrics(y_all, preds[column].to_numpy(dtype=float))}
        )
    aggregate = pd.DataFrame(aggregate_records)

    metadata = {
        "eligible_tehran_rows": len(work),
        "eligible_tehran_neighborhoods": int(work["neighborhood_slug"].nunique()),
        "spatial_group_folds": SPATIAL_CV_FOLDS,
    }
    return aggregate, folds, metadata


def segment_tehran_market(
    core: pd.DataFrame, uncertainty: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    eligible = uncertainty.loc[uncertainty["listings"] >= MIN_NEIGHBORHOOD_SAMPLE].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    q1, q2 = eligible["stabilized_price_per_sqm"].quantile([1 / 3, 2 / 3]).tolist()
    eligible["market_segment"] = pd.cut(
        eligible["stabilized_price_per_sqm"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Affordable", "Mid-market", "Premium"],
        include_lowest=True,
    )

    mapping = eligible.set_index("neighborhood_slug")["market_segment"].astype(str).to_dict()
    tehran = core.loc[core["city_slug"].eq(TARGET_CITY)].copy()
    tehran["market_segment"] = tehran["neighborhood_slug"].map(mapping)
    tehran = tehran.loc[tehran["market_segment"].notna()].copy()

    amenity_cols = ["has_elevator", "has_parking", "has_warehouse", "is_rebuilt"]
    for amenity in amenity_cols:
        tehran[f"{amenity}_numeric"] = bool_numeric(tehran[amenity])

    profile = (
        tehran.groupby("market_segment", observed=True)
        .agg(
            listings=("asking_price", "size"),
            neighborhoods=("neighborhood_slug", "nunique"),
            median_asking_price=("asking_price", "median"),
            median_price_per_sqm=("asking_price_per_sqm", "median"),
            median_building_size_sqm=("building_size_sqm", "median"),
            median_rooms=("rooms_numeric", "median"),
            median_construction_year_jalali=("construction_year_jalali", "median"),
            elevator_share=("has_elevator_numeric", "mean"),
            parking_share=("has_parking_numeric", "mean"),
            warehouse_share=("has_warehouse_numeric", "mean"),
            rebuilt_share=("is_rebuilt_numeric", "mean"),
        )
        .reset_index()
    )
    profile["listing_share_pct"] = profile["listings"] / profile["listings"].sum() * 100.0
    profile["market_segment"] = pd.Categorical(
        profile["market_segment"],
        categories=["Affordable", "Mid-market", "Premium"],
        ordered=True,
    )
    profile = profile.sort_values("market_segment")

    neighborhood_table = eligible[
        [
            "neighborhood_slug",
            "market_segment",
            "listings",
            "median_price_per_sqm",
            "stabilized_price_per_sqm",
            "ci95_low",
            "ci95_high",
            "relative_ci_width_pct",
        ]
    ].sort_values(["market_segment", "stabilized_price_per_sqm"], ascending=[True, False])

    thresholds = {
        "affordable_upper_stabilized_ppsqm": float(q1),
        "mid_market_upper_stabilized_ppsqm": float(q2),
    }
    return neighborhood_table, profile, thresholds


def save_duplicate_chart(sensitivity: pd.DataFrame) -> None:
    if sensitivity.empty:
        return
    plot = sensitivity.sort_values("median_ppsqm_change_pct")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot["scope"], plot["median_ppsqm_change_pct"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Sensitivity of Median Asking Price/m² to Strict Duplicate-like Removal")
    ax.set_xlabel("Change in median asking price/m² (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "duplicate_sensitivity.png", dpi=180)
    plt.close(fig)


def save_uncertainty_chart(uncertainty: pd.DataFrame) -> None:
    if uncertainty.empty:
        return
    top = uncertainty.head(15).sort_values("stabilized_price_per_sqm")
    center = top["median_price_per_sqm"] / 1_000_000
    lower = (top["median_price_per_sqm"] - top["ci95_low"]) / 1_000_000
    upper = (top["ci95_high"] - top["median_price_per_sqm"]) / 1_000_000
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        center,
        top["neighborhood_slug"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
    )
    ax.set_title("Tehran Neighborhood Median Asking Price/m² with Bootstrap 95% CIs")
    ax.set_xlabel("Median asking price/m² (millions of source units)")
    ax.set_ylabel("Neighborhood")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tehran_neighborhood_uncertainty.png", dpi=180)
    plt.close(fig)


def save_spatial_chart(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(metrics["model"], metrics["r2_log_ppsqm"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Unseen-Neighborhood Spatial Holdout Performance")
    ax.set_xlabel("R² on log asking price/m²")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_holdout_model_comparison.png", dpi=180)
    plt.close(fig)


def save_segment_chart(profile: pd.DataFrame) -> None:
    if profile.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(profile["market_segment"].astype(str), profile["median_price_per_sqm"] / 1_000_000)
    ax.set_title("Tehran Market Segmentation by Neighborhood Price Tier")
    ax.set_ylabel("Median asking price/m² (millions of source units)")
    ax.set_xlabel("Market segment")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tehran_market_segments.png", dpi=180)
    plt.close(fig)


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def build_report(
    duplicate_meta: dict,
    duplicate_sensitivity: pd.DataFrame,
    uncertainty: pd.DataFrame,
    spatial_metrics: pd.DataFrame,
    segment_profile: pd.DataFrame,
    thresholds: dict,
) -> str:
    national_row = duplicate_sensitivity.loc[duplicate_sensitivity["scope"].eq("national")]
    national_change = float(national_row.iloc[0]["median_ppsqm_change_pct"]) if not national_row.empty else np.nan

    spatial_lookup = spatial_metrics.set_index("model") if not spatial_metrics.empty else pd.DataFrame()
    spatial_r2 = (
        float(spatial_lookup.loc["Spatial gradient boosting", "r2_log_ppsqm"])
        if not spatial_lookup.empty and "Spatial gradient boosting" in spatial_lookup.index
        else np.nan
    )
    spatial_mape = (
        float(spatial_lookup.loc["Spatial gradient boosting", "median_absolute_percentage_error"])
        if not spatial_lookup.empty and "Spatial gradient boosting" in spatial_lookup.index
        else np.nan
    )
    property_r2 = (
        float(spatial_lookup.loc["Property-only gradient boosting", "r2_log_ppsqm"])
        if not spatial_lookup.empty and "Property-only gradient boosting" in spatial_lookup.index
        else np.nan
    )

    narrowest = uncertainty.nsmallest(5, "relative_ci_width_pct") if not uncertainty.empty else pd.DataFrame()
    widest = uncertainty.nlargest(5, "relative_ci_width_pct") if not uncertainty.empty else pd.DataFrame()

    lines = [
        "# Robustness & Risk Analysis",
        "",
        "This layer asks a different question from the main predictive work: **how fragile are the conclusions?** It tests duplicate-like record sensitivity, neighborhood sampling uncertainty, generalization to completely unseen Tehran neighborhoods, and transparent market-tier segmentation.",
        "",
        "## 1. Duplicate / repeat-listing risk",
        "",
        f"The strict same-month fingerprint flags **{duplicate_meta['strict_duplicate_like_candidate_rows']:,} candidate rows** and **{duplicate_meta['strict_duplicate_like_extra_rows']:,} extra rows** beyond the first copy. That is **{duplicate_meta['strict_duplicate_like_extra_share_pct']:.2f}%** of the core sample.",
        "",
        f"After removing only those strict duplicate-like extras, the nationwide median asking price/m² changes by **{national_change:+.3f}%**. This is a sensitivity test, not a claim that every flagged record is a true duplicate.",
        "",
        f"A broader fingerprint that ignores month and asking price identifies **{duplicate_meta['cross_month_repeat_like_clusters']:,} cross-month repeat-like clusters** covering **{duplicate_meta['cross_month_repeat_like_rows']:,} rows**. This broader flag is deliberately treated as an upper-bound risk signal because distinct units can share the same observable characteristics.",
        "",
        "![Duplicate sensitivity](duplicate_sensitivity.png)",
        "",
        "## 2. Tehran neighborhood uncertainty",
        "",
        f"Bootstrap 95% confidence intervals are calculated for the median asking price/m² of every Tehran neighborhood with at least **{MIN_NEIGHBORHOOD_SAMPLE} listings**, using **{BOOTSTRAP_REPS} bootstrap replications** per neighborhood.",
        "",
        "These intervals quantify **sampling uncertainty conditional on the observed Divar listings**. They do not capture platform-selection bias, seller strategy, duplicate risk, or unobserved property quality.",
        "",
        "![Neighborhood uncertainty](tehran_neighborhood_uncertainty.png)",
        "",
        "### Most statistically precise eligible neighborhood medians",
        "",
        "| Neighborhood | Listings | Median price/m² | Relative CI width |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in narrowest.iterrows():
        lines.append(
            f"| {row['neighborhood_slug']} | {int(row['listings']):,} | {row['median_price_per_sqm']:,.0f} | {row['relative_ci_width_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "### Least precise eligible neighborhood medians",
            "",
            "| Neighborhood | Listings | Median price/m² | Relative CI width |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in widest.iterrows():
        lines.append(
            f"| {row['neighborhood_slug']} | {int(row['listings']):,} | {row['median_price_per_sqm']:,.0f} | {row['relative_ci_width_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 3. Spatial holdout: completely unseen neighborhoods",
            "",
            f"A grouped cross-validation design holds out entire Tehran neighborhoods. No listing from a held-out neighborhood is available during training. The spatial model **does not use the neighborhood name**; it must generalize from property characteristics plus approximate latitude/longitude/privacy radius.",
            "",
            f"Across the out-of-neighborhood predictions, the property-only nonlinear model achieves **R² = {property_r2:.3f}** on log price/m², while the spatial model achieves **R² = {spatial_r2:.3f}** with a median absolute percentage error of **{spatial_mape * 100:.1f}%**.",
            "",
            "This is deliberately a harder validation problem than a random row split and gives a more realistic view of geographic generalization risk.",
            "",
            "![Spatial holdout comparison](spatial_holdout_model_comparison.png)",
            "",
            "## 4. Affordable / Mid-market / Premium segmentation",
            "",
            "Eligible Tehran neighborhoods are divided into three equal-count tiers using the **sample-size-stabilized neighborhood price benchmark**. The labels are relative analytical tiers, not legal or investment classifications.",
            "",
            f"- Affordable: stabilized benchmark ≤ **{thresholds.get('affordable_upper_stabilized_ppsqm', np.nan):,.0f}** source units/m²",
            f"- Mid-market: above Affordable and ≤ **{thresholds.get('mid_market_upper_stabilized_ppsqm', np.nan):,.0f}** source units/m²",
            "- Premium: above the Mid-market threshold",
            "",
            "| Segment | Neighborhoods | Listings | Listing share | Median price/m² | Median size | Elevator share | Parking share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in segment_profile.iterrows():
        lines.append(
            f"| {row['market_segment']} | {int(row['neighborhoods'])} | {int(row['listings']):,} | {row['listing_share_pct']:.1f}% | {row['median_price_per_sqm']:,.0f} | {row['median_building_size_sqm']:.0f} m² | {fmt_pct(row['elevator_share'])} | {fmt_pct(row['parking_share'])} |"
        )

    lines.extend(
        [
            "",
            "![Tehran market tiers](tehran_market_segments.png)",
            "",
            "## Risk interpretation",
            "",
            "- Duplicate fingerprints are probabilistic diagnostics because no verified property identity key is available in the analytical file.",
            "- Bootstrap intervals describe uncertainty of the observed listing sample, not uncertainty about the entire housing stock or completed transactions.",
            "- Spatial holdout performance can still benefit from approximate coordinates and may not transfer to cities with different market structure.",
            "- Market tiers are relative to eligible Tehran neighborhoods in this dataset and period; thresholds should not be reused as permanent market definitions.",
            "- All monetary values remain in source units until the source denomination is independently validated.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Robustness & Risk Analysis\n\nPortfolio companion notebook for duplicate sensitivity, neighborhood uncertainty, unseen-neighborhood validation, and Tehran market segmentation."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path('..') if Path.cwd().name == 'notebooks' else Path('.')\n"
            "OUT = ROOT / 'outputs'\n"
            "dup = pd.read_csv(OUT / 'duplicate_sensitivity.csv')\n"
            "uncertainty = pd.read_csv(OUT / 'tehran_neighborhood_uncertainty.csv')\n"
            "spatial = pd.read_csv(OUT / 'spatial_holdout_metrics.csv')\n"
            "segments = pd.read_csv(OUT / 'tehran_market_segment_profile.csv')"
        ),
        nbf.v4.new_markdown_cell("## Duplicate sensitivity"),
        nbf.v4.new_code_cell("dup"),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'duplicate_sensitivity.png')))"),
        nbf.v4.new_markdown_cell("## Neighborhood median uncertainty"),
        nbf.v4.new_code_cell(
            "uncertainty[['neighborhood_slug','listings','median_price_per_sqm','ci95_low','ci95_high','relative_ci_width_pct']].head(15)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'tehran_neighborhood_uncertainty.png')))"),
        nbf.v4.new_markdown_cell("## Unseen-neighborhood spatial holdout"),
        nbf.v4.new_code_cell("spatial"),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'spatial_holdout_model_comparison.png')))"),
        nbf.v4.new_markdown_cell("## Tehran market tiers"),
        nbf.v4.new_code_cell("segments"),
        nbf.v4.new_code_cell("display(Image(filename=str(OUT / 'tehran_market_segments.png')))"),
        nbf.v4.new_markdown_cell(
            "### Interpretation\nAll four sections are sensitivity / risk diagnostics. They do not convert listing data into verified transaction evidence or causal valuation estimates."
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_DIR / "03_robustness_risk_analysis.ipynb")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared file not found: {DATA_FILE}\nRun `python src/prepare_sales_data.py` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    core, cutoffs = build_core_sample(df)

    duplicate_sensitivity, duplicate_clusters, duplicate_meta = duplicate_risk_analysis(core)
    duplicate_sensitivity.to_csv(OUTPUT_DIR / "duplicate_sensitivity.csv", index=False)
    duplicate_clusters.to_csv(OUTPUT_DIR / "duplicate_cluster_distribution.csv", index=False)
    save_duplicate_chart(duplicate_sensitivity)

    uncertainty = neighborhood_uncertainty(core)
    uncertainty.to_csv(OUTPUT_DIR / "tehran_neighborhood_uncertainty.csv", index=False)
    save_uncertainty_chart(uncertainty)

    spatial_metrics, spatial_folds, spatial_meta = spatial_holdout_analysis(core)
    spatial_metrics.to_csv(OUTPUT_DIR / "spatial_holdout_metrics.csv", index=False)
    spatial_folds.to_csv(OUTPUT_DIR / "spatial_holdout_fold_metrics.csv", index=False)
    save_spatial_chart(spatial_metrics)

    segment_neighborhoods, segment_profile, thresholds = segment_tehran_market(core, uncertainty)
    segment_neighborhoods.to_csv(OUTPUT_DIR / "tehran_neighborhood_segments.csv", index=False)
    segment_profile.to_csv(OUTPUT_DIR / "tehran_market_segment_profile.csv", index=False)
    save_segment_chart(segment_profile)

    metadata = {
        **cutoffs,
        **duplicate_meta,
        **spatial_meta,
        **thresholds,
        "bootstrap_replications_per_neighborhood": BOOTSTRAP_REPS,
        "minimum_neighborhood_sample_for_uncertainty_and_tiers": MIN_NEIGHBORHOOD_SAMPLE,
        "interpretation_note": (
            "Duplicate detection is fingerprint-based and probabilistic; confidence intervals quantify sampling "
            "uncertainty in observed listings; spatial holdout tests unseen-neighborhood generalization; market "
            "tiers are relative analytical segments rather than certified valuation classes."
        ),
    }
    with (OUTPUT_DIR / "robustness_risk_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    report = build_report(
        duplicate_meta,
        duplicate_sensitivity,
        uncertainty,
        spatial_metrics,
        segment_profile,
        thresholds,
    )
    (OUTPUT_DIR / "ROBUSTNESS_RISK_ANALYSIS.md").write_text(report, encoding="utf-8")
    build_notebook()

    print("Robustness / risk analysis complete")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
