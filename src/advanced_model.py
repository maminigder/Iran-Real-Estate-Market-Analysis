"""Build an advanced, portfolio-ready hedonic pricing model.

The model is intentionally designed for interpretability rather than black-box
prediction. It estimates log asking price per square metre using a regularized
linear (Ridge) hedonic specification with location, property characteristics,
listing time, and amenity controls.

Key safeguards:
- temporal holdout rather than a random test split;
- neighborhood-within-city controls;
- log target to reduce the influence of extreme prices;
- explicit naive location baseline;
- permutation importance on raw feature groups;
- amenity results described as adjusted associations, never causal premiums.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from market_analysis import build_core_sample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "residential_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 220_000
MAX_TEST_ROWS = 80_000
IMPORTANCE_SAMPLE_ROWS = 12_000
MIN_CATEGORY_FREQUENCY = 100
RIDGE_ALPHA = 5.0

AMENITIES = [
    "has_elevator",
    "has_parking",
    "has_warehouse",
    "is_rebuilt",
    "has_balcony",
    "has_heating_system",
    "has_cooling_system",
]

NUMERIC_FEATURES = [
    "log_building_size",
    "rooms_numeric",
    "construction_year_jalali",
    "floor_numeric",
    "total_floors_numeric",
    "units_per_floor_numeric",
    "listing_month_index",
]

CATEGORICAL_FEATURES = [
    "location_key",
    "property_type",
    "cat3_slug",
    "user_type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + AMENITIES

PERSIAN_ARABIC_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def extract_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.translate(PERSIAN_ARABIC_DIGITS)
    extracted = text.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def normalize_boolean_label(series: pd.Series) -> pd.Series:
    """Represent boolean-like values as true/false/unknown categories."""
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": "true",
        "false": "false",
        "1": "true",
        "0": "false",
        "yes": "true",
        "no": "false",
    }
    return text.map(mapping).fillna("unknown").astype(str)


def prepare_model_frame(core: pd.DataFrame) -> pd.DataFrame:
    df = core.copy()
    df = df.loc[df["listing_month"].notna()].copy()

    df["log_building_size"] = np.log(df["building_size_sqm"].astype(float))
    df["floor_numeric"] = extract_number(df["floor"])
    df["total_floors_numeric"] = extract_number(df["total_floors_count"])
    df["units_per_floor_numeric"] = extract_number(df["unit_per_floor"])

    month_number = df["listing_month"].dt.year * 12 + df["listing_month"].dt.month
    df["listing_month_index"] = month_number - int(month_number.min())

    city = df["city_slug"].astype("string").fillna("__missing_city__")
    neighborhood = df["neighborhood_slug"].astype("string").fillna("__unknown_neighborhood__")
    df["location_key"] = (city + "::" + neighborhood).astype(str)

    for col in CATEGORICAL_FEATURES:
        if col == "location_key":
            continue
        df[col] = df[col].astype("string").fillna("__missing__").astype(str)

    for col in AMENITIES:
        df[col] = normalize_boolean_label(df[col])

    df["target_log_ppsqm"] = np.log(df["asking_price_per_sqm"].astype(float))
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_month = pd.Timestamp(df["listing_month"].max()).to_period("M").to_timestamp()
    cutoff = (max_month.to_period("M") - 5).to_timestamp()

    train = df.loc[df["listing_month"] < cutoff].copy()
    test = df.loc[df["listing_month"] >= cutoff].copy()

    if len(train) < 10_000 or len(test) < 5_000:
        raise ValueError(
            "Temporal split produced an unexpectedly small sample; inspect date coverage before modeling."
        )
    return train, test, cutoff


def sample_rows(df: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(df) <= maximum:
        return df.copy()
    return df.sample(n=maximum, random_state=RANDOM_STATE)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_FREQUENCY,
        sparse_output=True,
    )
    amenity_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_encoder, CATEGORICAL_FEATURES),
            ("amenities", amenity_encoder, AMENITIES),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=RIDGE_ALPHA, solver="lsqr")),
        ]
    )


def location_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    location_median = train.groupby("location_key")["target_log_ppsqm"].median()
    city_median = train.groupby("city_slug")["target_log_ppsqm"].median()
    global_median = float(train["target_log_ppsqm"].median())

    pred = test["location_key"].map(location_median)
    missing = pred.isna()
    if missing.any():
        pred.loc[missing] = test.loc[missing, "city_slug"].map(city_median)
    return pred.fillna(global_median).to_numpy(dtype=float)


def evaluate_predictions(
    y_true_log: np.ndarray,
    pred_log: np.ndarray,
) -> dict[str, float]:
    actual = np.exp(y_true_log)
    predicted = np.exp(pred_log)
    absolute_percentage_error = np.abs(predicted - actual) / actual

    return {
        "r2_log_ppsqm": float(r2_score(y_true_log, pred_log)),
        "mae_log_ppsqm": float(mean_absolute_error(y_true_log, pred_log)),
        "rmse_log_ppsqm": float(np.sqrt(mean_squared_error(y_true_log, pred_log))),
        "median_absolute_percentage_error": float(np.median(absolute_percentage_error)),
        "share_predictions_within_20_percent": float(
            np.mean(absolute_percentage_error <= 0.20)
        ),
        "share_predictions_within_30_percent": float(
            np.mean(absolute_percentage_error <= 0.30)
        ),
    }


def permutation_group_importance(
    model: Pipeline,
    test: pd.DataFrame,
) -> pd.DataFrame:
    sampled = sample_rows(test, IMPORTANCE_SAMPLE_ROWS)
    X = sampled[MODEL_FEATURES].copy()
    y = sampled["target_log_ppsqm"].to_numpy(dtype=float)
    baseline_r2 = float(model.score(X, y))
    rng = np.random.default_rng(RANDOM_STATE)

    records: list[dict] = []
    for feature in MODEL_FEATURES:
        permuted = X.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        permuted_r2 = float(model.score(permuted, y))
        records.append(
            {
                "feature_group": feature,
                "baseline_r2": baseline_r2,
                "r2_after_permutation": permuted_r2,
                "r2_drop": baseline_r2 - permuted_r2,
            }
        )

    return pd.DataFrame(records).sort_values("r2_drop", ascending=False)


def amenity_associations(
    model: Pipeline,
    model_frame: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    counterfactual_sample = sample_rows(test, IMPORTANCE_SAMPLE_ROWS)
    base_X = counterfactual_sample[MODEL_FEATURES].copy()
    records: list[dict] = []

    for amenity in AMENITIES:
        known = model_frame.loc[model_frame[amenity].isin(["true", "false"])].copy()
        medians = known.groupby(amenity)["asking_price_per_sqm"].median()
        raw_pct = np.nan
        if "true" in medians.index and "false" in medians.index and medians["false"] > 0:
            raw_pct = 100.0 * (float(medians["true"]) / float(medians["false"]) - 1.0)

        X_false = base_X.copy()
        X_true = base_X.copy()
        X_false[amenity] = "false"
        X_true[amenity] = "true"
        pred_false = model.predict(X_false)
        pred_true = model.predict(X_true)
        adjusted_pct = 100.0 * (np.exp(np.median(pred_true - pred_false)) - 1.0)

        records.append(
            {
                "amenity": amenity,
                "known_observations": int(len(known)),
                "raw_median_ppsqm_association_pct": float(raw_pct) if pd.notna(raw_pct) else np.nan,
                "model_adjusted_association_pct": float(adjusted_pct),
                "interpretation": (
                    "Association conditional on modeled controls; not a causal price premium."
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "model_adjusted_association_pct",
        ascending=False,
    )


def save_model_charts(
    metrics: dict,
    y_true_log: np.ndarray,
    pred_log: np.ndarray,
    importance: pd.DataFrame,
    amenities: pd.DataFrame,
) -> None:
    # Model vs baseline validation performance.
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Location baseline", "Hedonic Ridge model"]
    values = [metrics["baseline_r2_log_ppsqm"], metrics["model"]["r2_log_ppsqm"]]
    ax.bar(labels, values)
    ax.set_ylabel("Out-of-time R² on log asking price per m²")
    ax.set_title("Validation Performance: Baseline vs Hedonic Model")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "advanced_model_validation.png", dpi=180)
    plt.close(fig)

    # Actual vs predicted values on a deterministic subsample.
    n = min(15_000, len(y_true_log))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_true_log), size=n, replace=False)
    actual_m = np.exp(y_true_log[idx]) / 1_000_000
    predicted_m = np.exp(pred_log[idx]) / 1_000_000
    upper = float(np.quantile(np.concatenate([actual_m, predicted_m]), 0.99))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual_m, predicted_m, s=6, alpha=0.2)
    ax.plot([0, upper], [0, upper], linestyle="--")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Actual asking price per m² (millions of source units)")
    ax.set_ylabel("Predicted asking price per m² (millions of source units)")
    ax.set_title("Out-of-Time Actual vs Predicted Asking Price per m²")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "advanced_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    top_importance = importance.head(12).sort_values("r2_drop")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_importance["feature_group"], top_importance["r2_drop"])
    ax.set_xlabel("Decrease in validation R² after permutation")
    ax.set_ylabel("Feature group")
    ax.set_title("Raw Feature-Group Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "advanced_feature_importance.png", dpi=180)
    plt.close(fig)

    amenity_plot = amenities.sort_values("model_adjusted_association_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        amenity_plot["amenity"],
        amenity_plot["model_adjusted_association_pct"],
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Model-adjusted association with asking price per m² (%)")
    ax.set_ylabel("Amenity")
    ax.set_title("Adjusted Amenity Associations (Not Causal Effects)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "advanced_amenity_associations.png", dpi=180)
    plt.close(fig)


def write_model_card(metrics: dict) -> None:
    text = f"""# Hedonic Model Card

## Purpose

This model estimates **asking price per square metre** for Iranian residential-sale listings using observable property, location, time, and amenity characteristics. It is built for market analysis and portfolio demonstration, not automated valuation or investment decisions.

## Model

- Estimator: Ridge regression (regularized linear hedonic model)
- Target: natural log of asking price per m²
- Ridge alpha: {RIDGE_ALPHA}
- Location control: city + neighborhood composite category
- Rare-category threshold: {MIN_CATEGORY_FREQUENCY} training observations
- Training maximum: {MAX_TRAIN_ROWS:,} rows
- Validation maximum: {MAX_TEST_ROWS:,} rows
- Validation design: latest six observed listing months held out from model fitting

## Validation

- Temporal cutoff: {metrics['temporal_holdout_start']}
- Training rows used: {metrics['training_rows_used']:,}
- Validation rows used: {metrics['validation_rows_used']:,}
- Location-baseline R²: {metrics['baseline_r2_log_ppsqm']:.3f}
- Hedonic-model R²: {metrics['model']['r2_log_ppsqm']:.3f}
- Median absolute percentage error: {metrics['model']['median_absolute_percentage_error']:.1%}
- Predictions within 20%: {metrics['model']['share_predictions_within_20_percent']:.1%}

## Interpretation

The model captures conditional associations after controlling for included variables. The adjusted amenity estimates are **not causal premiums**. Unobserved quality, micro-location, interior condition, seller strategy, duplicated listings, and selection into Divar can still influence estimated relationships.

## Intended use

Appropriate uses include exploratory market analysis, portfolio demonstration, reproducible research, and hypothesis generation. It should not be used as a certified appraisal, transaction-price estimator, lending model, or investment recommendation engine.
"""
    (OUTPUT_DIR / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def write_executive_summary(
    metrics: dict,
    amenities: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    top_feature = importance.iloc[0]
    top_amenities = amenities.reindex(
        amenities["model_adjusted_association_pct"].abs().sort_values(ascending=False).index
    ).head(3)

    amenity_lines = "\n".join(
        f"- **{row.amenity}**: {row.model_adjusted_association_pct:+.1f}% adjusted association"
        for row in top_amenities.itertuples()
    )

    text = f"""# Executive Summary — Advanced Hedonic Analysis

## What the advanced model adds

The descriptive phase shows where prices differ. The hedonic model asks a harder question: **how much of the variation in asking price per m² can be explained after simultaneously accounting for location, size, rooms, construction year, floor structure, listing time, property type, and amenities?**

A regularized Ridge model is trained on pre-holdout data and tested only on the latest six observed listing months. This makes validation more realistic than a random train/test split because the model must generalize forward in time.

## Validation result

- Training rows used: **{metrics['training_rows_used']:,}**
- Out-of-time validation rows used: **{metrics['validation_rows_used']:,}**
- Validation period starts: **{metrics['temporal_holdout_start']}**
- Naive location-baseline R²: **{metrics['baseline_r2_log_ppsqm']:.3f}**
- Hedonic-model R²: **{metrics['model']['r2_log_ppsqm']:.3f}**
- Median absolute percentage error: **{metrics['model']['median_absolute_percentage_error']:.1%}**
- Share of predictions within 20% of the listed price per m²: **{metrics['model']['share_predictions_within_20_percent']:.1%}**

The feature group with the largest validation-performance loss when permuted is **{top_feature['feature_group']}** (R² drop: {top_feature['r2_drop']:.3f}), indicating that it carries substantial predictive information in the fitted model.

## Amenity associations

The strongest modeled amenity associations by absolute magnitude are:

{amenity_lines}

These percentages are **conditional model associations, not causal price premiums**. Amenities can proxy for neighborhood quality, building class, maintenance level, developer quality, and other unobserved characteristics.

## Business interpretation

The model makes the project more decision-relevant in three ways. First, it separates raw price differences from differences that remain after controlling for observable property characteristics. Second, the temporal holdout shows whether the relationships retain predictive value on later listings. Third, the importance analysis identifies which information groups contribute most to explaining cross-sectional asking-price variation.

## Important limitations

- Divar prices are asking/listing values, not verified transaction prices.
- The source is not a full census of the Iranian housing market.
- The model does not directly observe architectural style, interior design quality, exact address, financing conditions, or seller urgency.
- Adjusted associations should not be interpreted as causal effects.
- The source currency denomination remains labeled as source units until independently validated.

For technical detail, see `outputs/MODEL_CARD.md`, `docs/methodology.md`, and the portfolio notebook in `notebooks/`.
"""
    (OUTPUT_DIR / "EXECUTIVE_SUMMARY.md").write_text(text, encoding="utf-8")


def write_portfolio_notebook(
    metrics: dict,
    amenities: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()

    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Iran Real Estate Market Analysis — Advanced Portfolio Walkthrough\n\n"
            "This notebook presents the advanced hedonic-model layer of the project. "
            "The underlying pipeline is reproducible from the source scripts and the official Divar dataset."
        ),
        nbf.v4.new_markdown_cell(
            f"## Out-of-time validation\n\n"
            f"The latest six observed listing months are held out from training. "
            f"The validation period begins **{metrics['temporal_holdout_start']}**.\n\n"
            f"- Hedonic model R²: **{metrics['model']['r2_log_ppsqm']:.3f}**\n"
            f"- Location baseline R²: **{metrics['baseline_r2_log_ppsqm']:.3f}**\n"
            f"- Median absolute percentage error: **{metrics['model']['median_absolute_percentage_error']:.1%}**\n"
            f"- Predictions within 20%: **{metrics['model']['share_predictions_within_20_percent']:.1%}**"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path('..')\n"
            "amenities = pd.read_csv(ROOT / 'outputs' / 'advanced_amenity_associations.csv')\n"
            "importance = pd.read_csv(ROOT / 'outputs' / 'advanced_feature_importance.csv')\n"
            "display(amenities)\n"
            "display(importance.head(12))"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation chart\n\n![Validation](../outputs/advanced_model_validation.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Actual vs predicted asking price per m²\n\n"
            "![Actual vs predicted](../outputs/advanced_actual_vs_predicted.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Feature-group importance\n\n"
            "Permutation is applied one raw feature group at a time on the temporal holdout. "
            "A larger R² drop means the model relies more heavily on that information group.\n\n"
            "![Importance](../outputs/advanced_feature_importance.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Amenity associations\n\n"
            "These are adjusted model associations after included controls and **must not be read as causal premiums**.\n\n"
            "![Amenities](../outputs/advanced_amenity_associations.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibility\n\n"
            "Run the pipeline in order:\n\n"
            "```bash\n"
            "python src/download_data.py\n"
            "python src/prepare_sales_data.py\n"
            "python src/market_analysis.py\n"
            "python src/advanced_model.py\n"
            "```\n\n"
            "The GitHub Actions workflow performs the same process automatically."
        ),
    ]

    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }

    nbf.write(nb, NOTEBOOK_DIR / "01_advanced_market_analysis.ipynb")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared file not found: {DATA_FILE}\n"
            "Run `python src/prepare_sales_data.py` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATA_FILE)
    core, cutoffs = build_core_sample(raw)
    model_frame = prepare_model_frame(core)
    train_full, test_full, cutoff = temporal_split(model_frame)

    train = sample_rows(train_full, MAX_TRAIN_ROWS)
    test = sample_rows(test_full, MAX_TEST_ROWS)

    model = build_pipeline()
    model.fit(train[MODEL_FEATURES], train["target_log_ppsqm"])

    y_test = test["target_log_ppsqm"].to_numpy(dtype=float)
    pred_model = model.predict(test[MODEL_FEATURES])
    pred_baseline = location_baseline(train, test)

    model_metrics = evaluate_predictions(y_test, pred_model)
    baseline_metrics = evaluate_predictions(y_test, pred_baseline)

    metrics = {
        "model_name": "regularized_hedonic_ridge",
        "target": "log_asking_price_per_sqm",
        "ridge_alpha": RIDGE_ALPHA,
        "temporal_holdout_start": cutoff.date().isoformat(),
        "training_period_start": train_full["listing_month"].min().date().isoformat(),
        "training_period_end": train_full["listing_month"].max().date().isoformat(),
        "validation_period_start": test_full["listing_month"].min().date().isoformat(),
        "validation_period_end": test_full["listing_month"].max().date().isoformat(),
        "training_rows_available": int(len(train_full)),
        "training_rows_used": int(len(train)),
        "validation_rows_available": int(len(test_full)),
        "validation_rows_used": int(len(test)),
        "baseline_r2_log_ppsqm": baseline_metrics["r2_log_ppsqm"],
        "baseline": baseline_metrics,
        "model": model_metrics,
        "core_sample_price_tail_cutoffs": cutoffs,
        "interpretation_note": (
            "Model outputs are conditional associations and out-of-time predictions of listing prices. "
            "They are not causal effects or verified transaction-price estimates."
        ),
    }

    importance = permutation_group_importance(model, test)
    amenity_table = amenity_associations(model, model_frame, test)

    importance.to_csv(OUTPUT_DIR / "advanced_feature_importance.csv", index=False)
    amenity_table.to_csv(OUTPUT_DIR / "advanced_amenity_associations.csv", index=False)
    with (OUTPUT_DIR / "advanced_model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_model_charts(metrics, y_test, pred_model, importance, amenity_table)
    write_model_card(metrics)
    write_executive_summary(metrics, amenity_table, importance)
    write_portfolio_notebook(metrics, amenity_table, importance)

    print("Advanced hedonic analysis complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nTop feature groups:")
    print(importance.head(10).to_string(index=False))
    print("\nAdjusted amenity associations:")
    print(amenity_table.to_string(index=False))


if __name__ == "__main__":
    main()
