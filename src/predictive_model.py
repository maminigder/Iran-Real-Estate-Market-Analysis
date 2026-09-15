"""Add a nonlinear predictive benchmark to the Iran real-estate project.

The existing Ridge model remains the interpretable hedonic layer. This script
adds a complementary nonlinear model built for predictive performance. It uses
cross-fitted target encoding for high-cardinality location categories and a
HistGradientBoostingRegressor, then evaluates strictly on the same out-of-time
holdout used by the hedonic model.

The purpose is model comparison, not automated appraisal. All targets remain
advertised/listing prices rather than verified transaction prices.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from advanced_model import (
    AMENITIES,
    IMPORTANCE_SAMPLE_ROWS,
    MAX_TEST_ROWS,
    MAX_TRAIN_ROWS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    evaluate_predictions,
    location_baseline,
    prepare_model_frame,
    sample_rows,
    temporal_split,
)
from market_analysis import build_core_sample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "residential_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
ADVANCED_METRICS_FILE = OUTPUT_DIR / "advanced_model_metrics.json"

PREDICTIVE_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "location_latitude",
    "location_longitude",
    "location_radius",
]

PREDICTIVE_CATEGORICAL_FEATURES = [
    "city_slug",
    "location_key",
    "property_type",
    "cat3_slug",
    "user_type",
] + AMENITIES

PREDICTIVE_FEATURES = PREDICTIVE_NUMERIC_FEATURES + PREDICTIVE_CATEGORICAL_FEATURES


def prepare_predictive_frame(core: pd.DataFrame) -> pd.DataFrame:
    df = prepare_model_frame(core)

    for col in ["location_latitude", "location_longitude", "location_radius"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in PREDICTIVE_CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("__missing__").astype(str)

    return df


def build_predictive_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # TargetEncoder uses internal cross-fitting in fit_transform, reducing
    # leakage when high-cardinality location categories are encoded.
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encode",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, PREDICTIVE_NUMERIC_FEATURES),
            ("categorical", categorical, PREDICTIVE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )


def permutation_group_importance(
    model: Pipeline,
    test: pd.DataFrame,
) -> pd.DataFrame:
    sampled = sample_rows(test, IMPORTANCE_SAMPLE_ROWS)
    X = sampled[PREDICTIVE_FEATURES].copy()
    y = sampled["target_log_ppsqm"].to_numpy(dtype=float)
    baseline_r2 = float(model.score(X, y))
    rng = np.random.default_rng(RANDOM_STATE)

    records: list[dict] = []
    for feature in PREDICTIVE_FEATURES:
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


def adjusted_amenity_associations(
    model: Pipeline,
    model_frame: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    sample = sample_rows(test, IMPORTANCE_SAMPLE_ROWS)
    base_X = sample[PREDICTIVE_FEATURES].copy()
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
                "raw_median_ppsqm_association_pct": (
                    float(raw_pct) if pd.notna(raw_pct) else np.nan
                ),
                "nonlinear_model_adjusted_association_pct": float(adjusted_pct),
                "interpretation": (
                    "Conditional model association; not a causal price premium."
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "nonlinear_model_adjusted_association_pct",
        ascending=False,
    )


def city_validation_table(
    test: pd.DataFrame,
    pred_log: np.ndarray,
) -> pd.DataFrame:
    result = test[["city_slug", "asking_price_per_sqm"]].copy()
    result["predicted_price_per_sqm"] = np.exp(pred_log)
    result["absolute_percentage_error"] = (
        (result["predicted_price_per_sqm"] - result["asking_price_per_sqm"]).abs()
        / result["asking_price_per_sqm"]
    )

    summary = (
        result.groupby("city_slug")
        .agg(
            validation_listings=("asking_price_per_sqm", "size"),
            median_actual_ppsqm=("asking_price_per_sqm", "median"),
            median_predicted_ppsqm=("predicted_price_per_sqm", "median"),
            median_absolute_percentage_error=("absolute_percentage_error", "median"),
        )
        .reset_index()
    )

    summary["median_prediction_bias_pct"] = 100.0 * (
        summary["median_predicted_ppsqm"] / summary["median_actual_ppsqm"] - 1.0
    )
    return summary.sort_values("validation_listings", ascending=False)


def load_ridge_metrics() -> dict:
    if not ADVANCED_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {ADVANCED_METRICS_FILE}. Run `python src/advanced_model.py` first."
        )
    return json.loads(ADVANCED_METRICS_FILE.read_text(encoding="utf-8"))


def model_comparison_table(
    baseline_metrics: dict,
    ridge_metrics: dict,
    boosted_metrics: dict,
) -> pd.DataFrame:
    rows = []
    for name, metrics in [
        ("Location median baseline", baseline_metrics),
        ("Regularized hedonic Ridge", ridge_metrics["model"]),
        ("Target-encoded gradient boosting", boosted_metrics),
    ]:
        rows.append(
            {
                "model": name,
                "r2_log_ppsqm": metrics["r2_log_ppsqm"],
                "mae_log_ppsqm": metrics["mae_log_ppsqm"],
                "rmse_log_ppsqm": metrics["rmse_log_ppsqm"],
                "median_absolute_percentage_error": metrics[
                    "median_absolute_percentage_error"
                ],
                "share_predictions_within_20_percent": metrics[
                    "share_predictions_within_20_percent"
                ],
                "share_predictions_within_30_percent": metrics[
                    "share_predictions_within_30_percent"
                ],
            }
        )
    return pd.DataFrame(rows)


def save_charts(
    comparison: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_log: np.ndarray,
    importance: pd.DataFrame,
    amenities: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(comparison["model"], comparison["r2_log_ppsqm"])
    ax.set_ylabel("Out-of-time R² on log asking price per m²")
    ax.set_title("Model Comparison on the Temporal Holdout")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

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
    ax.set_title("Nonlinear Model: Out-of-Time Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictive_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    top = importance.head(14).sort_values("r2_drop")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature_group"], top["r2_drop"])
    ax.set_xlabel("Decrease in validation R² after permutation")
    ax.set_ylabel("Feature group")
    ax.set_title("Nonlinear Model Feature-Group Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictive_feature_importance.png", dpi=180)
    plt.close(fig)

    amenity_plot = amenities.sort_values("nonlinear_model_adjusted_association_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        amenity_plot["amenity"],
        amenity_plot["nonlinear_model_adjusted_association_pct"],
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Model-adjusted association with asking price per m² (%)")
    ax.set_ylabel("Amenity")
    ax.set_title("Nonlinear Adjusted Amenity Associations (Not Causal)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictive_amenity_associations.png", dpi=180)
    plt.close(fig)


def write_predictive_model_card(metrics: dict) -> None:
    m = metrics["model"]
    b = metrics["baseline"]
    text = f"""# Predictive Model Card

## Purpose

This model is the **nonlinear predictive benchmark** for the Iran Residential Real Estate Market Analysis project. It complements the regularized Ridge hedonic model, which remains the more directly interpretable association model.

## Specification

- Target: natural log of asking price per m²
- Categorical encoding: cross-fitted `TargetEncoder`
- Regressor: `HistGradientBoostingRegressor`
- Location inputs: city, city-neighborhood key, and approximate latitude/longitude/radius
- Property inputs: size, rooms, construction year, floor structure, property category/type, advertiser type, and available amenities
- Training maximum: {MAX_TRAIN_ROWS:,} observations
- Validation maximum: {MAX_TEST_ROWS:,} observations
- Validation design: latest six observed listing months held out from training

## Out-of-time validation

- Holdout start: {metrics['temporal_holdout_start']}
- Training rows used: {metrics['training_rows_used']:,}
- Validation rows used: {metrics['validation_rows_used']:,}
- Location-baseline R²: {b['r2_log_ppsqm']:.3f}
- Nonlinear-model R²: {m['r2_log_ppsqm']:.3f}
- Median absolute percentage error: {m['median_absolute_percentage_error']:.1%}
- Predictions within 20%: {m['share_predictions_within_20_percent']:.1%}
- Predictions within 30%: {m['share_predictions_within_30_percent']:.1%}

## Intended interpretation

This is a portfolio-grade predictive benchmark, not a certified automated valuation model. Performance is measured against later advertisements, not completed transaction prices. Target encoding is cross-fitted during training to reduce leakage from high-cardinality location variables.

## Limitations

Unobserved interior quality, exact address, seller urgency, duplicated listings, strategic asking prices, market-selection effects, and missing-feature mechanisms can materially affect predictions. Model-adjusted amenity differences remain associations rather than causal premiums.
"""
    (OUTPUT_DIR / "PREDICTIVE_MODEL_CARD.md").write_text(text, encoding="utf-8")


def write_executive_summary(
    metrics: dict,
    ridge_metrics: dict,
    importance: pd.DataFrame,
    amenities: pd.DataFrame,
) -> None:
    boosted = metrics["model"]
    baseline = metrics["baseline"]
    ridge = ridge_metrics["model"]
    top_feature = importance.iloc[0]
    top_amenities = amenities.reindex(
        amenities["nonlinear_model_adjusted_association_pct"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(4)

    amenity_lines = "\n".join(
        f"- **{row.amenity}**: {row.nonlinear_model_adjusted_association_pct:+.1f}% conditional association"
        for row in top_amenities.itertuples()
    )

    text = f"""# Executive Summary — Advanced Market Modeling

## Two complementary models

The project now uses two advanced layers rather than relying on one algorithm:

1. **Regularized hedonic Ridge model** — designed for transparent, controlled association analysis.
2. **Target-encoded gradient-boosting model** — designed to capture nonlinear relationships and interactions for stronger out-of-time prediction.

Both are tested on the same temporal holdout beginning **{metrics['temporal_holdout_start']}**, so later listings are not used to fit the models.

## Out-of-time model comparison

| Model | R² (log price/m²) | Median absolute % error | Within 20% |
| --- | ---: | ---: | ---: |
| Location median baseline | {baseline['r2_log_ppsqm']:.3f} | {baseline['median_absolute_percentage_error']:.1%} | {baseline['share_predictions_within_20_percent']:.1%} |
| Hedonic Ridge | {ridge['r2_log_ppsqm']:.3f} | {ridge['median_absolute_percentage_error']:.1%} | {ridge['share_predictions_within_20_percent']:.1%} |
| Nonlinear gradient boosting | {boosted['r2_log_ppsqm']:.3f} | {boosted['median_absolute_percentage_error']:.1%} | {boosted['share_predictions_within_20_percent']:.1%} |

The models answer different questions. The Ridge layer is kept for interpretability even if a nonlinear model predicts later listings more accurately. The nonlinear model is not used to make causal claims.

## What information matters most for prediction?

The largest validation-performance loss after permuting one raw feature group occurs for **{top_feature['feature_group']}** (R² drop {top_feature['r2_drop']:.3f}). This is a predictive-importance result, not a causal ranking.

## Amenity associations

The largest nonlinear model-adjusted amenity associations by absolute magnitude are:

{amenity_lines}

These values compare model predictions with an amenity toggled while included controls are held fixed. They remain **conditional associations**, because unobserved building quality and micro-location can still confound the relationship.

## Portfolio value

This structure demonstrates a full applied-analytics workflow: nationwide data preparation, robust descriptive statistics, explicit outlier rules, location-aware modeling, temporal validation, benchmark comparison, model interpretability, nonlinear prediction, diagnostic reporting, and automated reproducibility through GitHub Actions.

## Boundaries

The source contains asking prices rather than verified sale transactions. The project therefore should not be interpreted as an appraisal engine, investment recommendation system, or causal study of architectural features.
"""
    (OUTPUT_DIR / "EXECUTIVE_SUMMARY.md").write_text(text, encoding="utf-8")


def write_notebook(metrics: dict, ridge_metrics: dict) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    boosted = metrics["model"]
    baseline = metrics["baseline"]
    ridge = ridge_metrics["model"]

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Iran Real Estate Market Analysis — Advanced Portfolio Walkthrough\n\n"
            "This notebook summarizes the reproducible advanced-analysis layer: an "
            "interpretable hedonic Ridge model plus a nonlinear predictive benchmark."
        ),
        nbf.v4.new_markdown_cell(
            f"## Temporal validation\n\n"
            f"All models are compared on later listings beginning **{metrics['temporal_holdout_start']}**. "
            "The holdout is not used for model fitting.\n\n"
            f"| Model | R² | Median abs. % error | Within 20% |\n"
            f"| --- | ---: | ---: | ---: |\n"
            f"| Location baseline | {baseline['r2_log_ppsqm']:.3f} | {baseline['median_absolute_percentage_error']:.1%} | {baseline['share_predictions_within_20_percent']:.1%} |\n"
            f"| Hedonic Ridge | {ridge['r2_log_ppsqm']:.3f} | {ridge['median_absolute_percentage_error']:.1%} | {ridge['share_predictions_within_20_percent']:.1%} |\n"
            f"| Nonlinear boosting | {boosted['r2_log_ppsqm']:.3f} | {boosted['median_absolute_percentage_error']:.1%} | {boosted['share_predictions_within_20_percent']:.1%} |"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n\n"
            "ROOT = Path('..')\n"
            "comparison = pd.read_csv(ROOT / 'outputs' / 'model_comparison.csv')\n"
            "importance = pd.read_csv(ROOT / 'outputs' / 'predictive_feature_importance.csv')\n"
            "amenities = pd.read_csv(ROOT / 'outputs' / 'predictive_amenity_associations.csv')\n"
            "display(comparison)\n"
            "display(importance.head(12))\n"
            "display(amenities)"
        ),
        nbf.v4.new_markdown_cell(
            "## Model comparison\n\n![Model comparison](../outputs/model_comparison.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Nonlinear model: actual vs predicted\n\n"
            "![Actual vs predicted](../outputs/predictive_actual_vs_predicted.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Predictive feature-group importance\n\n"
            "Permutation importance measures the validation R² lost when one raw information group is shuffled. "
            "It is not a causal ranking.\n\n"
            "![Feature importance](../outputs/predictive_feature_importance.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Adjusted amenity associations\n\n"
            "These are conditional model associations, not causal price premiums.\n\n"
            "![Amenity associations](../outputs/predictive_amenity_associations.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproduce the full pipeline\n\n"
            "```bash\n"
            "python src/download_data.py\n"
            "python src/prepare_sales_data.py\n"
            "python src/market_analysis.py\n"
            "python src/advanced_model.py\n"
            "python src/predictive_model.py\n"
            "```"
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
            f"Prepared file not found: {DATA_FILE}. Run the preparation pipeline first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATA_FILE)
    core, cutoffs = build_core_sample(raw)
    model_frame = prepare_predictive_frame(core)
    train_full, test_full, cutoff = temporal_split(model_frame)

    train = sample_rows(train_full, MAX_TRAIN_ROWS)
    test = sample_rows(test_full, MAX_TEST_ROWS)

    model = build_predictive_pipeline()
    model.fit(train[PREDICTIVE_FEATURES], train["target_log_ppsqm"])

    y_test = test["target_log_ppsqm"].to_numpy(dtype=float)
    pred_model = model.predict(test[PREDICTIVE_FEATURES])
    pred_baseline = location_baseline(train, test)

    boosted_metrics = evaluate_predictions(y_test, pred_model)
    baseline_metrics = evaluate_predictions(y_test, pred_baseline)
    ridge_metrics = load_ridge_metrics()

    metrics = {
        "model_name": "target_encoded_hist_gradient_boosting",
        "target": "log_asking_price_per_sqm",
        "temporal_holdout_start": cutoff.date().isoformat(),
        "training_rows_available": int(len(train_full)),
        "training_rows_used": int(len(train)),
        "validation_rows_available": int(len(test_full)),
        "validation_rows_used": int(len(test)),
        "baseline": baseline_metrics,
        "ridge_reference": ridge_metrics["model"],
        "model": boosted_metrics,
        "core_sample_price_tail_cutoffs": cutoffs,
        "interpretation_note": (
            "Predictive model for later listing prices, not verified transaction values. "
            "Feature importance and adjusted amenity outputs are non-causal."
        ),
    }

    comparison = model_comparison_table(baseline_metrics, ridge_metrics, boosted_metrics)
    importance = permutation_group_importance(model, test)
    amenities = adjusted_amenity_associations(model, model_frame, test)
    city_validation = city_validation_table(test, pred_model)

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "predictive_feature_importance.csv", index=False)
    amenities.to_csv(OUTPUT_DIR / "predictive_amenity_associations.csv", index=False)
    city_validation.to_csv(OUTPUT_DIR / "predictive_city_validation.csv", index=False)
    with (OUTPUT_DIR / "predictive_model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_charts(comparison, y_test, pred_model, importance, amenities)
    write_predictive_model_card(metrics)
    write_executive_summary(metrics, ridge_metrics, importance, amenities)
    write_notebook(metrics, ridge_metrics)

    print("Nonlinear predictive benchmark complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nTop predictive feature groups:")
    print(importance.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
