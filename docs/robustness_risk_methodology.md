# Robustness & Risk Methodology

This document describes the project layer implemented in `src/robustness_risk_analysis.py`. It is designed to test **fragility, uncertainty, and geographic generalization**, rather than to produce another headline point estimate.

## 1. Duplicate-like record sensitivity

The analytical dataset does not expose a verified dwelling identity key or sufficient ad text to prove that two rows refer to the same physical property. The project therefore avoids claiming exact deduplication.

Two transparent fingerprints are used:

### Strict same-month duplicate-like fingerprint

The fingerprint combines city, neighborhood, listing month, asking price, building size, rooms, construction year, floor, property type, rounded approximate latitude/longitude, elevator, parking, storage, and renovation status.

Rows beyond the first observation with the same strict fingerprint are treated as **duplicate-like extras for a sensitivity test only**. National and major-city median asking price/m² are recalculated after removing those extras.

### Cross-month repeat-like fingerprint

A broader fingerprint omits month and asking price but retains location and observable property characteristics. A cluster is flagged when it contains multiple rows spanning multiple listing months.

This flag is deliberately interpreted as an **upper-bound repeat-listing risk indicator** because distinct apartments in the same building or development can legitimately share the same observable features.

## 2. Neighborhood uncertainty

For Tehran neighborhoods with at least 250 core-sample listings, the project estimates a non-parametric **95% bootstrap confidence interval for the median asking price/m²** using 300 deterministic bootstrap replications per neighborhood.

The relative confidence-interval width is also reported:

```text
relative CI width = (upper 95% bound - lower 95% bound) / observed median
```

These intervals quantify sampling uncertainty conditional on the observed Divar listings. They do **not** include platform-selection bias, duplicate uncertainty, strategic seller pricing, unobserved quality, or the gap between asking and transaction prices.

## 3. Spatial holdout for unseen neighborhoods

A standard random row split can overstate geographic generalization because listings from the same neighborhood may appear in both training and test data. The risk layer therefore uses **GroupKFold by Tehran neighborhood**.

In every fold, entire neighborhoods are absent from model training and only appear at evaluation time.

Three benchmarks are compared on the same held-out rows:

1. **Tehran median baseline** — one constant training-sample median.
2. **Property-only gradient boosting** — uses building size, rooms, construction year, floor structure, amenities, property category/type, and advertiser type.
3. **Spatial gradient boosting** — adds approximate latitude, longitude, and privacy radius.

The model intentionally does **not** use the neighborhood name in this test. This makes the question explicit: can observable property and approximate spatial information generalize to a neighborhood label that the model has never seen?

Evaluation metrics include R² and MAE/RMSE on log asking price/m², median absolute percentage error on the original scale, and shares of predictions within 20% and 30%.

This validation is more demanding than the main temporal holdout. It tests a different failure mode: **geographic extrapolation** rather than forecasting later listings in already represented markets.

## 4. Affordable / Mid-market / Premium segmentation

Only Tehran neighborhoods with at least 250 listings are eligible. Segmentation is based on the previously defined **sample-size-stabilized neighborhood asking-price benchmark**, not on a single listing.

Eligible neighborhoods are split into three equal-count tiers by the 33.3rd and 66.7th percentiles of the stabilized benchmark:

- **Affordable** — lower third;
- **Mid-market** — middle third;
- **Premium** — upper third.

The resulting tier profile reports neighborhood count, listing count/share, median asking price, median asking price/m², median building size, rooms, construction year, and amenity prevalence.

These names are **relative analytical market tiers within the eligible Tehran sample and period**. They are not official classifications, valuation grades, affordability standards, or investment recommendations.

## 5. Reproducibility and reporting

The script writes machine-readable CSV/JSON outputs, portfolio charts, a Markdown risk report, and `notebooks/03_robustness_risk_analysis.ipynb`. GitHub Actions rebuilds the risk layer after the city deep dive so the neighborhood stabilization output is available before confidence intervals and market tiers are calculated.

## 6. Interpretation guardrails

- Duplicate fingerprints identify similarity, not verified property identity.
- Bootstrap confidence intervals describe the observed listing sample, not the full housing stock.
- Approximate coordinates are privacy-preserving spatial signals, not exact addresses.
- Spatial holdout scores apply to Tehran neighborhoods represented by the eligibility rules and need not transfer to other cities.
- Market-tier thresholds are period- and sample-specific.
- Asking prices are not completed transaction prices.
- Monetary values remain in source units unless the source denomination is independently verified.
