# Executive Summary — Advanced Market Modeling

## Two complementary models

The project now uses two advanced layers rather than relying on one algorithm:

1. **Regularized hedonic Ridge model** — designed for transparent, controlled association analysis.
2. **Target-encoded gradient-boosting model** — designed to capture nonlinear relationships and interactions for stronger out-of-time prediction.

Both are tested on the same temporal holdout beginning **2024-10-01**, so later listings are not used to fit the models.

## Out-of-time model comparison

| Model | R² (log price/m²) | Median absolute % error | Within 20% |
| --- | ---: | ---: | ---: |
| Location median baseline | 0.193 | 28.4% | 38.6% |
| Hedonic Ridge | 0.281 | 38.0% | 25.1% |
| Nonlinear gradient boosting | 0.334 | 31.2% | 32.3% |

The models answer different questions. The Ridge layer is kept for interpretability even if a nonlinear model predicts later listings more accurately. The nonlinear model is not used to make causal claims.

## What information matters most for prediction?

The largest validation-performance loss after permuting one raw feature group occurs for **location_key** (R² drop 0.195). This is a predictive-importance result, not a causal ranking.

## Amenity associations

The largest nonlinear model-adjusted amenity associations by absolute magnitude are:

- **has_elevator**: +23.2% conditional association
- **has_parking**: +14.9% conditional association
- **has_balcony**: +1.4% conditional association
- **is_rebuilt**: +0.7% conditional association

These values compare model predictions with an amenity toggled while included controls are held fixed. They remain **conditional associations**, because unobserved building quality and micro-location can still confound the relationship.

## Portfolio value

This structure demonstrates a full applied-analytics workflow: nationwide data preparation, robust descriptive statistics, explicit outlier rules, location-aware modeling, temporal validation, benchmark comparison, model interpretability, nonlinear prediction, diagnostic reporting, and automated reproducibility through GitHub Actions.

## Boundaries

The source contains asking prices rather than verified sale transactions. The project therefore should not be interpreted as an appraisal engine, investment recommendation system, or causal study of architectural features.
