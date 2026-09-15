# Predictive Model Card

## Purpose

This model is the **nonlinear predictive benchmark** for the Iran Residential Real Estate Market Analysis project. It complements the regularized Ridge hedonic model, which remains the more directly interpretable association model.

## Specification

- Target: natural log of asking price per m²
- Categorical encoding: cross-fitted `TargetEncoder`
- Regressor: `HistGradientBoostingRegressor`
- Location inputs: city, city-neighborhood key, and approximate latitude/longitude/radius
- Property inputs: size, rooms, construction year, floor structure, property category/type, advertiser type, and available amenities
- Training maximum: 220,000 observations
- Validation maximum: 80,000 observations
- Validation design: latest six observed listing months held out from training

## Out-of-time validation

- Holdout start: 2024-10-01
- Training rows used: 220,000
- Validation rows used: 80,000
- Location-baseline R²: 0.193
- Nonlinear-model R²: 0.334
- Median absolute percentage error: 31.2%
- Predictions within 20%: 32.3%
- Predictions within 30%: 48.1%

## Intended interpretation

This is a portfolio-grade predictive benchmark, not a certified automated valuation model. Performance is measured against later advertisements, not completed transaction prices. Target encoding is cross-fitted during training to reduce leakage from high-cardinality location variables.

## Limitations

Unobserved interior quality, exact address, seller urgency, duplicated listings, strategic asking prices, market-selection effects, and missing-feature mechanisms can materially affect predictions. Model-adjusted amenity differences remain associations rather than causal premiums.
