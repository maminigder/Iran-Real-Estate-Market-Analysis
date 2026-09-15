# Hedonic Model Card

## Purpose

This model estimates **asking price per square metre** for Iranian residential-sale listings using observable property, location, time, and amenity characteristics. It is built for market analysis and portfolio demonstration, not automated valuation or investment decisions.

## Model

- Estimator: Ridge regression (regularized linear hedonic model)
- Target: natural log of asking price per m²
- Ridge alpha: 5.0
- Location control: city + neighborhood composite category
- Rare-category threshold: 100 training observations
- Training maximum: 220,000 rows
- Validation maximum: 80,000 rows
- Validation design: latest six observed listing months held out from model fitting

## Validation

- Temporal cutoff: 2024-10-01
- Training rows used: 220,000
- Validation rows used: 80,000
- Location-baseline R²: 0.193
- Hedonic-model R²: 0.281
- Median absolute percentage error: 38.0%
- Predictions within 20%: 25.1%

## Interpretation

The model captures conditional associations after controlling for included variables. The adjusted amenity estimates are **not causal premiums**. Unobserved quality, micro-location, interior condition, seller strategy, duplicated listings, and selection into Divar can still influence estimated relationships.

## Intended use

Appropriate uses include exploratory market analysis, portfolio demonstration, reproducible research, and hypothesis generation. It should not be used as a certified appraisal, transaction-price estimator, lending model, or investment recommendation engine.
