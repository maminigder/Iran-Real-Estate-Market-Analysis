# Executive Summary — Advanced Hedonic Analysis

## What the advanced model adds

The descriptive phase shows where prices differ. The hedonic model asks a harder question: **how much of the variation in asking price per m² can be explained after simultaneously accounting for location, size, rooms, construction year, floor structure, listing time, property type, and amenities?**

A regularized Ridge model is trained on pre-holdout data and tested only on the latest six observed listing months. This makes validation more realistic than a random train/test split because the model must generalize forward in time.

## Validation result

- Training rows used: **220,000**
- Out-of-time validation rows used: **80,000**
- Validation period starts: **2024-10-01**
- Naive location-baseline R²: **0.193**
- Hedonic-model R²: **0.281**
- Median absolute percentage error: **38.0%**
- Share of predictions within 20% of the listed price per m²: **25.1%**

The feature group with the largest validation-performance loss when permuted is **location_key** (R² drop: 0.185), indicating that it carries substantial predictive information in the fitted model.

## Amenity associations

The strongest modeled amenity associations by absolute magnitude are:

- **has_elevator**: +32.1% adjusted association
- **has_parking**: +25.8% adjusted association
- **has_balcony**: +6.1% adjusted association

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
