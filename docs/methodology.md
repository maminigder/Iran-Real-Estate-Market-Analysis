# Methodology

## 1. Research framing

This project is a descriptive, exploratory, and predictive analysis of residential real-estate listings across Iran. Its purpose is to identify market patterns that can be reproduced from the published Divar dataset and presented clearly in a professional portfolio.

The core unit of observation is a **property advertisement**, not a completed real-estate transaction.

## 2. Core sample

The analytical sample is restricted to records where:

- `cat2_slug == residential-sell`
- `price_value > 0`
- `building_size > 0`
- a city identifier is available

The preparation pipeline creates transparent plausibility flags for building size and construction year. Observations are preserved in the processed file; the main analysis uses only rows marked `analysis_ready`.

## 3. Price per square metre

For records with a positive asking price and positive building size:

```text
asking price per m² = price_value / building_size
```

Because real-estate prices are strongly right-skewed, city comparisons emphasize the **median** rather than the arithmetic mean. For the nationwide analytical sample, the lowest 0.5% and highest 0.5% of asking-price-per-m² observations are excluded from the core comparison sample. The original processed observations remain unchanged.

## 4. City comparisons

City rankings apply a minimum sample threshold of 100 usable listings by default. Both listing count and median asking price per m² are reported because market coverage differs substantially across cities.

## 5. Modernization indicators

The source data does not contain a validated architectural-style label such as "modern" or "traditional." The project therefore does **not** infer architectural style directly.

Observable modernization-related indicators include construction year, elevator, parking, storage/warehouse, renovation status, balcony, and heating/cooling systems where sufficient information is present. They are interpreted as associations with asking price rather than proof of architectural causation.

## 6. Construction year

Construction years are stored in Persian/Jalali format. Persian and Arabic numerals are normalized to Western digits while preserving the Jalali year. Descriptive groups are up to 1379, 1380–1389, 1390–1399, and 1400+.

## 7. Interpretable hedonic model

The first advanced layer is a **regularized Ridge hedonic regression** with natural log asking price per m² as the target. Controls include log building size, rooms, construction year, floor structure, listing-month trend, city-neighborhood location, property type/category, advertiser type, and available amenities.

Continuous features are median-imputed and standardized. Categorical features are one-hot encoded with rare categories grouped. Amenity missingness is retained as an explicit unknown category. Ridge regularization (`alpha = 5.0`) stabilizes estimates in the presence of many correlated characteristics and location indicators.

This model is retained primarily as the more interpretable controlled-association layer.

## 8. Nonlinear predictive benchmark

A second model is added to test whether nonlinear relationships and feature interactions improve prediction of later listings. It combines:

- cross-fitted `TargetEncoder` for high-cardinality categorical variables;
- `HistGradientBoostingRegressor` for nonlinear prediction;
- city and city-neighborhood location controls;
- approximate latitude, longitude, and privacy radius as additional spatial signals;
- the same property, time, advertiser, and amenity information used in the broader analytical pipeline.

Target encoding is cross-fitted during training to reduce target leakage. The nonlinear model is complementary to, rather than a replacement for, the Ridge model: the Ridge layer supports interpretation while gradient boosting provides a stronger predictive benchmark where the data supports it.

## 9. Temporal validation and baseline

Both advanced models use an **out-of-time holdout** rather than a random split. The latest six observed listing months are reserved for validation and are never used for fitting.

A naive benchmark predicts the median log asking price per m² for each training-sample city-neighborhood combination, with city and nationwide fallbacks for unseen locations. All models are compared on the same holdout using:

- R² on log asking price per m²;
- log-scale MAE and RMSE;
- median absolute percentage error on the original price-per-m² scale;
- share of predictions within 20% and 30% of the listed value.

This prevents a sophisticated model from appearing valuable merely because location alone is highly predictive.

## 10. Feature-group importance

Feature importance is measured by permutation on the temporal holdout. One raw feature group is shuffled while all other fields remain unchanged, and the loss in validation R² is recorded. This quantifies predictive information, not causal importance.

## 11. Adjusted amenity associations

For each amenity, the fitted models can generate paired counterfactual predictions for the same validation properties with the amenity set to `false` and `true`, while included controls remain fixed.

The percentage difference is reported as a **model-adjusted association**, not a causal premium. Unobserved building quality, exact micro-location, developer quality, interior condition, seller strategy, and selection effects can still confound the relationship.

## 12. City-level predictive diagnostics

The nonlinear model also reports validation performance by city, including sample size, median actual and predicted asking price per m², median absolute percentage error, and median prediction bias. This helps reveal geographic variation that can be hidden by a single nationwide score.

## 13. Monetary units

The official dataset documentation identifies `price_value` as the property price field but does not clearly document its currency denomination. The project therefore labels monetary values as **source units** unless independently validated.

## 14. Time coverage

The project reports minimum and maximum dates actually observed in the downloaded analytical sample instead of hard-coding a period from potentially stale documentation.

## 15. Limitations

Key limitations include:

- asking prices may differ from final transaction prices;
- Divar listings are not a complete census of Iranian real-estate transactions;
- duplicate, stale, or strategically priced advertisements may exist;
- feature missingness is not random;
- city and neighborhood representation can be uneven;
- approximate geographic fields are not exact addresses;
- architectural style and interior design quality are not directly observed;
- correlations and model-adjusted associations do not establish causation;
- future market regimes may differ from the historical sample.

## 16. Reproducibility

The workflow is fully code-based:

1. `src/download_data.py` downloads the official source file.
2. `src/prepare_sales_data.py` filters and standardizes residential-sale listings.
3. `src/market_analysis.py` creates nationwide descriptive outputs.
4. `src/advanced_model.py` fits the interpretable temporal hedonic Ridge model.
5. `src/predictive_model.py` fits the nonlinear benchmark, compares models, produces city diagnostics and predictive importance, and refreshes the executive summary and portfolio notebook.

The raw and large processed datasets are excluded from Git. GitHub Actions rebuilds the analysis from the official source and commits only compact validated portfolio outputs.
