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

The preparation pipeline also creates transparent plausibility flags for building size and construction year. Observations are preserved in the processed file; the main analysis uses only rows marked `analysis_ready`.

## 3. Price per square metre

For records with a positive asking price and positive building size:

```text
asking price per m² = price_value / building_size
```

Because real-estate prices are strongly right-skewed, city comparisons emphasize the **median** rather than the arithmetic mean.

For the nationwide analytical sample, the lowest 0.5% and highest 0.5% of asking-price-per-m² observations are excluded from the core comparison sample. This reduces the influence of obvious recording errors and extreme listings without modifying the stored processed dataset.

## 4. City comparisons

City rankings apply a minimum sample threshold. The default threshold in the code is 100 usable listings per city. This prevents tiny samples from appearing as precise market rankings.

Both listing count and median asking price per m² are reported because market coverage differs significantly across cities.

## 5. Modernization indicators

The source data does not contain a validated architectural-style label such as "modern" or "traditional." Therefore, the project does **not** infer architectural style directly.

Instead, the analysis uses observable variables that can reasonably be described as modernization-related indicators, including:

- construction year
- elevator availability
- parking availability
- storage/warehouse availability
- renovation status
- balcony availability
- heating and cooling systems, where coverage is sufficient

These variables are interpreted as associations with asking price, not as proof that a particular architectural style causes a price premium.

## 6. Construction year

Construction years are stored in Persian/Jalali format in the source data. Persian and Arabic numerals are normalized to Western digits while preserving the Jalali year itself. The project avoids unnecessary calendar conversion when the analysis only requires relative age groups.

The descriptive grouping is:

- up to 1379
- 1380–1389
- 1390–1399
- 1400 and newer

## 7. Advanced hedonic model

The advanced stage uses a **regularized hedonic Ridge regression**. The target is the natural logarithm of asking price per square metre. A log target reduces the influence of extreme values and allows percentage-style interpretation of modeled differences.

The model includes the following feature groups:

- log building size
- room count
- construction year
- floor number
- total floors
- units per floor
- listing month trend
- a city + neighborhood composite location control
- property type
- detailed residential category
- advertiser/user type
- elevator, parking, storage, renovation, balcony, heating, and cooling indicators

Rare categorical levels are grouped by the encoder using a minimum training-frequency threshold. Continuous features are median-imputed and standardized. Amenity missingness is represented explicitly as an `unknown` category rather than silently treating missing values as absence.

The model is regularized with Ridge regression (`alpha = 5.0`) so that highly correlated real-estate characteristics and large sets of location indicators do not produce unstable unregularized coefficients.

## 8. Temporal validation

The advanced model uses an **out-of-time holdout**, not a random train/test split. The latest six observed listing months are reserved for validation and are never used for fitting.

This design is more demanding and more realistic for a market-analysis use case because the model must generalize to later listings rather than merely reconstruct randomly held-out records from the same time distribution.

A naive benchmark is also calculated using the median log asking price per m² for each training-sample city-neighborhood location, with city and nationwide fallbacks for previously unseen locations. The advanced model is compared against this location baseline using out-of-time R² and percentage-error diagnostics.

## 9. Feature importance

Feature-group importance is measured by **permutation on the temporal holdout**. One raw feature group is shuffled at a time while all other columns remain unchanged. The decrease in validation R² is recorded.

This answers a practical question: how much predictive information does the fitted model lose when a given feature group is disrupted? It should not be interpreted as a causal ranking.

## 10. Adjusted amenity associations

For each amenity, the fitted model is used to create two counterfactual prediction sets on the same validation properties: one with the amenity set to `false` and one set to `true`, while every other modeled characteristic remains fixed.

The resulting percentage difference is reported as a **model-adjusted association**. This is intentionally not labeled a causal premium because unobserved building quality, micro-location, developer quality, interior condition, and seller strategy may remain confounded with the amenity.

## 11. Monetary units

The official dataset documentation identifies `price_value` as the property price field but does not clearly document its currency denomination. Individual advertisements often contain Persian price text consistent with Iranian market conventions, but this project does not rely on text inference to declare a universal unit.

Accordingly, code and charts refer to **source units** unless the denomination is independently validated.

## 12. Time coverage

The project reports the observed minimum and maximum dates from the downloaded analytical sample rather than hard-coding a date period. This avoids relying on potentially stale documentation when the currently published source file contains a wider observed range.

## 13. Limitations

Key limitations include:

- asking prices may differ from final transaction prices;
- Divar listings are not a complete census of all Iranian real-estate transactions;
- duplicate, stale, or strategically priced advertisements may exist;
- feature missingness is not random;
- city and neighborhood representation can be uneven;
- approximate geographic fields should not be interpreted as exact addresses;
- architectural style and interior design quality are not directly observed;
- correlations and model-adjusted associations do not establish causation;
- future market regimes may differ from the historical sample.

## 14. Reproducibility

The full workflow is code-based:

1. `src/download_data.py` downloads the official source file.
2. `src/prepare_sales_data.py` filters and standardizes the residential-sale sample.
3. `src/market_analysis.py` creates nationwide descriptive summaries and visualizations.
4. `src/advanced_model.py` fits the temporal hedonic model, evaluates it against a location baseline, measures feature-group importance, generates adjusted amenity associations, creates a model card and executive summary, and builds the portfolio notebook.

The raw dataset and generated local analytical files are excluded from Git. GitHub Actions rebuilds the analysis from the official source and commits only the compact validated outputs needed for the public portfolio.
