# Methodology

## 1. Research framing

This project is a descriptive and exploratory analysis of residential real-estate listings across Iran. Its purpose is to identify market patterns that can be reproduced from the published Divar dataset and presented clearly in a professional portfolio.

The core unit of observation is a **property advertisement**, not a completed real-estate transaction.

## 2. Core sample

The initial analytical sample is restricted to records where:

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

For the first nationwide descriptive analysis, the lowest 0.5% and highest 0.5% of asking-price-per-m² observations are excluded from the core comparison sample. This is intended to reduce the effect of obvious recording errors and extreme listings without modifying the stored processed dataset.

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
- heating and cooling systems, where coverage is sufficient

These variables are interpreted as associations with asking price, not as proof that a particular architectural style causes a price premium.

## 6. Construction year

Construction years are stored in Persian/Jalali format in the source data. Persian and Arabic numerals are normalized to Western digits while preserving the Jalali year itself. The project avoids unnecessary calendar conversion when the analysis only requires relative age groups.

The first descriptive grouping is:

- up to 1379
- 1380–1389
- 1390–1399
- 1400 and newer

## 7. Monetary units

The official dataset documentation identifies `price_value` as the property price field but does not clearly document its currency denomination. Individual advertisements often contain Persian price text consistent with Iranian market conventions, but this project does not rely on text inference to declare a universal unit.

Accordingly, code and charts refer to **source units** unless the denomination is independently validated.

## 8. Time coverage

The current Hugging Face dataset card describes a six-month period in 2024, while the dataset viewer currently displays `created_at_month` values over a wider range. The project treats this as a documentation inconsistency and reports the observed minimum and maximum dates from the downloaded analytical sample rather than hard-coding a date period.

## 9. Limitations

Key limitations include:

- asking prices may differ from final transaction prices;
- Divar listings are not a complete census of all Iranian real-estate transactions;
- duplicate, stale, or strategically priced advertisements may exist;
- feature missingness is not random;
- city and neighborhood representation can be uneven;
- approximate geographic fields should not be interpreted as exact addresses;
- correlations between amenities and price do not establish causation.

## 10. Reproducibility

The full workflow is code-based:

1. `src/download_data.py` downloads the official source file.
2. `src/prepare_sales_data.py` filters and standardizes the residential-sale sample.
3. `src/market_analysis.py` creates summaries and visualizations.

The raw dataset and generated local analytical files are excluded from Git so the repository remains lightweight and respects the source distribution model.
