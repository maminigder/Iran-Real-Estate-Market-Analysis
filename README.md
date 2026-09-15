# Iran Residential Real Estate Market Analysis

![Code quality](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/code-quality.yml/badge.svg)
![Build analysis outputs](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/run-analysis.yml/badge.svg)

A nationwide, reproducible data-analysis project examining residential property listings across Iran, with **market segmentation, Tehran neighborhood analytics, approximate spatial mapping, time-series tracking, modernization-related indicators, and out-of-time predictive modeling**.

**Author:** Mohammad Amin Igder  
**Coverage in the current analytical sample:** 420 Iranian cities  
**Core analytical sample:** 516,947 residential-sale listings

## Why this project

This project connects real-estate domain knowledge with Python-based market analysis and machine learning. It is designed as a professional portfolio project for work in real estate, business analysis, market research, commercial strategy, and data-informed decision making.

The analysis addresses questions such as:

- How do residential asking prices and price per square metre differ across Iranian cities?
- Which cities dominate listing activity in the nationwide sample?
- Which Tehran neighborhoods form the highest-price market segments after minimum-sample controls?
- How do approximate neighborhood locations and amenity profiles differ within Tehran?
- How do asking-price trends change after controlling for shifts in the mix of listed properties?
- How are property characteristics and amenities associated with asking prices?
- Can a nonlinear model generalize to later listings better than a simple location benchmark?
- What can be learned from listing data without confusing advertised prices with completed transaction prices?

## Data source

The project uses the **Divar Real Estate Ads Dataset**, published by Divar on Hugging Face. The official source contains **1,000,000 anonymized real-estate advertisements** and 57 fields covering listing category, city, neighborhood, price, property size, construction year, amenities, and approximate geographic information.

Source: https://huggingface.co/datasets/divarofficial/real_estate_ads

The raw dataset is **not redistributed in this repository**. The reproducible download script retrieves it from the official source.

> **Important:** prices are treated as advertised/listing values, not verified final transaction prices. The official documentation does not clearly specify the monetary denomination of `price_value`, so this project labels monetary values as **source units** unless independently verified.

## Validated nationwide sample

The automated pipeline successfully processed the official source data and produced the following analytical sample:

| Metric | Result |
| --- | ---: |
| Residential-sale rows with usable price and building size | 533,242 |
| Analysis-ready rows before tail filtering | 522,163 |
| Core sample after 0.5% lower/upper price-per-m² tail filtering | 516,947 |
| Cities represented | 420 |
| Observed listing-month range | Feb 2021 – Mar 2025 |
| Median building size | 110 m² |
| Median asking price | 2.80B source units |
| Median asking price per m² | 27.71M source units |

### Largest city samples

| City | Listings | Median asking price / m² |
| --- | ---: | ---: |
| Tehran | 91,836 | 83.33M |
| Mashhad | 33,624 | 30.20M |
| Karaj | 26,725 | 32.93M |
| Isfahan | 18,571 | 35.00M |
| Shiraz | 16,949 | 37.06M |

These are **descriptive listing-market statistics**, not transaction-price indices.

![Top cities by listing count](outputs/top_cities_by_listing_count.png)

![Median asking price per sqm by city](outputs/median_price_per_sqm_by_city.png)

## Tehran & major-cities deep dive

The project now includes a second analytical layer focused on **local market structure** rather than only national averages. Tehran contains **91,836** listings in the core sample, with **345 neighborhood identifiers** and **104 neighborhoods** meeting the stricter 250-listing ranking threshold.

To avoid unstable small-sample rankings, neighborhood medians are combined with a transparent sample-size stabilization toward the Tehran-wide median. The map uses median approximate listing coordinates for each neighborhood; it does not imply exact addresses or official neighborhood boundaries.

### Highest stabilized Tehran neighborhood benchmarks

| Neighborhood | Listings | Stabilized asking price / m² | Raw median vs Tehran |
| --- | ---: | ---: | ---: |
| Zafaraniyeh | 795 | 190.39M | +181.2% |
| Elahiyeh | 642 | 178.04M | +170.7% |
| Velenjak | 531 | 175.91M | +179.7% |
| Niavaran | 878 | 167.76M | +136.1% |
| Farmaniyeh | 638 | 166.02M | +147.3% |
| Darrous | 706 | 161.16M | +133.1% |
| Saadat Abad | 1,590 | 155.82M | +102.3% |
| Qeytariyeh | 950 | 149.71M | +103.2% |
| Shahrak-e Gharb | 480 | 147.37M | +124.3% |
| Pasdaran | 927 | 145.29M | +96.6% |

![Tehran neighborhood asking-price map](outputs/tehran_neighborhood_price_map.png)

![Tehran neighborhood price ranking](outputs/tehran_neighborhood_price_ranking.png)

The project also compares the amenity profile of the largest Tehran neighborhood samples, including elevator, parking, storage, and rebuilt-status coverage.

![Tehran neighborhood amenity profile](outputs/tehran_neighborhood_amenity_profile.png)

### Major-city composition-adjusted time trends

The five largest markets by listing count are selected dynamically: **Tehran, Mashhad, Karaj, Isfahan, and Shiraz**. City-months require at least **150 listings**, and the first month eligible across all five markets is **May 2024** for the common-base chart.

Raw monthly medians can move simply because the composition of listed properties changes. A within-city regularized model therefore controls for neighborhood, property type/category, size, rooms, construction year, floor structure, advertiser type, and amenities. The monthly residual pattern is converted into a **composition-adjusted asking-price index**.

| City | Eligible months | Latest median asking price / m² | Raw change from first eligible month | Composition-adjusted change |
| --- | ---: | ---: | ---: | ---: |
| Tehran | 10 | 97.37M | -4.3% | +21.4% |
| Shiraz | 8 | 40.00M | +9.6% | +20.8% |
| Isfahan | 9 | 35.61M | -5.7% | +36.6% |
| Karaj | 9 | 33.97M | -2.9% | +6.0% |
| Mashhad | 9 | 31.25M | -4.6% | +17.1% |

The divergence between raw and adjusted series is itself useful: it shows why a simple monthly median can be misleading when the mix of neighborhoods, sizes, ages, and property types changes. The adjusted series is still a descriptive listing-market measure, **not an official house-price or repeat-sales index**.

![Major cities composition-adjusted price index](outputs/major_cities_composition_adjusted_price_index.png)

![Major cities monthly listing volume](outputs/major_cities_monthly_listing_volume.png)

![Major cities market positioning](outputs/major_cities_market_positioning.png)

Full city report: [`outputs/CITY_DEEP_DIVE.md`](outputs/CITY_DEEP_DIVE.md)

## Advanced modeling: interpretation + prediction

A single model is not used for every purpose. The project deliberately separates **interpretability** from **predictive performance**:

1. **Regularized hedonic Ridge model** — an interpretable multivariable model for controlled associations.
2. **Target-encoded gradient-boosting model** — a nonlinear benchmark that captures interactions and complex relationships.
3. **Location median baseline** — a deliberately simple benchmark based on city-neighborhood medians.

All three are evaluated on the **same out-of-time holdout: October 2024 through March 2025**. The validation observations are not used to fit either advanced model.

### Out-of-time model comparison

| Model | R² on log price/m² | Log RMSE | Median absolute % error | Predictions within 20% |
| --- | ---: | ---: | ---: | ---: |
| Location median baseline | 0.193 | 1.535 | **28.4%** | **38.6%** |
| Hedonic Ridge | 0.281 | 1.448 | 38.0% | 25.1% |
| Nonlinear gradient boosting | **0.334** | **1.394** | 31.2% | 32.3% |

The nonlinear model explains substantially more out-of-time variation and has the lowest log-scale RMSE. The simple location baseline, however, retains a lower median percentage error. This trade-off is reported explicitly rather than selecting a metric after seeing the results.

![Model comparison](outputs/model_comparison.png)

### What drives predictive performance?

Permutation analysis on the temporal holdout indicates that the model relies most heavily on:

- city-neighborhood location;
- building size;
- detailed property category;
- elevator availability;
- spatial longitude / latitude and city information;
- room count, construction year, and parking availability.

![Predictive feature importance](outputs/predictive_feature_importance.png)

The project also produces city-level validation diagnostics so nationwide performance cannot hide markets where predictions are systematically weaker or biased.

## Amenities and modernization-related indicators

Raw nationwide comparisons show substantial differences in median asking price per m² between properties with and without several amenities:

| Feature | Without feature | With feature |
| --- | ---: | ---: |
| Elevator | 28.13M | 39.76M |
| Parking | 22.85M | 34.00M |
| Storage / warehouse | 24.67M | 33.33M |
| Rebuilt / renovated status | 28.75M | 35.75M |

After a nonlinear model controls for the included location and property variables, the estimated associations become smaller: elevator availability is associated with approximately **+23.2%** and parking with approximately **+14.9%** in predicted asking price per m² on the counterfactual validation sample. These are **conditional model associations, not causal price premiums**.

![Adjusted amenity associations](outputs/predictive_amenity_associations.png)

![Price per sqm by construction period](outputs/price_per_sqm_by_construction_period.png)

## Methodological note: modernization vs. architectural style

The Divar dataset does **not** directly label properties as having a "modern" or "traditional" architectural style. This project therefore does not invent that classification.

Instead, construction year and available amenities are treated as **observable modernization-related indicators**. Conclusions are framed as statistical associations in listing data rather than proof that an architectural style causes a particular price premium.

Full methodology: [`docs/methodology.md`](docs/methodology.md)

## Portfolio artifacts

- [`outputs/CITY_DEEP_DIVE.md`](outputs/CITY_DEEP_DIVE.md) — Tehran neighborhood and major-city time-series report.
- [`notebooks/02_city_deep_dive.ipynb`](notebooks/02_city_deep_dive.ipynb) — portfolio walkthrough for neighborhood and trend analysis.
- [`outputs/tehran_neighborhood_summary.csv`](outputs/tehran_neighborhood_summary.csv) — detailed Tehran neighborhood metrics.
- [`outputs/major_cities_monthly_trends.csv`](outputs/major_cities_monthly_trends.csv) — raw and composition-adjusted monthly series.
- [`outputs/EXECUTIVE_SUMMARY.md`](outputs/EXECUTIVE_SUMMARY.md) — business-facing interpretation of the advanced nationwide analysis.
- [`outputs/MODEL_CARD.md`](outputs/MODEL_CARD.md) — hedonic Ridge model documentation.
- [`outputs/PREDICTIVE_MODEL_CARD.md`](outputs/PREDICTIVE_MODEL_CARD.md) — nonlinear predictive model documentation.
- [`notebooks/01_advanced_market_analysis.ipynb`](notebooks/01_advanced_market_analysis.ipynb) — advanced-model walkthrough.
- [`outputs/predictive_city_validation.csv`](outputs/predictive_city_validation.csv) — city-level out-of-time diagnostic table.
- [`outputs/model_comparison.csv`](outputs/model_comparison.csv) — machine-readable benchmark comparison.

## Data-quality note

The official Hugging Face dataset card describes a six-month 2024 collection period, while the currently published data contains `created_at_month` values covering a wider interval. In the filtered residential-sale sample, the pipeline observes dates from **February 2021 through March 2025**. The project therefore reports the dates actually present in the downloaded data rather than hard-coding the dataset-card period.

## Repository structure

```text
Iran-Real-Estate-Market-Analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── methodology.md
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_advanced_market_analysis.ipynb
│   └── 02_city_deep_dive.ipynb
├── src/
│   ├── download_data.py
│   ├── prepare_sales_data.py
│   ├── market_analysis.py
│   ├── advanced_model.py
│   ├── predictive_model.py
│   └── city_deep_dive.py
├── outputs/
│   ├── CITY_DEEP_DIVE.md
│   ├── EXECUTIVE_SUMMARY.md
│   ├── MODEL_CARD.md
│   ├── PREDICTIVE_MODEL_CARD.md
│   ├── tehran_neighborhood_summary.csv
│   ├── major_cities_monthly_trends.csv
│   ├── model_comparison.csv
│   └── charts and supporting tables...
└── .github/workflows/
    ├── code-quality.yml
    └── run-analysis.yml
```

## Reproduce the full analysis locally

```bash
python -m venv .venv
pip install -r requirements.txt
python src/download_data.py
python src/prepare_sales_data.py
python src/market_analysis.py
python src/advanced_model.py
python src/predictive_model.py
python src/city_deep_dive.py
```

The large raw and processed data files are intentionally excluded from Git. GitHub Actions rebuilds the project from the official source and commits compact validated outputs automatically.

## Reproducibility and data ethics

- Source data is anonymized by the publisher.
- Raw data is not committed to this repository.
- Cleaning rules are implemented in code rather than manually in a spreadsheet.
- Price-per-m² outlier treatment, neighborhood thresholds, and monthly sample thresholds are explicit and reproducible.
- Asking prices are never presented as completed-sale transaction prices.
- Geographic fields are treated as approximate rather than exact addresses or official polygons.
- Amenity-price relationships are described as associations, not causal effects.
- Predictive validation is temporal rather than a convenient random split.
- The composition-adjusted time series is clearly separated from an official transaction-price index.
- A simple location benchmark is retained so model complexity must justify itself empirically.

## License and attribution

The Divar source dataset is published under the **Open Database License (ODbL)**. Dataset licensing and attribution remain with the original publisher. The code in this repository is licensed separately under the MIT License; see [`LICENSE`](LICENSE).

## Next analytical extensions

- Add duplicate-listing sensitivity checks and robustness analysis.
- Test spatial holdouts to measure generalization to unseen neighborhoods.
- Investigate quantile models for lower, middle, and premium market segments.
- Add uncertainty intervals around neighborhood and time-index estimates.
- Validate the source monetary denomination before presenting converted currency values.

---

*This is an analytical portfolio project and should not be interpreted as investment, valuation, legal, or certified appraisal advice.*
