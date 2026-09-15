# Iran Residential Real Estate Market Analysis

![Code quality](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/code-quality.yml/badge.svg)
![Build analysis outputs](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/run-analysis.yml/badge.svg)

A nationwide, reproducible data-analysis project examining residential property listings across Iran, with a focus on **asking-price patterns, regional differences, property characteristics, and observable indicators of housing modernization**.

**Author:** Mohammad Amin Igder  
**Coverage in the current analytical sample:** 420 Iranian cities  
**Core analytical sample:** 516,947 residential-sale listings

## Why this project

This project connects real-estate domain knowledge with Python-based market analysis. It is designed as a professional portfolio project for work in real estate, business analysis, market research, commercial strategy, and data-informed decision making.

The analysis addresses questions such as:

- How do residential asking prices and price per square metre differ across Iranian cities?
- Which cities dominate listing activity in the nationwide sample?
- How are property characteristics and amenities associated with asking prices?
- How do construction period, elevators, parking, storage, and renovation status relate to market value?
- What can be learned from listing data without confusing advertised prices with completed transaction prices?

## Data source

The project uses the **Divar Real Estate Ads Dataset**, published by Divar on Hugging Face. The official source contains **1,000,000 anonymized real-estate advertisements** and 57 fields covering listing category, city, neighborhood, price, property size, construction year, amenities, and approximate geographic information.

Source: https://huggingface.co/datasets/divarofficial/real_estate_ads

The raw dataset is **not redistributed in this repository**. The reproducible download script retrieves it from the official source.

> **Important:** prices are treated as advertised/listing values, not verified final transaction prices. The official documentation does not clearly specify the monetary denomination of `price_value`, so this project labels monetary values as **source units** unless independently verified.

## First validated nationwide run

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

The five largest city samples in the core dataset are:

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

## Amenities and modernization-related indicators

The nationwide descriptive sample shows clear associations between several amenities and median asking price per m²:

| Feature | Without feature | With feature |
| --- | ---: | ---: |
| Elevator | 28.13M | 39.76M |
| Parking | 22.85M | 34.00M |
| Storage / warehouse | 24.67M | 33.33M |
| Rebuilt / renovated status | 28.75M | 35.75M |

These comparisons **do not establish causation**. Properties with these features may also differ systematically by city, neighborhood, age, size, or quality. A later multivariable model can estimate adjusted associations.

![Price per sqm by construction period](outputs/price_per_sqm_by_construction_period.png)

## Methodological note: modernization vs. architectural style

The Divar dataset does **not** directly label properties as having a "modern" or "traditional" architectural style. This project therefore does not invent that classification.

Instead, construction year and available amenities are treated as **observable modernization-related indicators**. Conclusions are framed as statistical associations in listing data rather than proof that an architectural style causes a particular price premium.

Full methodology: [`docs/methodology.md`](docs/methodology.md)

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
├── src/
│   ├── download_data.py
│   ├── prepare_sales_data.py
│   └── market_analysis.py
├── outputs/
│   ├── market_summary.json
│   ├── city_summary.csv
│   ├── amenity_summary.csv
│   ├── construction_year_summary.csv
│   └── charts...
└── .github/workflows/
    ├── code-quality.yml
    └── run-analysis.yml
```

## Reproduce the analysis locally

```bash
python -m venv .venv
pip install -r requirements.txt
python src/download_data.py
python src/prepare_sales_data.py
python src/market_analysis.py
```

The large raw and processed data files are intentionally excluded from Git. Generated validated summary outputs and charts are produced automatically by GitHub Actions.

## Reproducibility and data ethics

- Source data is anonymized by the publisher.
- Raw data is not committed to this repository.
- Cleaning rules are implemented in code rather than manually in a spreadsheet.
- Price-per-m² outlier treatment and city sample thresholds are explicit and reproducible.
- Asking prices are never presented as completed-sale transaction prices.
- Geographic fields are treated as approximate rather than exact addresses.
- Amenity-price relationships are described as associations, not causal effects.

## License and attribution

The Divar source dataset is published under the **Open Database License (ODbL)**. Dataset licensing and attribution remain with the original publisher. The code in this repository is licensed separately under the MIT License; see [`LICENSE`](LICENSE).

## Next analytical extensions

- Add a multivariable model controlling for city, size, rooms, construction year, and amenities.
- Compare Tehran and other large markets at neighborhood level where sample coverage permits.
- Build time-based city trends only after checking monthly sample stability.
- Add an executive-summary notebook for portfolio presentation.
- Develop a careful connection between modernization indicators and the broader real-estate research question.

---

*This is an analytical portfolio project and should not be interpreted as investment, valuation, or legal advice.*
