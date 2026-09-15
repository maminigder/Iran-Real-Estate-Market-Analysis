# Iran Residential Real Estate Market Analysis

A nationwide data-analysis portfolio project examining residential property listings across Iran, with a focus on **pricing patterns, regional differences, property characteristics, and indicators of housing modernization**.

**Author:** Mohammad Amin Igder  
**Project status:** In development

## Project objective

This project is designed to answer practical real-estate questions such as:

- How do residential asking prices and price per square metre differ across Iranian cities?
- Which property characteristics are associated with higher asking prices?
- How do building age and modernization-related amenities such as elevators, parking, storage, and renovation status relate to market value?
- How concentrated is listing activity across cities and neighborhoods?
- What patterns can be identified without confusing listing prices with completed transaction prices?

The project connects real-estate domain knowledge with reproducible data analysis in Python and is intended as a professional portfolio project for roles in real estate, business analysis, market research, and commercial strategy.

## Data source

The analysis uses the **Divar Real Estate Ads Dataset**, published by Divar on Hugging Face. The source contains **1,000,000 anonymized real-estate advertisements** and 57 fields covering listing category, city, neighborhood, asking price, property size, construction year, amenities, and approximate geographic information.

Source: https://huggingface.co/datasets/divarofficial/real_estate_ads

The raw dataset is **not stored in this repository**. The download script retrieves it from the official source.

> **Important:** `price_value` represents an advertised/listing value where available. It should not be interpreted as a verified final transaction price.

## Analytical scope

The first version focuses on **residential properties offered for sale** (`residential-sell`). Rental and temporary-rental listings are excluded from the core sale-price analysis so that financially different listing types are not mixed together.

Main analytical dimensions:

1. **National market overview** — listing volume, price distributions, property sizes, and coverage.
2. **City comparison** — asking price and price per m² across cities with sufficient observations.
3. **Property characteristics** — size, number of rooms, building age, floor, and property type.
4. **Modernization indicators** — elevator, parking, warehouse/storage, renovation status, and selected building systems where data quality permits.
5. **Time patterns** — listing trends using the date field after validating coverage in the filtered residential-sale sample.
6. **Data-quality review** — missingness, implausible values, outliers, and limitations.

## Methodological note: modernization vs. architectural style

The Divar dataset does **not** directly label a property as having a "modern" or "traditional" architectural style. Therefore, this project does not make that unsupported classification.

Instead, construction year and available amenities are treated as **observable modernization-related indicators**. Any conclusions are framed as statistical associations in listing data rather than proof of architectural causation.

## Repository structure

```text
Iran-Real-Estate-Market-Analysis/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── download_data.py
│   ├── prepare_sales_data.py
│   └── market_analysis.py
└── outputs/
    └── README.md
```

As the project develops, notebooks and final charts will be added after the data pipeline has been run and the results have been validated.

## Quick start

### 1. Create a Python environment

```bash
python -m venv .venv
```

Activate it, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download the official dataset

```bash
python src/download_data.py
```

### 3. Prepare the residential-sale analytical dataset

```bash
python src/prepare_sales_data.py
```

### 4. Run the first market analysis

```bash
python src/market_analysis.py
```

Generated local data files are ignored by Git to avoid republishing the source dataset or committing large files.

## Reproducibility and data ethics

- The source dataset is anonymized by its publisher.
- Raw data is not committed to this repository.
- Cleaning rules are implemented in code rather than applied manually.
- Outlier treatment and sample-size thresholds are documented in the analysis output.
- Asking prices are not presented as completed-sale transaction prices.
- Geographic fields should be treated as approximate according to the source dataset's privacy design.

## License and attribution

The source Divar dataset is published under the **Open Database License (ODbL)**. Dataset licensing and attribution remain with the original publisher. The code in this repository is licensed separately under the MIT License; see `LICENSE`.

## Planned next stages

- Validate the cleaned nationwide residential-sale sample.
- Produce city-level price-per-m² rankings with minimum-sample safeguards.
- Build property-age and amenities analyses.
- Add clear publication-quality charts.
- Add a reproducible Jupyter notebook for portfolio presentation.
- Develop a final executive summary with real-estate and business insights.

---

*This is an analytical portfolio project and should not be interpreted as investment, valuation, or legal advice.*
