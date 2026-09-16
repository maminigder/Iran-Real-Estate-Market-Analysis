# Iran Residential Real Estate Market Analysis

![Code quality](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/code-quality.yml/badge.svg)
![Build analysis outputs](https://github.com/maminigder/Iran-Real-Estate-Market-Analysis/actions/workflows/run-analysis.yml/badge.svg)

A nationwide, reproducible real-estate market intelligence project combining **market analysis, neighborhood analytics, spatial modeling, time-series adjustment, machine learning, uncertainty analysis, and robustness testing**.

**Author:** Mohammad Amin Igder  
**Project scope:** Market Research • Market Intelligence • Applied Analytics • Decision Support

> **Recruiter / hiring manager:** start with the [Executive Portfolio — 30-second project overview](EXECUTIVE_PORTFOLIO.md).

## Project at a glance

| Key metric | Result |
| --- | ---: |
| **Residential-sale listings analyzed** | **516,947** |
| **Iranian cities covered** | **420** |
| **Tehran listings analyzed** | **91,836** |
| **Best out-of-time nationwide model R²** | **0.334** |
| **Median error on completely unseen Tehran neighborhoods** | **20.9%** |

The project starts from **1,000,000 anonymized Divar real-estate advertisements** and turns them into a reproducible analytical workflow with data-quality controls, city and neighborhood benchmarks, predictive models, temporal validation, spatial holdouts, bootstrap uncertainty, duplicate-risk sensitivity, and market segmentation.

### Three visuals that summarize the project

**1. Tehran neighborhood market structure**

![Tehran neighborhood asking-price map](outputs/tehran_neighborhood_price_map.png)

**2. Nationwide out-of-time model comparison**

![Model comparison](outputs/model_comparison.png)

**3. Tehran Affordable / Mid-market / Premium segmentation**

![Tehran market segments](outputs/tehran_market_segments.png)

## What this project demonstrates

- **Real-estate market intelligence:** nationwide city comparisons, Tehran neighborhood benchmarking, amenity profiles, and market segmentation.
- **Applied machine learning:** interpretable hedonic Ridge regression plus nonlinear gradient boosting.
- **Validation discipline:** later-period temporal holdout and grouped spatial holdout on neighborhoods never seen during training.
- **Risk-aware analytics:** bootstrap confidence intervals, duplicate-like listing sensitivity, explicit sample thresholds, and benchmark comparisons.
- **Reproducibility:** code-based data preparation, documented methodology, notebooks, and automated GitHub Actions workflows.

## Business problem

Residential real-estate markets are highly local. National averages can hide major differences between cities and neighborhoods, and raw monthly asking-price changes can be distorted when the mix of listed properties changes.

This project asks practical questions:

- How do residential asking prices and price per square metre differ across Iranian cities?
- Which Tehran neighborhoods occupy higher- and lower-price market segments?
- Which property and location characteristics carry the most predictive information?
- Can a model generalize to later listings and to neighborhoods it has never seen before?
- How robust are the conclusions to duplicate-like listings, sampling uncertainty, and changing property mix?

## Data source and validated sample

The project uses the **Divar Real Estate Ads Dataset**, published by Divar on Hugging Face. The source contains **1,000,000 anonymized real-estate advertisements** and 57 fields covering listing category, city, neighborhood, price, size, construction year, amenities, and approximate geographic information.

Source: https://huggingface.co/datasets/divarofficial/real_estate_ads

The raw dataset is **not redistributed in this repository**. The reproducible pipeline downloads it from the official source.

> **Important:** prices are advertised/listing values, not verified completed transaction prices. The source documentation does not clearly verify the monetary denomination of `price_value`, so monetary results are labelled **source units**.

| Sample metric | Result |
| --- | ---: |
| Residential-sale rows with usable price and size | 533,242 |
| Analysis-ready rows before tail filtering | 522,163 |
| Core sample after 0.5% lower/upper price-per-m² tail filtering | 516,947 |
| Cities represented | 420 |
| Observed listing-month range | Feb 2021 – Mar 2025 |
| Median building size | 110 m² |
| Median asking price / m² | 27.71M source units |

### Largest city samples

| City | Listings | Median asking price / m² |
| --- | ---: | ---: |
| Tehran | 91,836 | 83.33M |
| Mashhad | 33,624 | 30.20M |
| Karaj | 26,725 | 32.93M |
| Isfahan | 18,571 | 35.00M |
| Shiraz | 16,949 | 37.06M |

![Top cities by listing count](outputs/top_cities_by_listing_count.png)

## Tehran & major-cities deep dive

Tehran contains **91,836** listings and **345 neighborhood identifiers** in the core sample. Neighborhood price ranking requires at least **250 listings** and uses a sample-size-stabilized benchmark so smaller samples do not dominate the ranking through noise.

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

![Tehran neighborhood price ranking](outputs/tehran_neighborhood_price_ranking.png)

Full city analysis: [`outputs/CITY_DEEP_DIVE.md`](outputs/CITY_DEEP_DIVE.md)

## Major-city time trends

The five largest markets by listing count are **Tehran, Mashhad, Karaj, Isfahan, and Shiraz**. A regularized within-city model controls for observed changes in neighborhood, property type/category, size, rooms, construction year, floor structure, advertiser type, and amenities. The monthly residual pattern is then converted into a **composition-adjusted asking-price index**.

| City | Eligible months | Latest median asking price / m² | Raw change | Composition-adjusted change |
| --- | ---: | ---: | ---: | ---: |
| Tehran | 10 | 97.37M | -4.3% | +21.4% |
| Shiraz | 8 | 40.00M | +9.6% | +20.8% |
| Isfahan | 9 | 35.61M | -5.7% | +36.6% |
| Karaj | 9 | 33.97M | -2.9% | +6.0% |
| Mashhad | 9 | 31.25M | -4.6% | +17.1% |

The adjusted series is a descriptive listing-market measure, **not an official house-price or repeat-sales index**.

![Major cities composition-adjusted price index](outputs/major_cities_composition_adjusted_price_index.png)

## Advanced modeling: interpretation + prediction

The project deliberately separates **interpretability** from **predictive performance**:

1. **Regularized hedonic Ridge model** — controlled, interpretable multivariable associations.
2. **Target-encoded gradient boosting** — nonlinear predictive benchmark.
3. **Location median baseline** — simple benchmark used to test whether complexity adds value.

All three are evaluated on the same **out-of-time holdout: October 2024 through March 2025**.

| Model | R² on log price/m² | Log RMSE | Median absolute % error | Predictions within 20% |
| --- | ---: | ---: | ---: | ---: |
| Location median baseline | 0.193 | 1.535 | **28.4%** | **38.6%** |
| Hedonic Ridge | 0.281 | 1.448 | 38.0% | 25.1% |
| **Nonlinear gradient boosting** | **0.334** | **1.394** | 31.2% | 32.3% |

The nonlinear model explains more out-of-time variation and has the lowest log RMSE. The simple location baseline still has a lower median percentage error, so the project reports the trade-off rather than cherry-picking one metric.

![Predictive feature importance](outputs/predictive_feature_importance.png)

## Robustness & risk analysis

The project includes a dedicated layer asking: **how fragile are the conclusions?**

### Duplicate-like listing sensitivity

A strict same-month property fingerprint identified **1,093 extra duplicate-like rows**, only **0.21%** of the core analytical sample. Removing them changed the nationwide median asking price/m² by only **+0.251%** and did not change the Tehran median.

A broader cross-month fingerprint identifies potential repeat-like clusters, but these are explicitly treated as an **upper-bound risk signal**, not verified duplicate properties.

![Duplicate sensitivity](outputs/duplicate_sensitivity.png)

### Bootstrap uncertainty for Tehran neighborhoods

Every Tehran neighborhood with at least **250 listings** receives a non-parametric **95% bootstrap confidence interval** for median asking price/m² using 300 replications. This makes ranking uncertainty visible rather than presenting neighborhood medians as perfectly precise.

![Tehran neighborhood uncertainty](outputs/tehran_neighborhood_uncertainty.png)

### Spatial holdout: completely unseen neighborhoods

A grouped validation design holds out entire Tehran neighborhoods. The spatial model does **not** use the neighborhood name; it must generalize from property characteristics plus approximate latitude, longitude, and privacy radius.

| Model | R² on log price/m² | Median absolute % error | Within 20% | Within 30% |
| --- | ---: | ---: | ---: | ---: |
| Tehran median baseline | -0.018 | 36.8% | 26.7% | 41.2% |
| Property-only gradient boosting | 0.154 | 25.1% | 40.9% | 57.9% |
| **Spatial gradient boosting** | **0.201** | **20.9%** | **48.2%** | **65.7%** |

This is intentionally harder than ordinary row-level validation and provides a more realistic view of geographic generalization risk.

Full risk analysis: [`outputs/ROBUSTNESS_RISK_ANALYSIS.md`](outputs/ROBUSTNESS_RISK_ANALYSIS.md)

Risk methodology: [`docs/robustness_risk_methodology.md`](docs/robustness_risk_methodology.md)

## Tehran market segmentation

Eligible Tehran neighborhoods are divided into three relative market tiers using the sample-size-stabilized neighborhood benchmark.

| Segment | Neighborhoods | Listings | Listing share | Median asking price / m² | Median size | Elevator | Parking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affordable | 35 | 22,270 | 32.9% | 56.74M | 64 m² | 62.7% | 68.4% |
| Mid-market | 34 | 18,181 | 26.9% | 85.36M | 87 m² | 76.6% | 84.2% |
| Premium | 35 | 27,191 | 40.2% | 132.99M | 111 m² | 89.3% | 96.2% |

These are **relative analytical market tiers**, not official affordability or investment classifications.

## Amenities and modernization-related indicators

After controlling for included location and property variables in the nonlinear model, elevator availability is associated with approximately **+23.2%** and parking with approximately **+14.9%** in predicted asking price/m² on the counterfactual validation sample.

These are **conditional model associations, not causal price premiums**.

![Adjusted amenity associations](outputs/predictive_amenity_associations.png)

## Methodological guardrails

The dataset does **not** directly label properties as having a validated "modern" or "traditional" architectural style. Construction year and amenities are therefore treated as **observable modernization-related indicators**, not invented architectural-style labels.

Other limitations include asking-vs-transaction price differences, incomplete platform coverage, approximate geographic fields, non-random missingness, possible repeat listings, and unobserved quality.

Full methodology: [`docs/methodology.md`](docs/methodology.md)

## Portfolio navigation

- **[`EXECUTIVE_PORTFOLIO.md`](EXECUTIVE_PORTFOLIO.md)** — recruiter-facing 30-second overview.
- [`outputs/CITY_DEEP_DIVE.md`](outputs/CITY_DEEP_DIVE.md) — Tehran neighborhoods and major-city trends.
- [`outputs/ROBUSTNESS_RISK_ANALYSIS.md`](outputs/ROBUSTNESS_RISK_ANALYSIS.md) — duplicate sensitivity, uncertainty, spatial validation, and segmentation.
- [`outputs/EXECUTIVE_SUMMARY.md`](outputs/EXECUTIVE_SUMMARY.md) — advanced modeling summary.
- [`outputs/MODEL_CARD.md`](outputs/MODEL_CARD.md) — hedonic Ridge model documentation.
- [`outputs/PREDICTIVE_MODEL_CARD.md`](outputs/PREDICTIVE_MODEL_CARD.md) — nonlinear model documentation.
- [`notebooks/01_advanced_market_analysis.ipynb`](notebooks/01_advanced_market_analysis.ipynb) — advanced modeling walkthrough.
- [`notebooks/02_city_deep_dive.ipynb`](notebooks/02_city_deep_dive.ipynb) — city/neighborhood walkthrough.
- [`notebooks/03_robustness_risk_analysis.ipynb`](notebooks/03_robustness_risk_analysis.ipynb) — robustness walkthrough.

## Repository structure

```text
Iran-Real-Estate-Market-Analysis/
├── README.md
├── EXECUTIVE_PORTFOLIO.md
├── LICENSE
├── requirements.txt
├── docs/
│   ├── methodology.md
│   └── robustness_risk_methodology.md
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_advanced_market_analysis.ipynb
│   ├── 02_city_deep_dive.ipynb
│   └── 03_robustness_risk_analysis.ipynb
├── src/
│   ├── download_data.py
│   ├── prepare_sales_data.py
│   ├── market_analysis.py
│   ├── advanced_model.py
│   ├── predictive_model.py
│   ├── city_deep_dive.py
│   └── robustness_risk_analysis.py
├── outputs/
│   ├── EXECUTIVE_SUMMARY.md
│   ├── CITY_DEEP_DIVE.md
│   ├── ROBUSTNESS_RISK_ANALYSIS.md
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
python src/robustness_risk_analysis.py
```

The raw and large processed files are intentionally excluded from Git. GitHub Actions rebuilds the project from the official source and commits compact validated outputs automatically.

## Reproducibility and data ethics

- Source data is anonymized by the publisher.
- Raw data is not committed to this repository.
- Cleaning, outlier, uncertainty, duplicate-sensitivity, and segmentation rules are implemented in code.
- Asking prices are never presented as completed-sale transaction prices.
- Geographic fields are treated as approximate rather than exact addresses or official polygons.
- Amenity-price relationships are described as associations, not causal effects.
- Predictive validation includes both temporal and geographic holdouts.
- Duplicate-like fingerprints are labelled as probabilistic diagnostics, not verified property matches.
- Simple benchmarks are retained so model complexity must justify itself empirically.

## License and attribution

The Divar source dataset is published under the **Open Database License (ODbL)**. Dataset licensing and attribution remain with the original publisher. The code in this repository is licensed separately under the MIT License; see [`LICENSE`](LICENSE).

---

*This is an analytical portfolio project and should not be interpreted as investment, valuation, legal, or certified appraisal advice.*