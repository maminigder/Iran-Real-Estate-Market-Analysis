# Iran Residential Real Estate Market Analysis — Executive Portfolio

**Mohammad Amin Igder**  
Real Estate • Market Analysis • Business Analytics • Python • Machine Learning

> A nationwide real-estate market intelligence project built from large-scale Iranian property-listing data. The goal was not simply to predict prices, but to turn a complex listing dataset into a defensible market-analysis case study with geographic, statistical, predictive, and risk-validation layers.

## 30-second overview

| | |
| --- | ---: |
| **Core analytical sample** | **516,947** residential-sale listings |
| **Geographic coverage** | **420** Iranian cities |
| **Tehran sample** | **91,836** listings |
| **Tehran neighborhoods analyzed** | **345** identifiers |
| **Primary validation methods** | Temporal holdout + unseen-neighborhood spatial holdout |
| **Main analytical stack** | Python, pandas, scikit-learn, matplotlib, GitHub Actions |

The project starts with **1,000,000 anonymized Divar real-estate advertisements**, then applies reproducible filtering, data-quality checks, outlier treatment, feature engineering, city/neighborhood analysis, machine-learning benchmarks, uncertainty analysis, duplicate-risk sensitivity, and automated reporting.

## The business problem

Residential property markets are highly local. A simple national average hides differences between cities and neighborhoods, while raw asking-price trends can be distorted by changes in the types of properties listed each month.

I built this project to answer four practical questions:

- **Where are the strongest price differences across cities and Tehran neighborhoods?**
- **Which property and location characteristics carry the most predictive information?**
- **Can a model generalize to later listings and to neighborhoods it has never seen before?**
- **How robust are the conclusions to duplicate-like listings, sampling uncertainty, and changing market composition?**

## What I built

The analysis combines several complementary layers rather than relying on one model:

**Market intelligence:** nationwide city comparisons, Tehran neighborhood benchmarking, amenity profiles, price-per-m² analysis, and Affordable / Mid-market / Premium segmentation.

**Interpretable modeling:** a regularized **hedonic Ridge regression** used to estimate controlled associations while keeping the model explainable.

**Predictive modeling:** a **target-encoded gradient-boosting model** used to capture nonlinear relationships and interactions.

**Time-series adjustment:** a city-level **composition-adjusted asking-price index** designed to reduce distortions caused by changes in property mix.

**Risk and validation:** bootstrap confidence intervals, duplicate-like listing sensitivity, temporal holdout validation, and a grouped **spatial holdout** where entire Tehran neighborhoods are excluded from training.

## Five key insights

### 1. Location is the strongest predictive signal

Permutation analysis shows that the combined **city-neighborhood location feature is the most important predictive input** in the nationwide model. This confirms that Iranian residential asking prices are strongly segmented geographically, even after controlling for size, age, rooms, property type, and amenities.

![Predictive feature importance](outputs/predictive_feature_importance.png)

### 2. Nonlinear modeling improves out-of-time explanatory power

On the same **October 2024–March 2025 temporal holdout**, the nonlinear model achieved **R² = 0.334**, compared with **0.281** for the hedonic Ridge model and **0.193** for the simple location-median baseline.

The result is deliberately reported with multiple metrics because the simple baseline still performs better on median percentage error. The project therefore avoids cherry-picking a single score.

![Model comparison](outputs/model_comparison.png)

### 3. Spatial information helps on completely unseen neighborhoods

A harder validation holds out entire Tehran neighborhoods. The model cannot rely on the neighborhood name because those areas are absent from training.

| Model | R² on log price/m² | Median absolute % error | Predictions within 20% |
| --- | ---: | ---: | ---: |
| Tehran median baseline | -0.018 | 36.8% | 26.7% |
| Property-only gradient boosting | 0.154 | 25.1% | 40.9% |
| **Spatial gradient boosting** | **0.201** | **20.9%** | **48.2%** |

Approximate latitude/longitude therefore add useful information when the model must generalize beyond familiar neighborhoods.

![Spatial holdout comparison](outputs/spatial_holdout_model_comparison.png)

### 4. Tehran market tiers show clear differences in property profile

Eligible Tehran neighborhoods were divided into three relative market segments using a sample-size-stabilized neighborhood benchmark.

| Segment | Listings | Median price/m² | Median size | Elevator | Parking |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affordable | 22,270 | 56.74M | 64 m² | 62.7% | 68.4% |
| Mid-market | 18,181 | 85.36M | 87 m² | 76.6% | 84.2% |
| Premium | 27,191 | 132.99M | 111 m² | 89.3% | 96.2% |

Premium-market listings are not only more expensive; they also tend to be larger and substantially more likely to include amenities such as elevators and parking.

![Tehran market segments](outputs/tehran_market_segments.png)

### 5. The headline conclusions are relatively robust to strict duplicate-like records

A strict same-month property fingerprint identified only **1,093 extra duplicate-like rows**, equal to about **0.21%** of the core analytical sample. Removing them changed the nationwide median asking price/m² by only **+0.251%** and did not change Tehran's median.

This does not prove that all repeat-listing risk is eliminated, but it shows that the main national price benchmark is not being driven by the strict duplicate-like cases identified by the sensitivity test.

## Why this project is different from a basic portfolio analysis

The project was designed to demonstrate **decision-quality analytics**, not only chart creation. It includes explicit minimum-sample rules, temporal validation, unseen-location validation, uncertainty intervals, benchmark comparisons, reproducible data pipelines, model limitations, and risk diagnostics.

It also avoids claims the data cannot support. The source contains **asking prices rather than verified transaction prices**, approximate geographic information, and no validated label for "modern" versus "traditional" architecture. The analysis therefore treats amenities and construction year as observable modernization-related indicators and avoids causal or appraisal claims.

## Technical workflow

`Divar source data → cleaning & validation → feature engineering → nationwide analysis → Tehran deep dive → hedonic model → nonlinear model → temporal validation → spatial holdout → bootstrap uncertainty → duplicate sensitivity → market segmentation → automated GitHub reporting`

**Tools:** Python, pandas, NumPy, scikit-learn, matplotlib, Parquet, Jupyter, GitHub Actions, Git/GitHub.

## Recruiter takeaway

This project demonstrates the ability to combine **real-estate domain knowledge with analytical problem solving**: defining a market question, structuring messy data, building defensible benchmarks, testing model generalization, communicating uncertainty, and translating technical results into business-facing market insights.

### Explore the full project

- [Full repository README](README.md)
- [Tehran & major-cities deep dive](outputs/CITY_DEEP_DIVE.md)
- [Robustness & risk analysis](outputs/ROBUSTNESS_RISK_ANALYSIS.md)
- [Advanced modeling executive summary](outputs/EXECUTIVE_SUMMARY.md)
- [Core methodology](docs/methodology.md)

---

**Important:** all monetary figures are shown in **source units** because the dataset documentation does not clearly verify the monetary denomination. Results describe the listing market represented in the dataset and should not be interpreted as certified valuations, completed transaction prices, or investment advice.
