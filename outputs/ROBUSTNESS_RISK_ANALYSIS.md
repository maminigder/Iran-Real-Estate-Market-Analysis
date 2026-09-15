# Robustness & Risk Analysis

This layer asks a different question from the main predictive work: **how fragile are the conclusions?** It tests duplicate-like record sensitivity, neighborhood sampling uncertainty, generalization to completely unseen Tehran neighborhoods, and transparent market-tier segmentation.

## 1. Duplicate / repeat-listing risk

The strict same-month fingerprint flags **2,091 candidate rows** and **1,093 extra rows** beyond the first copy. That is **0.21%** of the core sample.

After removing only those strict duplicate-like extras, the nationwide median asking price/m² changes by **+0.251%**. This is a sensitivity test, not a claim that every flagged record is a true duplicate.

A broader fingerprint that ignores month and asking price identifies **12,182 cross-month repeat-like clusters** covering **41,181 rows**. This broader flag is deliberately treated as an upper-bound risk signal because distinct units can share the same observable characteristics.

![Duplicate sensitivity](duplicate_sensitivity.png)

## 2. Tehran neighborhood uncertainty

Bootstrap 95% confidence intervals are calculated for the median asking price/m² of every Tehran neighborhood with at least **250 listings**, using **300 bootstrap replications** per neighborhood.

These intervals quantify **sampling uncertainty conditional on the observed Divar listings**. They do not capture platform-selection bias, seller strategy, duplicate risk, or unobserved property quality.

![Neighborhood uncertainty](tehran_neighborhood_uncertainty.png)

### Most statistically precise eligible neighborhood medians

| Neighborhood | Listings | Median price/m² | Relative CI width |
| --- | ---: | ---: | ---: |
| sazamn-barnameh | 1,667 | 104,375,000 | 1.8% |
| kooy-e-ferdos | 1,377 | 101,818,182 | 1.9% |
| bagh-feyz | 824 | 100,000,000 | 2.0% |
| central-janat-abad | 1,321 | 105,882,353 | 2.0% |
| south-janat-abad | 1,483 | 109,259,259 | 2.0% |

### Least precise eligible neighborhood medians

| Neighborhood | Listings | Median price/m² | Relative CI width |
| --- | ---: | ---: | ---: |
| abshar-tehran | 271 | 60,256,410 | 33.7% |
| dehkade-olympic | 270 | 96,463,938 | 20.4% |
| chitgar | 894 | 41,176,471 | 16.5% |
| aghdasieh | 261 | 212,500,000 | 13.0% |
| shahrak-e-gharb | 480 | 186,881,720 | 11.9% |

## 3. Spatial holdout: completely unseen neighborhoods

A grouped cross-validation design holds out entire Tehran neighborhoods. No listing from a held-out neighborhood is available during training. The spatial model **does not use the neighborhood name**; it must generalize from property characteristics plus approximate latitude/longitude/privacy radius.

Across the out-of-neighborhood predictions, the property-only nonlinear model achieves **R² = 0.154** on log price/m², while the spatial model achieves **R² = 0.201** with a median absolute percentage error of **20.9%**.

This is deliberately a harder validation problem than a random row split and gives a more realistic view of geographic generalization risk.

![Spatial holdout comparison](spatial_holdout_model_comparison.png)

## 4. Affordable / Mid-market / Premium segmentation

Eligible Tehran neighborhoods are divided into three equal-count tiers using the **sample-size-stabilized neighborhood price benchmark**. The labels are relative analytical tiers, not legal or investment classifications.

- Affordable: stabilized benchmark ≤ **71,480,212** source units/m²
- Mid-market: above Affordable and ≤ **98,089,399** source units/m²
- Premium: above the Mid-market threshold

| Segment | Neighborhoods | Listings | Listing share | Median price/m² | Median size | Elevator share | Parking share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affordable | 35 | 22,270 | 32.9% | 56,744,565 | 64 m² | 62.7% | 68.4% |
| Mid-market | 34 | 18,181 | 26.9% | 85,362,319 | 87 m² | 76.6% | 84.2% |
| Premium | 35 | 27,191 | 40.2% | 132,989,691 | 111 m² | 89.3% | 96.2% |

![Tehran market tiers](tehran_market_segments.png)

## Risk interpretation

- Duplicate fingerprints are probabilistic diagnostics because no verified property identity key is available in the analytical file.
- Bootstrap intervals describe uncertainty of the observed listing sample, not uncertainty about the entire housing stock or completed transactions.
- Spatial holdout performance can still benefit from approximate coordinates and may not transfer to cities with different market structure.
- Market tiers are relative to eligible Tehran neighborhoods in this dataset and period; thresholds should not be reused as permanent market definitions.
- All monetary values remain in source units until the source denomination is independently validated.
