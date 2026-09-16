# v1.0.0 — First Portfolio Release

This release marks the first complete portfolio-ready version of the **Iran Residential Real Estate Market Analysis** project.

## Highlights

- Analyzed **516,947** residential-sale listings across **420 Iranian cities**.
- Built a **Tehran and major-cities deep dive** covering neighborhood benchmarks, approximate spatial mapping, market segmentation, and composition-adjusted time trends.
- Added an interpretable **hedonic Ridge regression** and a nonlinear **gradient-boosting model** for nationwide price analysis.
- Evaluated model performance using an **out-of-time holdout** covering later listings.
- Added **spatial holdout validation** to test generalization on Tehran neighborhoods excluded entirely from training.
- Added a dedicated **robustness and risk layer** including duplicate-like listing sensitivity, bootstrap confidence intervals, uncertainty analysis, and benchmark comparisons.
- Added **Affordable / Mid-market / Premium** neighborhood segmentation for Tehran.
- Added recruiter-facing documentation including an **Executive Portfolio page** and a streamlined **Project at a Glance** README section.
- Added reproducible Python workflows, notebooks, methodology documentation, and GitHub Actions.

## Core portfolio metrics

| Metric | Result |
| --- | ---: |
| Core residential-sale sample | 516,947 listings |
| Cities covered | 420 |
| Tehran listings | 91,836 |
| Tehran neighborhood identifiers | 345 |
| Best nationwide out-of-time R² | 0.334 |
| Median error on unseen Tehran neighborhoods | 20.9% |

## Scope of v1.0.0

This version includes:

- Nationwide market overview
- City-level comparisons
- Tehran neighborhood intelligence
- Approximate spatial analysis
- Composition-adjusted time trends
- Hedonic modeling
- Nonlinear machine learning
- Temporal validation
- Spatial holdout validation
- Duplicate-risk sensitivity
- Bootstrap uncertainty analysis
- Market segmentation
- Recruiter-facing portfolio documentation

## Methodological guardrails

The analysis uses **advertised/listing prices**, not verified completed transaction prices. Geographic fields are treated as approximate, and amenity-price relationships are described as statistical associations rather than causal effects. The dataset does not directly label architectural style, so modernization-related variables are used only as observable indicators.

## Status

**v1.0.0 is considered the completed baseline release of the project.**

Future updates should be limited to bug fixes, materially better data, independent validation, or clearly justified analytical improvements.
