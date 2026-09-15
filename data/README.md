# Data directory

This project uses the official **Divar Real Estate Ads Dataset** published on Hugging Face:

https://huggingface.co/datasets/divarofficial/real_estate_ads

The source dataset contains 1,000,000 anonymized real-estate advertisements and is distributed under the Open Database License (ODbL).

## Folder layout

- `data/raw/` — downloaded source files. Not tracked by Git.
- `data/processed/` — cleaned analytical datasets produced by the preparation script. Not tracked by Git.

The repository intentionally does not redistribute the full source dataset. Run:

```bash
python src/download_data.py
python src/prepare_sales_data.py
```

to reproduce the local analytical dataset.

## Important interpretation notes

- `price_value` is treated as a listing/asking value, not a confirmed transaction price.
- The official dataset card describes a six-month 2024 collection period, while the current dataset viewer shows `created_at_month` values spanning a wider range. The pipeline therefore reports observed date coverage directly from the downloaded file instead of assuming a period in advance.
- The source documentation does not clearly state the monetary denomination of `price_value`; project outputs preserve the source units unless independently verified.
