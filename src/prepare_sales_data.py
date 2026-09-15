"""Prepare a reproducible residential-sale analytical dataset.

The script reads the large source CSV in chunks, keeps residential sale
listings, standardizes selected fields, and writes a compact Parquet file.
No manual spreadsheet cleaning is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "real_estate_ads.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "residential_sales.parquet"

USECOLS = [
    "cat2_slug",
    "cat3_slug",
    "city_slug",
    "neighborhood_slug",
    "created_at_month",
    "user_type",
    "price_mode",
    "price_value",
    "land_size",
    "building_size",
    "deed_type",
    "floor",
    "rooms_count",
    "total_floors_count",
    "unit_per_floor",
    "has_balcony",
    "has_elevator",
    "has_warehouse",
    "has_parking",
    "construction_year",
    "is_rebuilt",
    "has_heating_system",
    "has_cooling_system",
    "building_direction",
    "floor_material",
    "property_type",
    "location_latitude",
    "location_longitude",
    "location_radius",
]

PERSIAN_ARABIC_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)

ROOM_MAP = {
    "بدون اتاق": 0,
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج یا بیشتر": 5,
}


def extract_number(series: pd.Series) -> pd.Series:
    """Extract the first numeric token after normalizing Persian/Arabic digits."""
    text = series.astype("string").str.translate(PERSIAN_ARABIC_DIGITS)
    extracted = text.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def normalize_rooms(series: pd.Series) -> pd.Series:
    mapped = series.astype("string").map(ROOM_MAP)
    numeric = extract_number(series)
    return mapped.fillna(numeric).astype("Float64")


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Core project scope: residential properties offered for sale.
    df = chunk.loc[chunk["cat2_slug"].eq("residential-sell")].copy()

    # Financial and size fields are kept in the units supplied by Divar.
    df["asking_price"] = pd.to_numeric(df["price_value"], errors="coerce")
    df["building_size_sqm"] = pd.to_numeric(df["building_size"], errors="coerce")
    df["land_size_sqm"] = pd.to_numeric(df["land_size"], errors="coerce")

    # Remove rows that cannot support a price-per-square-metre calculation.
    df = df.loc[
        df["asking_price"].gt(0)
        & df["building_size_sqm"].gt(0)
    ].copy()

    df["asking_price_per_sqm"] = df["asking_price"] / df["building_size_sqm"]
    df["listing_month"] = pd.to_datetime(df["created_at_month"], errors="coerce")
    df["rooms_numeric"] = normalize_rooms(df["rooms_count"])
    df["construction_year_jalali"] = extract_number(df["construction_year"]).astype("Float64")

    # Plausibility flags are transparent and retained instead of silently
    # deleting observations. The analysis script uses them for its core sample.
    df["plausible_building_size"] = df["building_size_sqm"].between(20, 2_000)
    df["plausible_construction_year"] = df["construction_year_jalali"].between(1300, 1450) | df[
        "construction_year_jalali"
    ].isna()
    df["analysis_ready"] = (
        df["plausible_building_size"]
        & df["plausible_construction_year"]
        & np.isfinite(df["asking_price_per_sqm"])
        & df["asking_price_per_sqm"].gt(0)
        & df["city_slug"].notna()
    )

    keep = [
        "cat3_slug",
        "city_slug",
        "neighborhood_slug",
        "listing_month",
        "user_type",
        "price_mode",
        "asking_price",
        "building_size_sqm",
        "land_size_sqm",
        "asking_price_per_sqm",
        "deed_type",
        "floor",
        "rooms_numeric",
        "total_floors_count",
        "unit_per_floor",
        "has_balcony",
        "has_elevator",
        "has_warehouse",
        "has_parking",
        "construction_year_jalali",
        "is_rebuilt",
        "has_heating_system",
        "has_cooling_system",
        "building_direction",
        "floor_material",
        "property_type",
        "location_latitude",
        "location_longitude",
        "location_radius",
        "plausible_building_size",
        "plausible_construction_year",
        "analysis_ready",
    ]
    return df[keep]


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {RAW_FILE}\n"
            "Run `python src/download_data.py` first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prepared_chunks: list[pd.DataFrame] = []
    total_rows = 0
    residential_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(
            RAW_FILE,
            usecols=USECOLS,
            chunksize=100_000,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)
        cleaned = clean_chunk(chunk)
        residential_rows += len(cleaned)
        prepared_chunks.append(cleaned)
        print(
            f"Chunk {i}: source={len(chunk):,}, "
            f"usable residential-sale rows={len(cleaned):,}"
        )

    result = pd.concat(prepared_chunks, ignore_index=True)
    result.to_parquet(OUTPUT_FILE, index=False)

    print("\nPreparation complete")
    print(f"Source rows read: {total_rows:,}")
    print(f"Residential-sale rows with price and building size: {residential_rows:,}")
    print(f"Analysis-ready rows: {int(result['analysis_ready'].sum()):,}")
    print(f"Cities represented: {result['city_slug'].nunique(dropna=True):,}")
    if result["listing_month"].notna().any():
        print(
            "Observed listing-month range: "
            f"{result['listing_month'].min().date()} to "
            f"{result['listing_month'].max().date()}"
        )
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
