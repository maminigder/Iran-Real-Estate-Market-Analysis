"""Download the official Divar real-estate dataset from Hugging Face.

The raw file is stored under data/raw/ and is ignored by Git.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "divarofficial/real_estate_ads"
FILENAME = "real_estate_ads.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}/{FILENAME} ...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=RAW_DIR,
    )

    path = Path(downloaded_path)
    size_mb = path.stat().st_size / (1024**2)
    print(f"Saved to: {path}")
    print(f"File size: {size_mb:,.1f} MB")


if __name__ == "__main__":
    main()
