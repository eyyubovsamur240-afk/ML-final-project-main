# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## How to place it
1. Download the dataset archive from Kaggle.
2. Extract it so the CSV lives **directly in this folder**:
   ```
   data/house_sale.csv
   ```
   (`house_sale.csv` is the file inside the Kaggle archive: about 100,775 rows, 51 columns.)
   With the Kaggle API: `kaggle datasets download -d sehriyarmemmedli/binaaz-sale-project -p data --unzip`
3. If your filename differs, set the path once at the top of
   `src/data_prep.py` (the `DATA_PATH` constant) — do not scatter paths
   across the codebase.

## Notes / gotchas
- Column names are in **Azerbaijani** (e.g. `Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Keep the UTF-8 encoding when you read the file.
- **Leakage warning:** any column derived from the target `price`
  (e.g. a per-m² unit price, or a precomputed total) must be **dropped**
  before training. Name the columns you removed in your report.
- Record the dataset version/date you downloaded, for reproducibility.
