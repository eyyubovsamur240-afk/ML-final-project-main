import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
RANDOM_STATE = 42
DEFAULT_DATA_PATH = "data/house_sale.csv"

# Property categories (values of the 'Kateqoriya' column)
RESIDENTIAL = ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi"]  # new build, old stock, house
LAND = "Torpaq"
COMMERCIAL = "Obyekt"
NO_ROOMS = ["Torpaq", "Obyekt", "Qaraj"]  # categories where "rooms" does not apply

# Baku city centre: Fountains Square (source: Wikipedia, "Fountains Square, Baku")
CENTRE_LAT, CENTRE_LNG = 40.37083, 49.83694

# Bounding box of Azerbaijan, used to detect invalid coordinates
AZ_LAT_RANGE = (38.3, 42.0)
AZ_LNG_RANGE = (44.7, 51.0)

# Split sizes: 70% train, 15% validation, 15% test
HOLDOUT_SIZE = 0.30

# Locations with fewer TRAIN listings than this are grouped into "other"
MIN_LOCATION_COUNT = 30

# Feature groups (all names are English, see build_clean_table)
NUMERIC = ["area_m2", "land_area_sot", "rooms", "floor", "total_floors", "lat", "lng", "dist_centre_km"]
BINARY = ["has_bill_of_sale", "mortgage", "is_vip", "is_featured"]
CATEGORICAL = ["category", "repair", "location"]

# Helpers
def to_number(series: pd.Series) -> pd.Series:
    """Extract a number from text: '145 m²' -> 145.0, '1.3 sot' -> 1.3, '1 250 m²' -> 1250.0."""
    cleaned = (series.astype(str)
              .str.replace(r"[^\d.,]", "", regex=True)  # keep digits, dot and comma only
              .str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")

def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two points on Earth, in kilometres."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# Loading and de-duplication
def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV without modifying it."""
    return pd.read_csv(path, low_memory=False)

def remove_repeated_scrapes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent scrape of each listing URL. The same listing was scraped several times (up to 17 copies), and some listings changed price between scrapes, so we keep the latest copy."""
    df = df.copy()
    # Timestamps come in mixed formats (some with '+00', some without)
    df["scraped_at"] = pd.to_datetime(df["datetime_scrape_y"], format="mixed", utc=True)
    return (df.sort_values(["estate_rel_url_x", "scraped_at"]).drop_duplicates(subset="estate_rel_url_x", keep="last").reset_index(drop=True))

def parse_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Split 'Sahə' into floor area (m²) and land area (sot). Land plots ('Torpaq') store their land area in 'Sahə' in sot and have no building, so their floor area is 0. Houses store land area in 'Torpaq sahəsi'."""
    df = df.copy()
    area_raw = to_number(df["Sahə"])
    is_sot = df["Sahə"].astype(str).str.contains("sot")
    df["area_m2"] = np.where(is_sot, 0.0, area_raw)
    df["land_area_sot"] = np.where(is_sot, area_raw, to_number(df["Torpaq sahəsi"]))
    return df

# Step 2: outliers (fixed domain rules, nothing learned from the data)
def apply_outlier_rules(df: pd.DataFrame):
    """Remove data-entry errors with fixed, category-aware domain rules. Price per m² and price per sot are computed ONLY to detect errors; they are never features.Returns the filtered DataFrame and a table with the number of rows each rule flagged."""
    category = df["Kateqoriya"]
    has_building = df["area_m2"] > 0
    is_land = category == LAND
    is_residential = category.isin(RESIDENTIAL)
    is_commercial = category == COMMERCIAL
    price_per_m2 = df["price"] / df["area_m2"].where(has_building)
    price_per_sot = df["price"] / df["land_area_sot"].where(is_land)
    rooms = df["Otaq sayı"]
    area_per_room = df["area_m2"] / rooms.where(rooms > 0)
    rules = {
        "price < 5,000 AZN (buildings)":
            has_building & (df["price"] < 5_000),
        "residential price < 15,000 AZN":
            is_residential & (df["price"] < 15_000),
        "residential area < 15 m2":
            is_residential & (df["area_m2"] < 15),
        "residential area per room > 300 m2":
            is_residential & (area_per_room > 300),
        "non-commercial price/m2 < 100 or > 15,000":
            has_building & ~is_commercial & ((price_per_m2 < 100) | (price_per_m2 > 15_000)),
        "commercial price/m2 < 100 or > 60,000":
            is_commercial & ((price_per_m2 < 100) | (price_per_m2 > 60_000)),
        "land area <= 0 or > 10,000 sot":
            is_land & ((df["land_area_sot"] <= 0) | (df["land_area_sot"] > 10_000)),
        "land price/sot < 50":
            is_land & (price_per_sot < 50),}
    to_drop = pd.Series(False, index=df.index)
    log = []
    for name, mask in rules.items():
        mask = mask.fillna(False)
        log.append({"rule": name, "rows_flagged": int(mask.sum())})
        to_drop |= mask
    return df[~to_drop].reset_index(drop=True), pd.DataFrame(log)

# Step 3: hidden duplicates
def remove_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate by numeric listing ID. Some URLs differ only by a '.html' suffix ('/items/4634576' vs '/items/4634576.html'), so the URL check alone misses them."""
    df = df.copy()
    df["listing_id"] = df["estate_rel_url_x"].astype(str).str.extract(r"/items/(\d+)", expand=False)
    return (df.sort_values(["listing_id", "scraped_at"]).drop_duplicates(subset="listing_id", keep="last").reset_index(drop=True))

def remove_reposts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact re-posts: the same property posted under a different listing ID. We use the strictest key (features + exact coordinates + identical description), so that genuinely different but similar flats (e.g. identical units in one new building) are kept."""
    key = ["Kateqoriya", "price", "area_m2", "land_area_sot", "Otaq sayı", "Mərtəbə", "lat", "lng", "description"]
    return (df.sort_values("scraped_at").drop_duplicates(subset=key, keep="last").reset_index(drop=True))

# Step 4: clean modelling table
def build_clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Select and translate the columns we keep (whitelist), parse rooms and floor.
    Excluded on purpose:
      1. unit_price, total_price          -> target leakage (derived from price)
      2. owner/shop names, address, URLs  -> identifiers
      3. views                            -> only known after the listing is published
      4. repair, bill_of_sale, mortgage   -> duplicates of Təmir, Çıxarış, İpoteka
      5. Binanın növü                     -> 99.8% missing
      6. dates, image URLs, free text     -> not used as features
    """
    # Rooms: fill gaps from the 'attributes' text ("4 otaqlı, 145 m²" -> 4)
    rooms_from_text = pd.to_numeric(df["attributes"].astype(str).str.extract(r"(\d+)\s*otaqlı", expand=False), errors="coerce")
    rooms = df["Otaq sayı"].fillna(rooms_from_text)
    # Floor: "7 / 9" -> floor = 7, total_floors = 9 (basement floors like "-1 / 9" allowed)
    floor_parts = df["Mərtəbə"].astype(str).str.extract(r"(-?\d+)\s*/\s*(\d+)")
    return pd.DataFrame({
        "listing_id":       df["listing_id"],
        "price":            df["price"],
        "category":         df["Kateqoriya"],
        "area_m2":          df["area_m2"],
        "land_area_sot":    df["land_area_sot"],
        "rooms":            rooms,
        "floor":            pd.to_numeric(floor_parts[0], errors="coerce"),
        "total_floors":     pd.to_numeric(floor_parts[1], errors="coerce"),
        "repair":           df["Təmir"].map({"var": "yes", "yoxdur": "no"}).fillna("unknown"),
        "has_bill_of_sale": (df["Çıxarış"] == "var").astype(int),
        "mortgage":         (df["İpoteka"] == "var").astype(int),
        "is_vip":           df["vip"].notna().astype(int),
        "is_featured":      df["featured"].notna().astype(int),
        "city":             df["city"],        # kept for reference, NOT a feature (99.6% "bakı")
        "location":         df["location"],
        "lat":              df["lat"],
        "lng":              df["lng"],
    })


def fill_structural_missing(clean: pd.DataFrame):
    """Fill values that are missing by design with fixed rules; drop genuinely unknown rooms.
    1. land_area_sot: flats, commercial units and garages have no separate land -> 0
    2. floor / total_floors: not recorded for houses, land, commercial, offices, garages -> 0
    3. rooms: not applicable to land, commercial units and garages -> 0
    4. rooms still missing (houses with an unknown room count) -> row dropped
    """
    clean = clean.copy()
    clean["land_area_sot"] = clean["land_area_sot"].fillna(0)
    clean["floor"] = clean["floor"].fillna(0)
    clean["total_floors"] = clean["total_floors"].fillna(0)
    no_rooms = clean["category"].isin(NO_ROOMS)
    clean.loc[no_rooms, "rooms"] = clean.loc[no_rooms, "rooms"].fillna(0)
    unknown_rooms = clean["rooms"].isna()
    clean = clean[~unknown_rooms].reset_index(drop=True)
    for col in ["rooms", "floor", "total_floors"]:
        clean[col] = clean[col].astype(int)
    return clean


def add_location_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Drop coordinates outside Azerbaijan; add log(price) and distance to the city centre."""
    in_az = clean["lat"].between(*AZ_LAT_RANGE) & clean["lng"].between(*AZ_LNG_RANGE)
    clean = clean[in_az].reset_index(drop=True)
    clean["log_price"] = np.log(clean["price"])
    clean["dist_centre_km"] = haversine_km(clean["lat"], clean["lng"], CENTRE_LAT, CENTRE_LNG)
    return clean


def clean_data(path: str = DEFAULT_DATA_PATH):
    """Run the full cleaning process and return the cleaned data and logs."""
    steps = []
    df = load_raw(path)
    steps.append(("Raw data", 0, len(df)))
    n = len(df)
    df = remove_repeated_scrapes(df)
    steps.append(("Repeated scrapes of the same URL", n - len(df), len(df)))
    df = parse_areas(df)
    n = len(df)
    df, outlier_log = apply_outlier_rules(df)
    steps.append(("Outliers (fixed domain rules)", n - len(df), len(df)))
    n = len(df)
    df = remove_duplicate_ids(df)
    steps.append(("Same listing ID, different URL format", n - len(df), len(df)))
    n = len(df)
    df = remove_reposts(df)
    steps.append(("Exact re-posts (same property and description)", n - len(df), len(df)))
    clean = build_clean_table(df)
    n = len(clean)
    clean = fill_structural_missing(clean)
    steps.append(("Houses with unknown room count", n - len(clean), len(clean)))
    n = len(clean)
    clean = add_location_features(clean)
    steps.append(("Coordinates outside Azerbaijan", n - len(clean), len(clean)))
    # Safety checks
    assert clean.isna().sum().sum() == 0, "Missing values remain after cleaning"
    assert not clean["listing_id"].duplicated().any(), "Duplicated listing IDs remain"
    cleaning_log = pd.DataFrame(steps, columns=["step", "rows_removed", "rows_left"])
    return clean, cleaning_log, outlier_log

# Leakage-free split and price-tier label
def split_data(clean: pd.DataFrame, random_state: int = RANDOM_STATE):
    """70 / 15 / 15 split with the price-tier threshold taken from the training split only.
    1. Train (70%) vs holdout (30%), stratified by property category.
    2. Threshold = median price of the TRAINING split; label = 1 if price > threshold.
    3. Holdout -> validation (15%) and test (15%), stratified by the price-tier label.
    """
    train_df, hold_df = train_test_split(clean, test_size=HOLDOUT_SIZE, stratify=clean["category"], random_state=random_state)
    threshold = float(train_df["price"].median())
    train_df = train_df.assign(price_tier=(train_df["price"] > threshold).astype(int))
    hold_df = hold_df.assign(price_tier=(hold_df["price"] > threshold).astype(int))
    val_df, test_df = train_test_split(hold_df, test_size=0.50, stratify=hold_df["price_tier"], random_state=random_state)

    # No listing may appear in two splits
    assert set(train_df["listing_id"]).isdisjoint(val_df["listing_id"])
    assert set(train_df["listing_id"]).isdisjoint(test_df["listing_id"])
    assert set(val_df["listing_id"]).isdisjoint(test_df["listing_id"])
    assert len(train_df) + len(val_df) + len(test_df) == len(clean)
    return train_df, val_df, test_df, threshold

# Encoding (fitted on train only)
class FeatureEncoder:
    """Groups rare locations and one-hot encodes categoricals. fit() learns, from the training split only, which locations are frequent and the final column layout. transform() applies exactly that layout to any split; categories unseen intraining become all-zero columns."""
    def __init__(self, min_location_count: int = MIN_LOCATION_COUNT):
        self.min_location_count = min_location_count
        self.frequent_locations_ = None
        self.feature_names_ = None

    def _encode(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["location"] = data["location"].where(data["location"].isin(self.frequent_locations_), "other")
        return pd.get_dummies(data[NUMERIC + BINARY + CATEGORICAL], columns=CATEGORICAL, dtype=int)

    def fit(self, train_df: pd.DataFrame) -> "FeatureEncoder":
        counts = train_df["location"].value_counts()
        self.frequent_locations_ = set(counts[counts >= self.min_location_count].index)
        self.feature_names_ = self._encode(train_df).columns.tolist()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("FeatureEncoder must be fitted before transform()")
        return self._encode(data).reindex(columns=self.feature_names_, fill_value=0)


# ---------------------------------------------------------------------------
# Public API
def load_splits(task: str = "regression", scaled: bool = False, data_path: str = DEFAULT_DATA_PATH, random_state: int = RANDOM_STATE) -> dict:
    """Prepare NumPy arrays for regression or classification. Regression uses log(price), classification uses price_tier.
    If scaled=True, scale the features for SVM. Return train, validation, test data and other useful information."""
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    clean, _, _ = clean_data(data_path)
    train_df, val_df, test_df, threshold = split_data(clean, random_state=random_state)
    encoder = FeatureEncoder().fit(train_df)
    X_train = encoder.transform(train_df).to_numpy(dtype=float)
    X_val = encoder.transform(val_df).to_numpy(dtype=float)
    X_test = encoder.transform(test_df).to_numpy(dtype=float)
    scaler = None
    if scaled:
        scaler = StandardScaler().fit(X_train)
        X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    target = "log_price" if task == "regression" else "price_tier"
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": train_df[target].to_numpy(),
        "y_val": val_df[target].to_numpy(),
        "y_test": test_df[target].to_numpy(),
        "feature_names": encoder.feature_names_,
        "tier_threshold": threshold,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "encoder": encoder, "scaler": scaler,
    }

# Command-line summary
def main():
    parser = argparse.ArgumentParser(description="Clean the bina.az data and report the split.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to house_sale.csv")
    args = parser.parse_args()
    pd.set_option("display.width", 120)
    clean, cleaning_log, outlier_log = clean_data(args.data)
    print("=== Cleaning steps ===")
    print(cleaning_log.to_string(index=False))
    print("\n=== Outlier rules (rows flagged; rules overlap) ===")
    print(outlier_log.to_string(index=False))
    train_df, val_df, test_df, threshold = split_data(clean)
    encoder = FeatureEncoder().fit(train_df)
    print(f"\n=== Split (price-tier threshold = {threshold:,.0f} AZN, train median) ===")
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        print(f"{name:11s} rows = {len(part):6,d}  share = {len(part) / len(clean):.2f}  "
              f"premium = {part['price_tier'].mean():.3f}")
    print(f"\nFrequent locations: {len(encoder.frequent_locations_)} (+ 'other')")
    print(f"Number of features: {len(encoder.feature_names_)}")

if __name__ == "__main__":
    main()