"""
fake_house_sale.py — a tiny SYNTHETIC file in the format of the Kaggle house_sale.csv.

It copies the *format* of the real scrape (column names, '145 m²' / '1.3 sot' areas,
'7 / 9' floors, repeated scrapes, '.html' URL variants, re-posts, outliers, leakage and
identifier columns) so src/data_prep.py can be tested in CI without the real data.
The numbers mean nothing: never use them in the report.

    python -m tests.fake_house_sale /tmp/house_sale.csv 600
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CATEGORIES = {"Yeni tikili": 0.55, "Köhnə tikili": 0.18, "Həyət evi/Bağ evi": 0.15,
              "Torpaq": 0.05, "Obyekt": 0.04, "Ofis": 0.02, "Qaraj": 0.01}
LOCATIONS = {  # name: (lat, lng)
    "Səbail r.": (40.36, 49.83), "Nəsimi r.": (40.38, 49.83), "Yasamal r.": (40.38, 49.81),
    "Nərimanov r.": (40.40, 49.87), "Xətai r.": (40.38, 49.95), "Binəqədi r.": (40.46, 49.83),
    "Masazır q.": (40.48, 49.76), "Mərdəkan q.": (40.49, 50.14), "Sahil m.": (40.37, 49.84),
    "Rare place q.": (40.30, 49.70),
}


def make_fake_house_sale(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(list(CATEGORIES), n, p=list(CATEGORIES.values()))
    loc_names = list(LOCATIONS)
    loc_p = np.array([1.0] * (len(loc_names) - 1) + [0.02])
    loc = rng.choice(loc_names, n, p=loc_p / loc_p.sum())
    lat = np.array([LOCATIONS[x][0] for x in loc]) + rng.normal(0, 0.01, n)
    lng = np.array([LOCATIONS[x][1] for x in loc]) + rng.normal(0, 0.01, n)

    is_land = cat == "Torpaq"
    is_house = cat == "Həyət evi/Bağ evi"
    is_flat = np.isin(cat, ["Yeni tikili", "Köhnə tikili"])
    has_rooms = ~np.isin(cat, ["Torpaq", "Obyekt", "Qaraj"])
    rooms = np.where(has_rooms, rng.integers(1, 7, n), np.nan)
    area = np.round(rng.uniform(30, 40, n) * np.where(has_rooms, rooms, 3))
    sot = np.round(rng.uniform(1, 20, n), 1)
    price = np.round(area * rng.uniform(1_000, 3_000, n), -2)
    price = np.where(is_land, np.round(sot * rng.uniform(5_000, 20_000, n), -2), price)
    floor = rng.integers(1, 10, n)
    total = floor + rng.integers(0, 10, n)

    df = pd.DataFrame({
        "estate_rel_url_x": [f"/items/{4_000_000 + i}" for i in range(n)],
        "datetime_scrape_y": "2024-10-05 22:14:10.089116+00",
        "price": price.astype(float),
        "currency_x": "AZN",
        "location": loc,
        "attributes": [
            (f"{int(r)} otaqlı, " if not np.isnan(r) else "") + f"{int(a)} m²"
            + (f", {f}/{t} mərtəbə" if fl else "")
            for r, a, f, t, fl in zip(rooms, area, floor, total, is_flat, strict=True)
        ],
        "city": "bakı",
        "vip": np.where(rng.random(n) < 0.1, "vipped", None),
        "featured": np.where(rng.random(n) < 0.05, "featured", None),
        "description": [f"Satılır, elan {i}" for i in range(n)],
        "unit_price": [f"{int(p / a):,} AZN/m²".replace(",", " ") for p, a in zip(price, area, strict=True)],
        "total_price": price.astype(float),
        "owner_name": "Owner",
        "address": "Some küç.",
        "lat": lat,
        "lng": lng,
        "views": rng.integers(10, 2000, n),
        "Binanın növü": None,
        "Kateqoriya": cat,
        "Mərtəbə": np.where(is_flat, [f"{f} / {t}" for f, t in zip(floor, total, strict=True)], None),
        "Otaq sayı": rooms,
        "Sahə": np.where(is_land, [f"{s} sot" for s in sot], [f"{int(a)} m²" for a in area]),
        "Torpaq sahəsi": np.where(is_house, [f"{s} sot" for s in sot], None),
        "Təmir": rng.choice(["var", "yoxdur", None], n, p=[0.8, 0.15, 0.05]),
        "Çıxarış": rng.choice(["var", "yoxdur"], n, p=[0.8, 0.2]),
        "İpoteka": np.where(rng.random(n) < 0.3, "var", None),
    })
    # Rooms only in the attributes text for a few listings; a few houses with no rooms at all
    gap = np.flatnonzero(is_flat)[:5]
    df.loc[gap, "Otaq sayı"] = np.nan
    unknown = np.flatnonzero(is_house)[:3]
    df.loc[unknown, "Otaq sayı"] = np.nan
    df.loc[unknown, "attributes"] = "150 m²"
    # Outliers caught by the domain rules, and one coordinate outside Azerbaijan
    df.loc[0, "price"] = 1_000.0
    df.loc[1, ["Sahə", "attributes"]] = ["10 m²", "1 otaqlı, 10 m²"]
    df.loc[2, ["lat", "lng"]] = [32.69, 12.59]

    # Repeated scrapes (later copy wins), '.html' variants and a re-post under a new ID
    later = df.sample(n // 5, random_state=seed).assign(
        datetime_scrape_y="2024-11-13 20:51:47.165261")
    html = df.sample(n // 20, random_state=seed + 1)
    html = html.assign(estate_rel_url_x=html["estate_rel_url_x"] + ".html",
                       datetime_scrape_y="2024-10-09 05:02:29.349209+00")
    repost = df.sample(n // 20, random_state=seed + 2)
    repost = repost.assign(estate_rel_url_x=[f"/items/{9_000_000 + i}" for i in range(len(repost))],
                           datetime_scrape_y="2024-10-10 05:02:29.349209+00")
    return pd.concat([df, later, html, repost], ignore_index=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "house_sale_fake.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    make_fake_house_sale(n).to_csv(out, index=False)
    print(f"wrote {out} (SYNTHETIC, for testing only)")
