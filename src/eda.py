"""
eda.py — the four EDA figures for the report (Problem 1b), reproducible from code.

Same figures as eda.ipynb, rebuilt from src/data_prep.py so `python -m src.run_all`
(or this module) regenerates them from a clean checkout:
  fig1_price_distribution.png   price and log(price) histograms
  fig2_area_rooms_vs_price.png  floor area and rooms vs price
  fig3_location_effects.png     price by location (lat/lng)
  fig4_class_balance.png        price-tier balance per split

    python -m src.eda --data data/house_sale.csv --out report/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import data_prep  # noqa: E402

STYLE = "seaborn-v0_8-whitegrid"
BAKU_LAT, BAKU_LNG = (40.25, 40.65), (49.6, 50.4)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation (Pearson on ranks), robust to outliers."""
    return float(a.rank().corr(b.rank()))


def fig_price_distribution(clean: pd.DataFrame, path: Path) -> dict:
    price, log_price = clean["price"], np.log(clean["price"])
    median, mean, p99 = price.median(), price.mean(), price.quantile(0.99)
    n_above = int((price > p99).sum())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(price[price <= p99], bins=80, color="steelblue")
    axes[0].axvline(median, color="red", linestyle="--", label=f"Median = {median:,.0f} AZN")
    axes[0].axvline(mean, color="orange", linestyle="--", label=f"Mean = {mean:,.0f} AZN")
    axes[0].set_title(f"(a) Sale price, skewness = {price.skew():.1f}\n"
                      f"(axis cut at 99th pct; {n_above} listings beyond)")
    axes[0].set_xlabel("Price (thousand AZN)")
    axes[0].set_ylabel("Number of listings")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:,.0f}"))
    axes[0].legend()
    axes[1].hist(log_price, bins=80, color="seagreen")
    axes[1].axvline(np.log(median), color="red", linestyle="--", label="Median")
    axes[1].set_title(f"(b) log(price), skewness = {log_price.skew():.2f}\n(full range)")
    axes[1].set_xlabel("log(price in AZN)")
    axes[1].set_ylabel("Number of listings")
    axes[1].legend()
    fig.suptitle(f"Figure 1. Price distribution of {len(clean):,} cleaned bina.az listings", y=1.03)
    _save(fig, path)
    return {"median_price": median, "mean_price": mean, "skew_price": price.skew(),
            "skew_log_price": log_price.skew(), "p99": p99, "n_above_p99": n_above}


def fig_area_rooms_vs_price(clean: pd.DataFrame, path: Path) -> dict:
    buildings = clean[clean["area_m2"] > 0]  # land plots have no floor area
    with_rooms = clean[clean["rooms"] > 0]   # categories with a room count
    rho_area = spearman(buildings["area_m2"], buildings["price"])
    rho_rooms = spearman(with_rooms["rooms"], with_rooms["price"])
    order = ["1", "2", "3", "4", "5", "6", "7+"]
    groups = with_rooms["rooms"].clip(upper=7).astype(int).astype(str).replace("7", "7+")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for category, part in buildings.groupby("category"):
        axes[0].scatter(part["area_m2"], part["price"], s=6, alpha=0.3, linewidths=0,
                        label=category)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(f"(a) Floor area vs price (log-log), Spearman ρ = {rho_area:.2f}")
    axes[0].set_xlabel("Floor area (m², log scale)")
    axes[0].set_ylabel("Price (AZN, log scale)")
    axes[0].legend(title="Category", markerscale=3, fontsize=8)
    present = [g for g in order if (groups == g).any()]
    axes[1].boxplot([with_rooms.loc[groups == g, "price"] for g in present], labels=present,
                    showfliers=False, patch_artist=True,
                    boxprops={"facecolor": "lightsteelblue"})
    axes[1].set_yscale("log")
    axes[1].set_title(f"(b) Price by number of rooms, Spearman ρ = {rho_rooms:.2f}")
    axes[1].set_xlabel("Number of rooms")
    axes[1].set_ylabel("Price (AZN, log scale)")
    fig.suptitle("Figure 2. Size and price (land plots excluded: no floor area)", y=1.02)
    _save(fig, path)
    return {"spearman_area_price": rho_area, "spearman_rooms_price": rho_rooms}


def fig_location_effects(clean: pd.DataFrame, path: Path) -> dict:
    geo = clean.assign(log_price=np.log(clean["price"])).sort_values("log_price")
    baku = geo[geo["lat"].between(*BAKU_LAT) & geo["lng"].between(*BAKU_LNG)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = axes[0].scatter(geo["lng"], geo["lat"], c=geo["log_price"], cmap="viridis", s=3,
                         alpha=0.6)  # expensive points drawn on top
    axes[0].set_title(f"(a) All listings (n = {len(geo):,})")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc, ax=axes[0], label="log(price)")
    hb = axes[1].hexbin(baku["lng"], baku["lat"], C=baku["log_price"],
                        reduce_C_function=np.median, gridsize=45, cmap="viridis", mincnt=5)
    axes[1].set_title(f"(b) Baku area: median log(price) per cell (n = {len(baku):,})")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(hb, ax=axes[1], label="median log(price)")
    fig.suptitle("Figure 3. Location effects on price", y=1.02)
    _save(fig, path)
    return {"spearman_dist_centre_price": spearman(clean["dist_centre_km"], clean["price"])}


def fig_class_balance(splits: dict, threshold: float, path: Path) -> dict:
    balance = pd.DataFrame({
        name: part["price_tier"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
        for name, part in splits.items()
    }).T
    balance.columns = ["standard (0)", "premium (1)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    balance.plot(kind="bar", ax=ax, color=["lightsteelblue", "darkorange"], rot=0)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"Figure 4. Price-tier class balance (threshold = {threshold:,.0f} AZN, "
                 "train median)")
    ax.set_ylabel("Share of listings")
    ax.set_ylim(0, 0.65)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    _save(fig, path)
    return {f"premium_share_{name}": float(part["price_tier"].mean())
            for name, part in splits.items()}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(data_path: str = data_prep.DATA_PATH, out_dir: str = "report/figures") -> dict:
    """Clean and split the data, write the four figures to out_dir, return the key numbers."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean, _, _ = data_prep.clean_data(data_path)
    train_df, val_df, test_df, threshold = data_prep.split_data(clean)
    splits = {"train": train_df, "validation": val_df, "test": test_df}
    stats = {"rows": len(clean), "tier_threshold": threshold}
    with plt.style.context(STYLE):
        stats |= fig_price_distribution(clean, out / "fig1_price_distribution.png")
        stats |= fig_area_rooms_vs_price(clean, out / "fig2_area_rooms_vs_price.png")
        stats |= fig_location_effects(clean, out / "fig3_location_effects.png")
        stats |= fig_class_balance(splits, threshold, out / "fig4_class_balance.png")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the EDA figures for the report.")
    parser.add_argument("--data", default=data_prep.DATA_PATH, help="path to house_sale.csv")
    parser.add_argument("--out", default="report/figures", help="output folder")
    args = parser.parse_args()
    stats = make_figures(args.data, args.out)
    for key, value in stats.items():
        print(f"{key:28s} {value:,.3f}" if isinstance(value, float) else f"{key:28s} {value:,}")
    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
