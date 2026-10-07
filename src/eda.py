import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from . import data_prep

# Configuration
FIGURES_DIR = "report/figures"
CATEGORY_COLORS = {
    "Yeni tikili": "tab:blue",
    "Köhnə tikili": "tab:orange",
    "Həyət evi/Bağ evi": "tab:red",
    "Obyekt": "tab:green",
    "Ofis": "tab:purple",
    "Qaraj": "tab:gray",
}
BAKU_LAT_RANGE = (40.25, 40.65)
BAKU_LNG_RANGE = (49.6, 50.4)

# Helpers
def _save(fig, out_dir, name):
    """Save a figure as PNG into out_dir and close it."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# Figure 1: price distribution
def plot_price_distribution(clean, out_dir=FIGURES_DIR):
    """Histogram of the raw price (axis cut at the 99th percentile) and of log(price). Returns the key numbers for the report."""
    price, log_price = clean["price"], clean["log_price"]
    stats = {
        "n_listings": len(clean),
        "price_median": float(price.median()),
        "price_mean": float(price.mean()),
        "price_skew": float(price.skew()),
        "log_price_skew": float(log_price.skew()),
        "price_p99": float(price.quantile(0.99)),
    }
    n_above = int((price > stats["price_p99"]).sum())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(price[price <= stats["price_p99"]], bins=80, color="steelblue")
    axes[0].axvline(stats["price_median"], color="red", linestyle="--", label=f"Median = {stats['price_median']:,.0f} AZN")
    axes[0].axvline(stats["price_mean"], color="orange", linestyle="--", label=f"Mean = {stats['price_mean']:,.0f} AZN")
    axes[0].set_title(f"(a) Sale price, skewness = {stats['price_skew']:.1f}\n(axis cut at 99th pct; {n_above} listings beyond)")
    axes[0].set_xlabel("Price (thousand AZN)")
    axes[0].set_ylabel("Number of listings")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:,.0f}"))
    axes[0].legend()
    axes[1].hist(log_price, bins=80, color="seagreen")
    axes[1].axvline(np.log(stats["price_median"]), color="red", linestyle="--", label="Median")
    axes[1].set_title(f"(b) log(price), skewness = {stats['log_price_skew']:.2f}\n(full range)")
    axes[1].set_xlabel("log(price in AZN)")
    axes[1].set_ylabel("Number of listings")
    axes[1].legend()
    fig.suptitle(f"Figure 1. Price distribution of {len(clean):,} cleaned bina.az listings", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_price_distribution.png")
    return stats

# Figure 2: size and price
def plot_area_rooms_vs_price(clean, out_dir=FIGURES_DIR):
    """Floor area vs price (log-log, coloured by category) and price by number of rooms. Land plots are excluded (no floor area)."""
    buildings = clean[clean["area_m2"] > 0]
    with_rooms = clean[clean["rooms"] > 0]
    stats = {
        "spearman_area_price": float(buildings["area_m2"].corr(buildings["price"], method="spearman")),
        "spearman_rooms_price": float(with_rooms["rooms"].corr(with_rooms["price"], method="spearman")),
    }
    # Rooms 7 and more are grouped as "7+"
    rooms_group = with_rooms["rooms"].clip(upper=7)
    order = [1, 2, 3, 4, 5, 6, 7]
    labels = ["1", "2", "3", "4", "5", "6", "7+"]
    stats["median_price_by_rooms"] = {label: float(with_rooms.loc[rooms_group == r, "price"].median()) for r, label in zip(order, labels, strict=True)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for category, part in buildings.groupby("category"):
        axes[0].scatter(part["area_m2"], part["price"], s=4, alpha=0.3, linewidths=0, color=CATEGORY_COLORS.get(category, "black"), label=category)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(f"(a) Floor area vs price (log-log), Spearman ρ = {stats['spearman_area_price']:.2f}")
    axes[0].set_xlabel("Floor area (m², log scale)")
    axes[0].set_ylabel("Price (AZN, log scale)")
    axes[0].legend(title="Category", markerscale=4, fontsize=8)
    groups = [with_rooms.loc[rooms_group == r, "price"] for r in order]
    positions = list(range(1, len(order) + 1))
    axes[1].boxplot(groups, positions=positions, showfliers=False, patch_artist=True, boxprops={"facecolor": "lightsteelblue"}, medianprops={"color": "black"})
    axes[1].set_xticks(positions, labels)
    axes[1].set_yscale("log")
    axes[1].set_title(f"(b) Price by number of rooms, Spearman ρ = {stats['spearman_rooms_price']:.2f}")
    axes[1].set_xlabel("Number of rooms")
    axes[1].set_ylabel("Price (AZN, log scale)")
    fig.suptitle("Figure 2. Size and price (land plots excluded: no floor area)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_area_rooms_vs_price.png")
    return stats

# Figure 3: location effects
def plot_location_effects(clean, out_dir=FIGURES_DIR, min_listings=200):
    """All listings coloured by log(price) and a Baku map of the median log(price) per hexagon. Returns the most and least expensive locations."""
    loc_stats = clean.groupby("location")["price"].agg(["count", "median"])
    loc_stats = loc_stats[loc_stats["count"] >= min_listings].sort_values("median", ascending=False)
    stats = {
        "spearman_dist_price": float(clean["dist_centre_km"].corr(clean["price"], method="spearman")),
        "most_expensive_locations": loc_stats.head(5)["median"].to_dict(),
        "cheapest_locations": loc_stats.tail(5)["median"].to_dict(),
    }
    baku = clean[clean["lat"].between(*BAKU_LAT_RANGE) & clean["lng"].between(*BAKU_LNG_RANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ordered = clean.sort_values("log_price")  # expensive points drawn on top
    points = axes[0].scatter(ordered["lng"], ordered["lat"], c=ordered["log_price"], cmap="viridis", s=3, alpha=0.6)
    axes[0].set_title(f"(a) All listings (n = {len(clean):,})")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(points, ax=axes[0], label="log(price)")
    cells = axes[1].hexbin(baku["lng"], baku["lat"], C=baku["log_price"], reduce_C_function=np.median, gridsize=45, cmap="viridis", mincnt=5)
    axes[1].set_title(f"(b) Baku area: median log(price) per cell (n = {len(baku):,})")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(cells, ax=axes[1], label="median log(price)")
    fig.suptitle("Figure 3. Location effects on price", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_location_effects.png")
    return stats

# Figure 4: class balance of the price-tier label
def plot_class_balance(train_df, val_df, test_df, threshold, out_dir=FIGURES_DIR):
    """Share of standard (0) and premium (1) listings in each split."""
    names = ["train", "validation", "test"]
    premium = np.array([part["price_tier"].mean() for part in (train_df, val_df, test_df)])
    standard = 1 - premium
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_0 = ax.bar(x - width / 2, standard, width, color="lightsteelblue", label="standard (0)")
    bars_1 = ax.bar(x + width / 2, premium, width, color="darkorange", label="premium (1)")
    ax.bar_label(bars_0, fmt="%.3f", fontsize=8)
    ax.bar_label(bars_1, fmt="%.3f", fontsize=8)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Share of listings")
    ax.set_title(f"Figure 4. Price-tier class balance (threshold = {threshold:,.0f} AZN, train median)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig4_class_balance.png")
    return {"premium_share": dict(zip(names, premium.round(3).tolist(), strict=True)),
            "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)}}

# Public API
def make_eda_figures(data_path=data_prep.DATA_PATH, out_dir=FIGURES_DIR):
    """Clean the data, split it, draw Figures 1-4 into out_dir and return the numbers quoted in the report."""
    clean, _, _ = data_prep.clean_data(data_path)
    train_df, val_df, test_df, threshold = data_prep.split_data(clean)
    stats = {"tier_threshold": threshold}
    stats.update(plot_price_distribution(clean, out_dir))
    stats.update(plot_area_rooms_vs_price(clean, out_dir))
    stats.update(plot_location_effects(clean, out_dir))
    stats.update(plot_class_balance(train_df, val_df, test_df, threshold, out_dir))
    return stats

# Command-line summary
def main():
    parser = argparse.ArgumentParser(description="Draw the EDA figures (Figures 1-4) into report/figures/.")
    parser.add_argument("--data", default=data_prep.DATA_PATH, help="path to house_sale.csv")
    parser.add_argument("--out", default=FIGURES_DIR, help="folder for the figures")
    args = parser.parse_args()
    stats = make_eda_figures(args.data, args.out)
    print(f"=== EDA figures saved to {args.out}/ ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()