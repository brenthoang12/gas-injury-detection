"""
Analyse MEMS sensor behaviour during zip-lock clean-air recordings.
Overlays all available clean-air runs on a single plot per sensor so
you can see baseline drift, noise level, and session-to-session spread.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

CLEAN_AIR_FILES = {
    "20260324": "20260324-experiment/zip_lock_clean.csv",
    "20260325": "20260325-experiment/zip_lock_clean.csv",
    "20260326a": "20260326-experiment/zip_lock_clean.csv",
    "20260326b": "20260326-experiment/zip_lock_clean_2.csv",
}

MEMS_COLS   = ["voc", "nh3", "hcho"]
MEMS_LABELS = {"voc": "VOC (V)", "nh3": "NH3 (V)", "hcho": "HCHO (V)"}

SESSION_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517"]
SG_WINDOW = 31


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def smooth(series: pd.Series, window: int = SG_WINDOW) -> pd.Series:
    wl = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    wl = max(wl, 5)
    return pd.Series(savgol_filter(series, window_length=wl, polyorder=2), index=series.index)


def plot_overlay(datasets: dict[str, pd.DataFrame]) -> None:
    """One subplot per MEMS sensor; all sessions overlaid."""
    fig, axes = plt.subplots(len(MEMS_COLS), 1, figsize=(13, 4 * len(MEMS_COLS)),
                             sharex=False)
    fig.suptitle("MEMS sensor response — zip-lock clean air (all sessions)", fontsize=12)

    for ax, col in zip(axes, MEMS_COLS):
        smoothed_all = []
        for (label, df), color in zip(datasets.items(), SESSION_COLORS):
            raw  = df[col]
            smth = smooth(raw)
            smoothed_all.append(smth)
            ax.plot(df["time_s"], raw,  color=color, lw=0.5, alpha=0.3)
            ax.plot(df["time_s"], smth, color=color, lw=1.4, label=label)

        combined = pd.concat(smoothed_all)
        ax.set_ylim(combined.min() * 0.95, combined.max() * 1.05)
        ax.set_ylabel(MEMS_LABELS[col])
        ax.set_xlabel("time (s)")
        ax.set_title(f"MEMS — {col}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_stats(datasets: dict[str, pd.DataFrame]) -> None:
    """Box plot of per-session distribution for each MEMS sensor."""
    fig, axes = plt.subplots(1, len(MEMS_COLS), figsize=(5 * len(MEMS_COLS), 5))
    fig.suptitle("MEMS baseline distribution — zip-lock clean air", fontsize=12)

    for ax, col in zip(axes, MEMS_COLS):
        data   = [df[col].values for df in datasets.values()]
        labels = list(datasets.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True, medianprops={"color": "black", "lw": 1.5})
        for patch, color in zip(bp["boxes"], SESSION_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        all_vals = pd.concat([df[col] for df in datasets.values()])
        ax.set_title(f"MEMS — {col}")
        ax.set_ylabel(MEMS_LABELS[col])
        ax.set_ylim(all_vals.min() * 0.95, all_vals.max() * 1.05)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()


def print_summary(datasets: dict[str, pd.DataFrame]) -> None:
    print("\n── MEMS clean-air baseline summary ───────────────────────────────────────")
    header = f"{'session':<12}" + "".join(f"  {c:<20}" for c in MEMS_COLS)
    print(header)
    print("  " + "  ".join(["mean ± std          "] * len(MEMS_COLS)))
    print("─" * (12 + 24 * len(MEMS_COLS)))
    for label, df in datasets.items():
        row = f"{label:<12}"
        for col in MEMS_COLS:
            m, s = df[col].mean(), df[col].std()
            row += f"  {m:.4f} ± {s:.4f}       "
        print(row)
    print("──────────────────────────────────────────────────────────────────────────\n")


def main():
    datasets = {label: load(path) for label, path in CLEAN_AIR_FILES.items()}
    print_summary(datasets)
    plot_overlay(datasets)
    plot_stats(datasets)


if __name__ == "__main__":
    main()
