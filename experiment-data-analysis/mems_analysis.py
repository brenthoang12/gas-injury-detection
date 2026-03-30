"""
MEMS sensor (voc, nh3, hcho) comparison across sweat and blood sessions.

Metrics per channel:
  - peak_delta  : max(smoothed) - baseline_mean   (how much it rises above baseline)
  - auc_delta   : trapezoid integral of (smoothed - baseline) over time  (total exposure)
  - time_to_peak: seconds from t=0 to where the peak occurs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

MEMS_COLS   = ["voc", "nh3", "hcho"]
MEMS_UNITS  = {"voc": "V", "nh3": "V", "hcho": "V"}
MEMS_COLORS = {"voc": "#378ADD", "nh3": "#1D9E75", "hcho": "#D85A30"}

BASELINE_S = 60   # first N seconds used to compute baseline mean

DATASETS = [
    {"label": "sweat (mar25)",    "path": "20260325-experiment/sweat.csv",          "type": "sweat"},
    {"label": "sweat (mar26)",    "path": "20260326-experiment/sweat.csv",           "type": "sweat"},
    {"label": "blood #1 (mar26)", "path": "20260326-experiment/1.5blood_sample_1.csv", "type": "blood"},
    {"label": "blood #2 (mar26)", "path": "20260326-experiment/1.5blood_sample_2.csv", "type": "blood"},
    {"label": "blood #3 (mar26)", "path": "20260326-experiment/1.5blood_sample_3.csv", "type": "blood"},
    {"label": "2blood+sweat (mar26)", "path": "20260326-experiment/2blood1sweat.csv",  "type": "mixed"},
]

TYPE_COLORS = {"sweat": "#1D9E75", "blood": "#E84040", "mixed": "#9B59B6"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def smooth(series: pd.Series, window: int = 21) -> pd.Series:
    wl = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    wl = max(wl, 5)
    return pd.Series(savgol_filter(series, window_length=wl, polyorder=2), index=series.index)


def compute_metrics(df: pd.DataFrame, baseline_s: float = BASELINE_S) -> dict:
    """Return a dict keyed by channel with peak_delta, auc_delta, time_to_peak."""
    results = {}
    for col in MEMS_COLS:
        s = smooth(df[col])
        baseline_mask = df["time_s"] <= baseline_s
        baseline_mean = s[baseline_mask].mean() if baseline_mask.any() else s.iloc[0]

        delta = s - baseline_mean
        peak_idx   = delta.idxmax()
        peak_delta = delta[peak_idx]
        time_to_peak = df["time_s"][peak_idx]
        auc_delta = float(np.trapezoid(delta.clip(lower=0), df["time_s"]))

        results[col] = {
            "baseline_mean": baseline_mean,
            "peak_delta":    peak_delta,
            "auc_delta":     auc_delta,
            "time_to_peak":  time_to_peak,
        }
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_timeseries() -> None:
    """One subplot per MEMS channel; all sessions overlaid."""
    fig, axes = plt.subplots(len(MEMS_COLS), 1, figsize=(13, 4 * len(MEMS_COLS)), sharex=False)

    for ax, col in zip(axes, MEMS_COLS):
        for ds in DATASETS:
            df  = load(ds["path"])
            s   = smooth(df[col])
            baseline_mean = s[df["time_s"] <= BASELINE_S].mean()
            delta = s - baseline_mean

            color = TYPE_COLORS[ds["type"]]
            ls    = "--" if ds["type"] == "sweat" else "-"
            ax.plot(df["time_s"], delta, label=ds["label"], color=color,
                    lw=1.4, linestyle=ls, alpha=0.85)

        ax.axhline(0, color="black", lw=0.7, linestyle=":")
        ax.set_title(f"{col} — delta from baseline")
        ax.set_ylabel(f"Δ {MEMS_UNITS[col]}")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("MEMS sensors — delta from baseline (smoothed)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_metrics() -> None:
    """Bar chart: peak_delta and auc_delta per channel per session."""
    records = []
    for ds in DATASETS:
        df      = load(ds["path"])
        metrics = compute_metrics(df)
        for col, m in metrics.items():
            records.append({
                "label":       ds["label"],
                "type":        ds["type"],
                "channel":     col,
                **m,
            })
    mdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, len(MEMS_COLS), figsize=(5 * len(MEMS_COLS), 5), sharey=False)
    fig2, axes2 = plt.subplots(1, len(MEMS_COLS), figsize=(5 * len(MEMS_COLS), 5), sharey=False)

    for ax, ax2, col in zip(axes, axes2, MEMS_COLS):
        sub = mdf[mdf["channel"] == col]
        colors = [TYPE_COLORS[t] for t in sub["type"]]

        # peak delta
        bars = ax.bar(sub["label"], sub["peak_delta"], color=colors, edgecolor="white")
        ax.set_title(f"{col} — peak Δ")
        ax.set_ylabel("peak Δ (V)")
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

        # AUC delta
        bars2 = ax2.bar(sub["label"], sub["auc_delta"], color=colors, edgecolor="white")
        ax2.set_title(f"{col} — AUC Δ")
        ax2.set_ylabel("AUC (V·s)")
        ax2.tick_params(axis="x", rotation=30)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}",
                     ha="center", va="bottom", fontsize=7)

    # legend proxy
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in TYPE_COLORS.items()]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle("Peak delta above baseline — per channel", fontsize=12)
    fig.tight_layout()

    fig2.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig2.suptitle("AUC (integrated delta above baseline) — per channel", fontsize=12)
    fig2.tight_layout()

    plt.show()


def print_summary() -> None:
    """Print a summary table of all metrics."""
    rows = []
    for ds in DATASETS:
        df      = load(ds["path"])
        metrics = compute_metrics(df)
        for col, m in metrics.items():
            rows.append({
                "session":      ds["label"],
                "type":         ds["type"],
                "channel":      col,
                "baseline (V)": round(m["baseline_mean"], 4),
                "peak Δ (V)":   round(m["peak_delta"], 4),
                "AUC (V·s)":    round(m["auc_delta"], 1),
                "t_peak (s)":   round(m["time_to_peak"], 1),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print_summary()
    plot_timeseries()
    plot_metrics()


if __name__ == "__main__":
    main()
