"""
H2S & Alcohol (ETOH) comparison — sweat vs blood.

Sweat : 20260326-experiment/sweat.csv
Blood : 20260326-experiment/1.5blood_sample_1.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from filter import (
    M_H2S, M_ETOH,
    H2S_OFFSET_PPM, ETOH_OFFSET_PPM,
    H2S_SCALE, VGAS_ETOH_OFFSET,
    detect_outliers_iqr, detect_outliers_roc,
    handle_outliers, lowpass_filter,
)

SWEAT_PATH = "20260325-experiment/sweat.csv"
BLOOD_PATH = "20260326-experiment/1.5blood_sample_1.csv"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def process_h2s(df: pd.DataFrame) -> pd.Series:
    vref     = df["h2s_vref"].mean()
    ppm      = ((df["h2s_vgas"] - vref) / M_H2S - H2S_OFFSET_PPM).clip(lower=0) * H2S_SCALE
    outliers = detect_outliers_iqr(ppm) | detect_outliers_roc(ppm)
    clean    = handle_outliers(ppm, outliers, method="interpolate")
    smooth   = savgol_filter(clean, window_length=51, polyorder=2)
    return lowpass_filter(pd.Series(smooth, index=df.index), cutoff_hz=0.005).clip(lower=0)


def process_etoh(df: pd.DataFrame) -> pd.Series:
    vref     = df["etoh_vref"].mean()
    vgas     = df["etoh_vgas"] - VGAS_ETOH_OFFSET
    ppm      = ((vgas - vref) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
    outliers = detect_outliers_iqr(ppm, 30) | detect_outliers_roc(ppm)
    clean    = handle_outliers(ppm, outliers, method="interpolate")
    return pd.Series(gaussian_filter1d(clean, sigma=80), index=df.index).clip(lower=0)


def plot_comparison(df_sweat: pd.DataFrame, df_blood: pd.DataFrame) -> None:
    datasets = [
        (df_sweat, "Sweat (Mar 25)",    "#1D9E75"),
        (df_blood, "Blood #1 (Mar 26)", "#E84040"),
    ]

    # pre-compute smoothed signals
    h2s_smooth  = [process_h2s(df)  for df, *_ in datasets]
    etoh_smooth = [process_etoh(df) for df, *_ in datasets]

    # shared y-axis limits across both columns
    h2s_ymax  = max(s.max() for s in h2s_smooth)  * 1.15
    etoh_ymax = max(s.max() for s in etoh_smooth) * 1.15

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # layout: row 0 = H2S, row 1 = ETOH; col 0 = sweat, col 1 = blood

    for col_idx, ((df, label, color), h2s, etoh) in enumerate(zip(datasets, h2s_smooth, etoh_smooth)):
        vref     = df["h2s_vref"].mean()
        h2s_raw  = ((df["h2s_vgas"] - vref) / M_H2S - H2S_OFFSET_PPM).clip(lower=0) * H2S_SCALE
        vref_e   = df["etoh_vref"].mean()
        vgas_e   = df["etoh_vgas"] - VGAS_ETOH_OFFSET
        etoh_raw = ((vgas_e - vref_e) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)

        # ── H2S ──────────────────────────────────────────────────────────────
        ax = axes[0, col_idx]
        ax.plot(df["time_s"], h2s_raw, color=color, lw=0.5, alpha=0.3)
        ax.plot(df["time_s"], h2s,     color=color, lw=1.8, label="filtered")
        ax.set_ylim(0, h2s_ymax)
        ax.set_title(f"H2S — {label}")
        ax.set_ylabel("H2S (ppm)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)

        # ── ETOH ─────────────────────────────────────────────────────────────
        ax = axes[1, col_idx]
        ax.plot(df["time_s"], etoh_raw, color=color, lw=0.5, alpha=0.3)
        ax.plot(df["time_s"], etoh,     color=color, lw=1.8, label="filtered")
        ax.set_ylim(0, etoh_ymax)
        ax.set_title(f"Alcohol (ETOH) — {label}")
        ax.set_ylabel("ETOH (ppm)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)

    fig.suptitle("Spec sensors — Sweat vs Blood", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    df_sweat = load(SWEAT_PATH)
    df_blood = load(BLOOD_PATH)
    plot_comparison(df_sweat, df_blood)


if __name__ == "__main__":
    main()
