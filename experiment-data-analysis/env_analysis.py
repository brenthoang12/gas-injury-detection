"""
Temperature & humidity comparison — sweat vs blood.

Sweat : 20260325-experiment/sweat.csv
Blood : 20260326-experiment/1.5blood_sample_3.csv  (most data)
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SWEAT_PATH = "20260325-experiment/sweat.csv"
BLOOD_PATH = "20260326-experiment/1.5blood_sample_1.csv"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def smooth(series: pd.Series, window: int = 31) -> pd.Series:
    wl = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    wl = max(wl, 5)
    return pd.Series(savgol_filter(series, window_length=wl, polyorder=2), index=series.index)


def plot_env(df_sweat: pd.DataFrame, df_blood: pd.DataFrame) -> None:
    datasets = [
        (df_sweat, "Sweat (Mar 25)",    "#1D9E75"),
        (df_blood, "Blood #3 (Mar 26)", "#E84040"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)

    for col_idx, (df, title, color) in enumerate(datasets):
        ax_temp = axes[0, col_idx]
        ax_rh   = axes[1, col_idx]

        ax_temp.plot(df["time_s"], df["temp_C"], color="#cccccc", lw=0.6, alpha=0.7, label="raw")
        ax_temp.plot(df["time_s"], smooth(df["temp_C"]), color=color, lw=1.5, label="smoothed")
        ax_temp.set_title(f"{title} — Temperature")
        ax_temp.set_ylabel("Temp (°C)")
        ax_temp.set_xlabel("time (s)")
        ax_temp.legend(fontsize=8)

        ax_rh.plot(df["time_s"], df["rh_pct"], color="#cccccc", lw=0.6, alpha=0.7, label="raw")
        ax_rh.plot(df["time_s"], smooth(df["rh_pct"]), color=color, lw=1.5, label="smoothed")
        ax_rh.set_title(f"{title} — Humidity")
        ax_rh.set_ylabel("RH (%)")
        ax_rh.set_xlabel("time (s)")
        ax_rh.legend(fontsize=8)

    fig.suptitle("Temperature & Humidity — Sweat vs Blood", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    df_sweat = load(SWEAT_PATH)
    df_blood = load(BLOOD_PATH)
    plot_env(df_sweat, df_blood)


if __name__ == "__main__":
    main()
