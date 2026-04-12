import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from filter import (
    M_H2S, M_ETOH,
    H2S_OFFSET_PPM, ETOH_OFFSET_PPM,
    H2S_SCALE, VGAS_ETOH_OFFSET,
    detect_outliers_iqr, detect_outliers_roc,
    handle_outliers, lowpass_filter,
)

EXPERIMENT_PATH_SWEAT_1 = "20260325-experiment/sweat.csv"
EXPERIMENT_PATH_SWEAT_2 = "20260326-experiment/sweat.csv"
EXPERIMENT_PATH_SWEAT_3 = "20260331-experiment/sweat.csv"
EXPERIMENT_PATH_SWEAT_4 = "20260404-experiment/sweat.csv"
EXPERIMENT_PATH_BLOOD_1 = "20260331-experiment/1.5blood_1.csv"
EXPERIMENT_PATH_BLOOD_2 = "20260331-experiment/1.5blood_2.csv"
EXPERIMENT_PATH_BLOOD_3 = "20260331-experiment/1.5blood_3.csv"

EXPERIMENT_PATH_CLEAN = "20260325-experiment/zip_lock_clean.csv"
TRIM_START_S    = 0.0

MEMS_COLS = ["voc", "nh3", "hcho"]
MEMS_COLORS = ["#378ADD", "#1D9E75", "#D85A30"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def trim(df: pd.DataFrame, start_s: float = 0.0, end_s: float = None) -> pd.DataFrame:
    mask = df["time_s"] >= start_s
    if end_s is not None:
        mask &= df["time_s"] <= end_s
    return df[mask].reset_index(drop=True)


def smooth_sg(series: pd.Series, window: int = 21) -> pd.Series:
    wl = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    wl = max(wl, 5)
    return pd.Series(savgol_filter(series, window_length=wl, polyorder=2), index=series.index)


def process_h2s(df: pd.DataFrame, use_mean_vref: bool = True) -> pd.Series:
    vref = df["h2s_vref"].mean() if use_mean_vref else df["h2s_vref"]
    ppm  = ((df["h2s_vgas"] - vref) / M_H2S - H2S_OFFSET_PPM).clip(lower=0) * H2S_SCALE
    outliers = detect_outliers_iqr(ppm) | detect_outliers_roc(ppm)
    clean    = handle_outliers(ppm, outliers, method="interpolate")
    smooth   = savgol_filter(clean, window_length=51, polyorder=2)
    return lowpass_filter(pd.Series(smooth, index=df.index), cutoff_hz=0.005).clip(lower=0)


def process_etoh(df: pd.DataFrame, use_mean_vref: bool = True) -> pd.Series:
    vref = df["etoh_vref"].mean() if use_mean_vref else df["etoh_vref"]
    vgas = df["etoh_vgas"] - VGAS_ETOH_OFFSET
    ppm  = ((vgas - vref) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
    outliers = detect_outliers_iqr(ppm, 30) | detect_outliers_roc(ppm)
    clean    = handle_outliers(ppm, outliers, method="interpolate")
    return pd.Series(
        # savgol_filter(clean, window_length=51, polyorder=2),
        gaussian_filter1d(clean, sigma=80),
        index=df.index,
    ).clip(lower=0)


def plot_experiment(df: pd.DataFrame, path: str, minimal: bool = False, use_mean_vref: bool = True) -> None:
    h2s_smooth  = process_h2s(df, use_mean_vref=use_mean_vref)
    etoh_smooth = process_etoh(df, use_mean_vref=use_mean_vref)

    if minimal:
        # ── Minimal view: temp, humidity, H2S, ETOH (2x2) ────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.45, wspace=0.3)

        # Temp
        ax = axes[0, 0]
        ax.plot(df["time_s"], df["temp_C"], color="#E07B39", lw=1.3)
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature")

        # Humidity
        ax = axes[0, 1]
        ax.plot(df["time_s"], df["rh_pct"], color="#5B8DB8", lw=1.3)
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_title("Humidity")

        # H2S
        ax = axes[1, 0]
        vref    = df["h2s_vref"].mean() if use_mean_vref else df["h2s_vref"]
        h2s_raw = ((df["h2s_vgas"] - vref) / M_H2S - H2S_OFFSET_PPM).clip(lower=0) * H2S_SCALE
        ax.plot(df["time_s"], h2s_raw,    color="#cccccc", lw=0.6, alpha=0.7, label="raw ppm")
        ax.plot(df["time_s"], h2s_smooth, color="#378ADD", lw=1.5, label="filtered ppm")
        ax.set_ylabel("H2S (ppm)")
        ax.set_title("H2S spec sensor")
        ax.legend(fontsize=8)
        ax.set_ylim(0, h2s_smooth.max() * 1.1)

        # ETOH
        ax = axes[1, 1]
        vref_e   = df["etoh_vref"].mean() if use_mean_vref else df["etoh_vref"]
        vgas_e   = df["etoh_vgas"] - VGAS_ETOH_OFFSET
        etoh_raw = ((vgas_e - vref_e) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
        ax.plot(df["time_s"], etoh_raw,    color="#cccccc", lw=0.6, alpha=0.7, label="raw ppm")
        ax.plot(df["time_s"], etoh_smooth, color="#D85A30", lw=1.5, label="filtered ppm")
        ax.set_ylabel("ETOH (ppm)")
        ax.set_title("ETOH spec sensor")
        ax.legend(fontsize=8)
        ax.set_ylim(0, etoh_smooth.max() * 1.1)

        for ax in fig.axes:
            ax.set_xlabel("time (s)")

        fig.suptitle(f"Sensor array (minimal) — {path}", fontsize=12)
        plt.tight_layout()
        plt.show()
        return

    # Layout: 3 rows x 2 cols; bottom row spans both columns
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.3)

    # ── MEMS sensors (2x2) ────────────────────────────────────────────────────
    positions = [(0, 0), (0, 1), (1, 0)]
    for (row, col_idx), (col, color) in zip(positions, zip(MEMS_COLS, MEMS_COLORS)):
        ax = fig.add_subplot(gs[row, col_idx])
        smoothed = smooth_sg(df[col])
        ax.plot(df["time_s"], df[col],  color="#cccccc", lw=0.6, alpha=0.7, label="raw")
        ax.plot(df["time_s"], smoothed, color=color,     lw=1.3, label="smoothed")
        ax.set_ylabel(f"{col} (V)")
        ax.set_title(f"MEMS — {col}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 3.3)

    # ── H2S spec sensor (row 1, col 1) ───────────────────────────────────────
    ax_h2s = fig.add_subplot(gs[1, 1])
    vref    = df["h2s_vref"].mean() if use_mean_vref else df["h2s_vref"]
    h2s_raw = ((df["h2s_vgas"] - vref) / M_H2S - H2S_OFFSET_PPM).clip(lower=0) * H2S_SCALE
    ax_h2s.plot(df["time_s"], h2s_raw,    color="#cccccc", lw=0.6, alpha=0.7, label="raw ppm")
    ax_h2s.plot(df["time_s"], h2s_smooth, color="#378ADD", lw=1.5, label="filtered ppm")
    ax_h2s.set_ylabel("H2S (ppm)")
    ax_h2s.set_title("H2S spec sensor")
    ax_h2s.legend(fontsize=8)
    ax_h2s.set_ylim(0, h2s_smooth.max() * 1.1)

    # ── ETOH spec sensor (row 2, spans both columns) ──────────────────────────
    ax_etoh = fig.add_subplot(gs[2, :])
    vref_e   = df["etoh_vref"].mean() if use_mean_vref else df["etoh_vref"]
    vgas_e   = df["etoh_vgas"] - VGAS_ETOH_OFFSET
    etoh_raw = ((vgas_e - vref_e) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
    ax_etoh.plot(df["time_s"], etoh_raw,    color="#cccccc", lw=0.6, alpha=0.7, label="raw ppm")
    ax_etoh.plot(df["time_s"], etoh_smooth, color="#D85A30", lw=1.5, label="filtered ppm")
    ax_etoh.set_ylabel("ETOH (ppm)")
    ax_etoh.set_title("ETOH spec sensor")
    ax_etoh.legend(fontsize=8)
    ax_etoh.set_ylim(0, etoh_smooth.max() * 1.1)

    for ax in fig.axes:
        ax.set_xlabel("time (s)")

    fig.suptitle(f"Sensor array — {path}", fontsize=12)
    plt.tight_layout()
    plt.show()


def main(path: str = EXPERIMENT_PATH_SWEAT_4, minimal: bool = True, use_mean_vref: bool = True):
    df = load_data(path)
    df = trim(df, 2300, 3000)
    print(f"Loaded {len(df)} samples from {path}")
    plot_experiment(df, path, minimal=minimal, use_mean_vref=use_mean_vref)


if __name__ == "__main__":
    main()
