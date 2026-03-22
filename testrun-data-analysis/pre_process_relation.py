import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ── Known concentration steps ─────────────────────────────────────────────────
H2S_STEPS = [
    {"label": "1.5 ppm H2S", "start_s": 0,    "end_s": 650,  "conc": 1.5},
    {"label": "1.0 ppm H2S", "start_s": 650,  "end_s": 1100, "conc": 1.0},
    {"label": "2.0 ppm H2S", "start_s": 1100, "end_s": None,  "conc": 2.0},
]

VOC_STEPS = [
    {"label": "2.0 ppm VOC", "start_s": 1400,    "end_s": 1900, "conc": 2.0},
    {"label": "7.0 ppm VOC", "start_s": 1900, "end_s": 2400, "conc": 7.0},
    {"label": "10.0 ppm VOC","start_s": 2400, "end_s": None,  "conc": 10.0},
]

MEMS_COLS = ["voc", "nh3", "hcho"]   # raw voltage columns


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df


def smooth(series: pd.Series, window: int = 21) -> pd.Series:
    """Light SG smoothing before ROC to reduce noise amplification."""
    return pd.Series(savgol_filter(series, window_length=window, polyorder=2), index=series.index)


def compute_roc(series: pd.Series, window: int = 10) -> pd.Series:
    """Rolling mean of absolute per-sample rate of change (V/s at 1Hz)."""
    return series.diff().abs().rolling(window=window, center=True, min_periods=1).mean()


def segment_roc(df: pd.DataFrame, steps: list, col: str, smooth_window: int = 21, roc_window: int = 10) -> list:
    """
    For each concentration step, compute mean and std ROC of a MEMS column.
    Returns list of dicts with label, conc, mean_roc, std_roc.
    """
    results = []
    smoothed = smooth(df[col], window=smooth_window)
    roc      = compute_roc(smoothed, window=roc_window)

    for step in steps:
        mask = df["time_s"] >= step["start_s"]
        if step["end_s"] is not None:
            mask &= df["time_s"] < step["end_s"]
        seg_roc = roc[mask]
        results.append({
            "label":    step["label"],
            "conc":     step["conc"],
            "mean_roc": seg_roc.mean(),
            "std_roc":  seg_roc.std(),
        })
    return results

def trimmed_data(df: pd.DataFrame, start_s: float = 0.0, end_s: float = None) -> pd.DataFrame:
    mask = df["time_s"] >= start_s
    if end_s is not None:
        mask &= df["time_s"] <= end_s
    return df[mask].reset_index(drop=True)

def shift_steps(steps: list, trim_s: float) -> list:
    """Adjust step start/end times after trimming."""
    shifted = []
    for step in steps:
        new_start = max(0.0, step["start_s"] - trim_s)
        new_end   = (step["end_s"] - trim_s) if step["end_s"] is not None else None
        if new_end is not None and new_end <= 0:
            continue  # step entirely trimmed away
        shifted.append({**step, "start_s": new_start, "end_s": new_end})
    return shifted

def plot_peak_response(df_h2s: pd.DataFrame, df_voc: pd.DataFrame,
                       steps_h2s: list, steps_voc: list) -> None:
    """
    For the 2.0 ppm step in both H2S and VOC runs, compute
    peak response = max(smoothed) - start(smoothed) for each MEMS sensor.
    """
    target_h2s = next(s for s in steps_h2s if s["conc"] == 1.5)  # 1.5 ppm H2S step
    target_voc = next(s for s in steps_voc if s["conc"] == 2.0)  # 2.0 ppm VOC step

    results = []
    for col in MEMS_COLS:
        for label, df, step in [("H2S 1.5ppm", df_h2s, target_h2s),
                                 ("VOC 2.0ppm", df_voc, target_voc)]:
            mask = df["time_s"] >= step["start_s"]
            if step["end_s"] is not None:
                mask &= df["time_s"] < step["end_s"]

            seg       = smooth(df.loc[mask, col])
            start_val = seg.iloc[0]
            max_val   = seg.max()
            delta     = max_val - start_val

            results.append({"sensor": col, "condition": label, "delta_V": delta})

    results_df = pd.DataFrame(results)

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(MEMS_COLS))
    width  = 0.35

    h2s_vals = results_df[results_df["condition"] == "H2S 1.5ppm"]["delta_V"].values
    voc_vals = results_df[results_df["condition"] == "VOC 2.0ppm"]["delta_V"].values

    ax.bar(x - width / 2, h2s_vals, width, label="H2S 1.5 ppm", color="#378ADD", alpha=0.85)
    ax.bar(x + width / 2, voc_vals, width, label="VOC 2.0 ppm", color="#D85A30", alpha=0.85)

    for idx, (h, v) in enumerate(zip(h2s_vals, voc_vals)):
        ax.text(idx - width / 2, h + 0.002, f"{h:.3f}V", ha="center", fontsize=9)
        ax.text(idx + width / 2, v + 0.002, f"{v:.3f}V", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MEMS_COLS)
    ax.set_ylabel("peak response ΔV (max − start)")
    ax.set_title("MEMS peak response — H2S 1.5 ppm vs VOC 2.0 ppm")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Print summary ──
    print("\n── Peak response summary ─────────────────────────────")
    print(f"  {'Sensor':<8} {'H2S 1.5ppm':>12}  {'VOC 2.0ppm':>12}  {'ratio V/H':>10}")
    for col, h, v in zip(MEMS_COLS, h2s_vals, voc_vals):
        ratio = v / h if h != 0 else float("inf")
        print(f"  {col:<8} {h:>12.4f}V  {v:>12.4f}V  {ratio:>10.2f}x")
    print("──────────────────────────────────────────────────────")

# ── Main analysis ─────────────────────────────────────────────────────────────
def analyse_cross_sensitivity(path_h2s: str, path_voc: str,
                               trim_h2s_s: float = 0.0,
                               trim_voc_s: float = 0.0) -> None:
    df_h2s = load_data(path_h2s)
    df_voc = load_data(path_voc)

    if trim_h2s_s > 0:
        df_h2s = trimmed_data(df_h2s, start_s=trim_h2s_s)
        df_h2s["time_s"] = df_h2s["time_s"] - df_h2s["time_s"].iloc[0]

    if trim_voc_s > 0:
        df_voc = trimmed_data(df_voc, start_s=trim_voc_s)
        df_voc["time_s"] = df_voc["time_s"] - df_voc["time_s"].iloc[0]

    # Shift step boundaries to match trimmed time
    steps_h2s = shift_steps(H2S_STEPS, trim_h2s_s)
    steps_voc = shift_steps(VOC_STEPS,  trim_voc_s)

    plot_peak_response(df_h2s, df_voc, steps_h2s, steps_voc)

    fig_ts, axes_ts = plt.subplots(len(MEMS_COLS), 2, figsize=(14, 4 * len(MEMS_COLS)))

    colors_h2s = ["#378ADD", "#1D9E75", "#D85A30"]
    colors_voc = ["#BA7517", "#378ADD", "#7F77DD"]

    for i, col in enumerate(MEMS_COLS):
        for j, (ax, df, steps, gas, colors) in enumerate([
            (axes_ts[i][0], df_h2s, steps_h2s, "H2S", colors_h2s),
            (axes_ts[i][1], df_voc, steps_voc,  "VOC", colors_voc),
        ]):
            smoothed = smooth(df[col])
            ax.plot(df["time_s"], df[col], color="#cccccc", lw=0.5, alpha=0.6, label="raw")
            ax.plot(df["time_s"], smoothed, color="#222222", lw=1.2, label="smoothed")

            t_max = df["time_s"].max()
            for step, color in zip(steps, colors):
                end = step["end_s"] if step["end_s"] is not None else t_max
                ax.axvspan(step["start_s"], end, alpha=0.12, color=color, label=step["label"])
                ax.axvline(step["start_s"], color=color, lw=0.8, ls="--")

            ax.set_title(f"{col} — {gas} exposure")
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"{col} (V)")
            ax.legend(fontsize=7)
            ax.set_ylim(0, 3.1) 

    fig_ts.suptitle("MEMS sensor response — H2S vs VOC exposure", fontsize=12)
    fig_ts.tight_layout()
    plt.show()


def main():
    analyse_cross_sensitivity(
        path_h2s="testrun-h2s-st-2/readings_20260312_144332.csv",
        path_voc="testrun-voc-st/readings_20260312_151908.csv",
        trim_h2s_s=0,
        trim_voc_s=1400,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()