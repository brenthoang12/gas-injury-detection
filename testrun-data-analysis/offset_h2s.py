import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
from scipy.signal import savgol_filter

H2S_SENSITIVITY_CODE = 216.09   # nA/ppm  (from sensor)
H2S_TIA_GAIN         = 49.9     # kΩ
M_H2S                = H2S_SENSITIVITY_CODE * H2S_TIA_GAIN * 1e-6

CLEAN_AIR_PATH = "testrun-h2s-st/readings_20260312_114657.csv"
IQR_K          = 1.5    

def load(path: str) -> pd.DataFrame:
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

def iqr_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask: True = clean sample, False = outlier."""
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series >= Q1 - k * IQR) & (series <= Q3 + k * IQR)
 
 
def compute_h2s_ppm(df: pd.DataFrame) -> pd.Series:
    vref = df["h2s_vref"].mean()           # fixed Vref from session average
    return ((df["h2s_vgas"] - vref) / M_H2S).clip(lower=0)


def evaluate_offset(path: str = CLEAN_AIR_PATH,
                    warm_up_s: float = 0.0,
                    iqr_k: float = IQR_K) -> float:
 
    # 1. Load & trim warm-up
    df = load(path)
    df = trim(df, start_s=warm_up_s)
    print(f"Loaded {len(df)} samples after {warm_up_s}s warm-up trim")
 
    # 2. Recompute ppm from voltages
    df["h2s_ppm_raw"] = compute_h2s_ppm(df)
 
    # 3. Reject outliers
    clean_mask         = iqr_mask(df["h2s_ppm_raw"], k=iqr_k)
    df["h2s_clean"]    = df["h2s_ppm_raw"].where(clean_mask)
    n_outliers         = (~clean_mask).sum()
    print(f"Outliers rejected: {n_outliers} / {len(df)}  ({100*n_outliers/len(df):.1f}%)")
 
    # 4. Compute offset candidates
    clean_vals = df["h2s_clean"].dropna()
    median_val = clean_vals.median()
    mean_val   = clean_vals.mean()
    std_val    = clean_vals.std()
    p5, p95    = clean_vals.quantile(0.05), clean_vals.quantile(0.95)
 
    print("\n── Offset candidates ─────────────────────────────────")
    print(f"  Median  (recommended) : {median_val:.4f} ppm")
    print(f"  Mean                  : {mean_val:.4f} ppm")
    print(f"  Std dev               : {std_val:.4f} ppm")
    print(f"  5th–95th percentile   : {p5:.4f} – {p95:.4f} ppm")
    print(f"  Samples used          : {len(clean_vals)}")
    print("──────────────────────────────────────────────────────")
    print(f"\n  → Paste into your code:")
    print(f"  H2S_OFFSET_PPM = {median_val:.4f}  # median of clean-air baseline")
 
    # 5. Plot
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
 
    # ── Top left: raw ppm over time with offset line ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["time_s"], df["h2s_ppm_raw"], color="#378ADD", lw=0.8, alpha=0.7, label="h2s_ppm (raw)")
    ax1.scatter(df.loc[~clean_mask, "time_s"], df.loc[~clean_mask, "h2s_ppm_raw"],
                color="#D85A30", s=15, zorder=5, label="outliers (IQR)")
    ax1.axhline(median_val, color="#D85A30", lw=1.5, ls="--", label=f"offset median = {median_val:.2f}")
    ax1.axhline(mean_val,   color="#639922", lw=1.0, ls=":",  label=f"mean = {mean_val:.2f}")
    ax1.axhspan(p5, p95, alpha=0.08, color="#378ADD", label="5–95th pct band")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("h2s_ppm")
    ax1.set_title("clean-air baseline — raw ppm with offset candidates")
    ax1.legend(fontsize=8)
 
    # ── Bottom left: histogram ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(clean_vals, bins=30, color="#378ADD", alpha=0.75, edgecolor="white", lw=0.4)
    ax2.axvline(median_val, color="#D85A30", lw=1.5, ls="--", label=f"median {median_val:.2f}")
    ax2.axvline(mean_val,   color="#639922", lw=1.0, ls=":",  label=f"mean {mean_val:.2f}")
    ax2.set_xlabel("h2s_ppm")
    ax2.set_ylabel("count")
    ax2.set_title("distribution of clean samples")
    ax2.legend(fontsize=8)
 
    # ── Bottom right: after offset applied ──
    ax3 = fig.add_subplot(gs[1, 1])
    corrected = (df["h2s_ppm_raw"] - median_val).clip(lower=0)
    ax3.plot(df["time_s"], corrected, color="#1D9E75", lw=0.8)
    ax3.axhline(0, color="#888780", lw=0.8, ls="--")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("h2s_ppm (offset applied)")
    ax3.set_title(f"after offset subtraction (offset = {median_val:.2f})")
 
    plt.suptitle("H2S offset evaluation — clean air recording", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()
 
    return median_val

if __name__ == "__main__":
    offset = evaluate_offset(
        path      = CLEAN_AIR_PATH,
        warm_up_s = 0,
        iqr_k     = IQR_K,
    )
