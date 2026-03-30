#   ––––– H2S
#   Median  (recommended) : 23.6041 ppm
#   Mean                  : 21.1374 ppm
#   Median  (recommended) : 19.0479 ppm
#   Mean                  : 20.6634 ppm
#   Median  (recommended) : 21.1419 ppm
#   Mean                  : 21.1453 ppm
#   ––––– EtOH
#   Median  (recommended) : 13.4562 ppm
#   Mean                  : 16.3479 ppm
#   Median  (recommended) : 26.8528 ppm
#   Mean                  : 27.4800 ppm
#   Median  (recommended) : 29.0403 ppm
#   Mean                  : 29.0571 ppm

from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


IQR_K = 1.5
CLEAN_AIR_PATH_0 = "20260326-experiment/zip_lock_clean_2.csv"
CLEAN_AIR_PATH_1 = "20260325-experiment/zip_lock_clean.csv"
CLEAN_AIR_PATH_2 = "20260324-experiment/zip_lock_clean.csv"


@dataclass
class SensorConfig:
    name: str            # e.g. "ETOH" or "H2S"
    sensitivity: float   # nA/ppm
    tia_gain: float      # kΩ
    vgas_col: str        # DataFrame column for Vgas
    vref_col: str        # DataFrame column for Vref
    scale: float = 1.0        # calibration scale (true_ppm / measured_ppm)
    vgas_offset: float = 0.0  # voltage offset subtracted from Vgas before ppm conversion

    @property
    def m(self) -> float:
        return self.sensitivity * self.tia_gain * 1e-6

    @property
    def ppm_col(self) -> str:
        return f"{self.name.lower()}_ppm_raw"

    @property
    def clean_col(self) -> str:
        return f"{self.name.lower()}_clean"


VGAS_ETOH_OFFSET = -0.1838

ETOH = SensorConfig(
    name="ETOH",
    sensitivity=21.5,
    tia_gain=249.0,
    vgas_col="etoh_vgas",
    vref_col="etoh_vref",
    vgas_offset=VGAS_ETOH_OFFSET,
)

H2S = SensorConfig(
    name="H2S",
    sensitivity=216.09,
    tia_gain=49.9,
    vgas_col="h2s_vgas",
    vref_col="h2s_vref",
    scale=1.5 / 1.0,  # TRUE_PPM_H2S / MEASURED_H2S
)



def iqr_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask: True = clean sample, False = outlier."""
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series >= Q1 - k * IQR) & (series <= Q3 + k * IQR)


def compute_ppm(df: pd.DataFrame, cfg: SensorConfig) -> pd.Series:
    vref = df[cfg.vref_col].mean()
    return (((df[cfg.vgas_col] - cfg.vgas_offset - vref) / cfg.m) * cfg.scale).clip(lower=0)


def _print_offset_summary(cfg: SensorConfig, n_total: int, n_outliers: int,
                          clean_vals: pd.Series, median_val: float,
                          mean_val: float, std_val: float,
                          p5: float, p95: float) -> None:
    print(f"[{cfg.name}] Outliers rejected: {n_outliers} / {n_total}  ({100*n_outliers/n_total:.1f}%)")
    print("\n── Offset candidates ─────────────────────────────────")
    print(f"  Median  (recommended) : {median_val:.4f} ppm")
    print(f"  Mean                  : {mean_val:.4f} ppm")
    print(f"  Std dev               : {std_val:.4f} ppm")
    print(f"  5th–95th percentile   : {p5:.4f} – {p95:.4f} ppm")
    print(f"  Samples used          : {len(clean_vals)}")
    print("──────────────────────────────────────────────────────")
    print(f"\n  → Paste into your code:")
    print(f"  {cfg.name}_OFFSET_PPM = {median_val:.4f}  # median of clean-air baseline")


def _plot_offset(cfg: SensorConfig, df: pd.DataFrame, clean_mask: pd.Series,
                 clean_vals: pd.Series, median_val: float,
                 mean_val: float, p5: float, p95: float) -> None:
    tag = cfg.name.lower()
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["time_s"], df[cfg.ppm_col], color="#378ADD", lw=0.8, alpha=0.7, label=f"{tag}_ppm (raw)")
    ax1.scatter(df.loc[~clean_mask, "time_s"], df.loc[~clean_mask, cfg.ppm_col],
                color="#D85A30", s=15, zorder=5, label="outliers (IQR)")
    ax1.axhline(median_val, color="#D85A30", lw=1.5, ls="--", label=f"offset median = {median_val:.2f}")
    ax1.axhline(mean_val,   color="#639922", lw=1.0, ls=":",  label=f"mean = {mean_val:.2f}")
    ax1.axhspan(p5, p95, alpha=0.08, color="#378ADD", label="5–95th pct band")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(f"{tag}_ppm")
    ax1.set_title("clean-air baseline — raw ppm with offset candidates")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(clean_vals, bins=30, color="#378ADD", alpha=0.75, edgecolor="white", lw=0.4)
    ax2.axvline(median_val, color="#D85A30", lw=1.5, ls="--", label=f"median {median_val:.2f}")
    ax2.axvline(mean_val,   color="#639922", lw=1.0, ls=":",  label=f"mean {mean_val:.2f}")
    ax2.set_xlabel(f"{tag}_ppm")
    ax2.set_ylabel("count")
    ax2.set_title("distribution of clean samples")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    corrected = (df[cfg.ppm_col] - median_val).clip(lower=0)
    ax3.plot(df["time_s"], corrected, color="#1D9E75", lw=0.8)
    ax3.axhline(0, color="#888780", lw=0.8, ls="--")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel(f"{tag}_ppm (offset applied)")
    ax3.set_title(f"after offset subtraction (offset = {median_val:.2f})")

    plt.suptitle(f"{cfg.name} offset evaluation — clean air recording", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def evaluate_offset(cfg: SensorConfig,
                    df: pd.DataFrame,
                    iqr_k: float = IQR_K) -> float:

    print(f"[{cfg.name}] Evaluating offset on {len(df)} samples")

    df = df.copy()
    df[cfg.ppm_col]   = compute_ppm(df, cfg)
    clean_mask        = iqr_mask(df[cfg.ppm_col], k=iqr_k)
    df[cfg.clean_col] = df[cfg.ppm_col].where(clean_mask)

    clean_vals = df[cfg.clean_col].dropna()
    median_val = clean_vals.median()
    mean_val   = clean_vals.mean()
    std_val    = clean_vals.std()
    p5, p95    = clean_vals.quantile(0.05), clean_vals.quantile(0.95)

    _print_offset_summary(cfg, len(df), (~clean_mask).sum(),
                          clean_vals, median_val, mean_val, std_val, p5, p95)
    _plot_offset(cfg, df, clean_mask, clean_vals, median_val, mean_val, p5, p95)

    return median_val


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


def main():
    df = load(CLEAN_AIR_PATH_1)
    df = trim(df, start_s=200)
    print(f"Loaded {len(df)} samples after {0}s warm-up trim")

    etoh_offset = evaluate_offset(H2S, df)

if __name__ == "__main__":
    main()