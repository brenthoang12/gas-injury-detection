import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, butter, filtfilt

H2S_SENSITIVITY_CODE = 216.09
ETOH_SENSITIVITY_CODE = 21.5

H2S_TIA_GAIN = 49.9  
ETOH_TIA_GAIN = 249.0

M_H2S = H2S_SENSITIVITY_CODE * H2S_TIA_GAIN * 1e-6
M_ETOH = ETOH_SENSITIVITY_CODE * ETOH_TIA_GAIN * 1e-6

H2S_OFFSET_PPM = 15.9215 # from offset_h2s.py
ETOH_OFFSET_PPM = 19.5510 # from offset_etoh.py

TRUE_PPM_H2S = 1.5   
MEASURED_H2S   = 1.0   
H2S_SCALE  = TRUE_PPM_H2S / MEASURED_H2S  

def graph_feature(data: pd.DataFrame, col: str, outlier_col: str = None) -> None:
    plt.figure()
    plt.plot(data["time_s"], data[col], label=col)
    if outlier_col and outlier_col in data.columns:
        outliers = data[data[outlier_col]]
        plt.scatter(outliers["time_s"], outliers[col], color="red", s=20, zorder=5, label="outliers")
    plt.xlabel("Time (s)")
    plt.ylabel(col)
    plt.title(col)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_features(data: pd.DataFrame, cols: list[str], outlier_cols: list[str] = None,
                   ncols: int = 2) -> None:
    """
    Plot multiple columns in a single window as subplots.
    cols: list of column names to plot
    outlier_cols: optional list of boolean columns (same length as cols) marking outliers per plot
    ncols: number of subplot columns in the grid
    """
    n = len(cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(data["time_s"], data[col], label=col)
        if outlier_cols and i < len(outlier_cols) and outlier_cols[i] and outlier_cols[i] in data.columns:
            outliers = data[data[outlier_cols[i]]]
            ax.scatter(outliers["time_s"], outliers[col], color="red", s=20, zorder=5, label="outliers")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def detect_outliers_zscore(series: pd.Series, window: int = 20, threshold: float = 2.5) -> pd.Series:
    """
    Rolling Z-score outlier detection.
    Returns boolean Series: True = outlier.
    """
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std  = series.rolling(window=window, center=True, min_periods=1).std()
    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z_score.abs() > threshold


def detect_outliers_iqr(series: pd.Series, window: int = 20, k: float = 1.5) -> pd.Series:
    """
    Rolling IQR outlier detection.
    Returns boolean Series: True = outlier.
    k=1.5 is standard; raise to 2.0-3.0 to be less aggressive.
    """
    Q1 = series.rolling(window=window, center=True, min_periods=1).quantile(0.25)
    Q3 = series.rolling(window=window, center=True, min_periods=1).quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)


def detect_outliers_roc(series: pd.Series, window: int = 20, threshold: float = None, z_thresh: float = 2.0) -> pd.Series:
    """
    Rate-of-change outlier detection.
    Flags samples where the change from the previous sample is abnormally large.

    window    : rolling window to compute expected rate of change
    threshold : fixed max allowed change per sample (ppm/s). If None, derived automatically.
    z_thresh  : if threshold=None, flags points where ROC exceeds z_thresh std devs from rolling mean ROC
    """
    roc = series.diff().abs()  # absolute change between consecutive samples

    if threshold is not None:
        # fixed threshold — flag anything changing faster than this per sample
        return roc > threshold
    else:
        # adaptive threshold — flag where ROC is unusually large relative to local behaviour
        rolling_mean_roc = roc.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std_roc  = roc.rolling(window=window, center=True, min_periods=1).std()
        z_score = (roc - rolling_mean_roc) / rolling_std_roc.replace(0, np.nan)
        return z_score > z_thresh


def handle_outliers(series: pd.Series, outlier_mask: pd.Series,
                    method: str = "interpolate", max_gap: int = 30,
                    roc_window: int = 5) -> pd.Series:
    """
    method="interpolate" : linear interpolation between clean neighbours
    method="ffill"       : carry forward last clean reading
    method="roc"         : extrapolate using recent rate of change
    roc_window           : how many clean samples to average slope from
    """
    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan

    if method == "interpolate":
        cleaned = cleaned.interpolate(method="linear", limit=max_gap, limit_direction="both")

    elif method == "ffill":
        cleaned = cleaned.ffill(limit=max_gap).bfill(limit=max_gap)

    elif method == "roc":
        values = cleaned.to_numpy(dtype=float)
        for i in range(len(values)):
            if not np.isnan(values[i]):
                continue

            # find clean samples before this gap
            before_idx = [j for j in range(i - 1, -1, -1) if not np.isnan(values[j])]
            if len(before_idx) < 2:
                continue  # not enough history, skip

            # compute average slope from last roc_window clean samples
            anchor_indices = before_idx[:roc_window]
            slopes = [
                values[anchor_indices[k]] - values[anchor_indices[k + 1]]
                for k in range(len(anchor_indices) - 1)
            ]
            avg_slope = np.mean(slopes)

            # fill forward up to max_gap
            gap_count = 0
            j = i
            while j < len(values) and np.isnan(values[j]) and gap_count < max_gap:
                steps_from_anchor = j - before_idx[0]
                values[j] = values[before_idx[0]] + avg_slope * steps_from_anchor
                gap_count += 1
                j += 1

        cleaned = pd.Series(values, index=series.index)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'interpolate', 'ffill', or 'roc'.")

    return cleaned


def trimmed_data(df: pd.DataFrame, start_s: float = 0.0, end_s: float = None) -> pd.DataFrame:
    mask = df["time_s"] >= start_s
    if end_s is not None:
        mask &= df["time_s"] <= end_s
    return df[mask].reset_index(drop=True)


def lowpass_filter(series: pd.Series, cutoff_hz: float = 0.009, fs: float = 1.0, order: int = 4) -> pd.Series:
    """
    Butterworth low-pass filter.
    cutoff_hz: frequencies above this are attenuated
    fs: sampling frequency in Hz (your system = 1 Hz)
    order: filter sharpness — higher = steeper cutoff
    """
    nyquist = fs / 2
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return pd.Series(filtfilt(b, a, series), index=series.index)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    return df

def main():
    # ---- Load data
    path_h2s_0 = "testrun-h2s-st-2/readings_20260312_144332.csv"
    path_h2s_1 = "testrun-h2s-st/readings_20260312_120603.csv"
    path_voc = "testrun-voc-st/readings_20260312_151908.csv"
    df = load_data(path_voc)
    # df = trimmed_data(df, 1400)

    # ---- Clean H2S data
    print(f"Vref  min={df['h2s_vref'].min()}  max={df['h2s_vref'].max()}  mean={df['h2s_vref'].mean()}")
    df['h2s_vref'] = df['h2s_vref'].mean()
    df['h2s_ppm'] = ((df['h2s_vgas'] - df['h2s_vref']) / M_H2S - H2S_OFFSET_PPM).clip(lower=0)
    df['h2s_ppm'] = df['h2s_ppm'] * H2S_SCALE

    df['h2s_outlier_roc'] = detect_outliers_roc(df["h2s_ppm"])
    print(f"ROC  outliers: {df['h2s_outlier_roc'].sum()} / {len(df)} ({100 * df['h2s_outlier_roc'].sum() / len(df):.1f}%)")
    df['h2s_clean_roc'] = handle_outliers(df["h2s_ppm"], df["h2s_outlier_roc"], method="interpolate")
    df['h2s_sf_roc'] = savgol_filter(df["h2s_clean_roc"], window_length=51, polyorder=2)
    
    df['h2s_outlier_iqr'] = detect_outliers_iqr(df["h2s_ppm"])
    print(f"IQR  outliers: {df['h2s_outlier_iqr'].sum()} / {len(df)} ({100 * df['h2s_outlier_iqr'].sum() / len(df):.1f}%)")
    df['h2s_clean_iqr'] = handle_outliers(df["h2s_ppm"], df["h2s_outlier_iqr"], method="interpolate")
    df['h2s_sf_iqr'] = savgol_filter(df["h2s_clean_iqr"], window_length=51, polyorder=2)

    df['h2s_lp'] = lowpass_filter(df['h2s_sf_iqr'], cutoff_hz=0.005)

    # ---- Clean ETOH data
    print(f"Vref  min={df['etoh_vref'].min()}  max={df['etoh_vref'].max()}  mean={df['etoh_vref'].mean()}")
    df['etoh_vref'] = df['etoh_vref'].mean()
    df['etoh_ppm'] = ((df['etoh_vgas'] - df['etoh_vref']) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
    
    df['etoh_outlier_iqr'] = detect_outliers_iqr(df["etoh_ppm"])
    df['etoh_clean_iqr'] = handle_outliers(df["etoh_ppm"], df["etoh_outlier_iqr"], method="interpolate")
    df['etoh_sf_iqr'] = savgol_filter(df["etoh_clean_iqr"], window_length=51, polyorder=2)

    df['etoh_lp'] = lowpass_filter(df['etoh_sf_iqr'], cutoff_hz=0.005)

    # ---- Graph
    # graph_feature(df, "h2s_lp", 'h2s_outlier_roc')
    # graph_features(df, ["h2s_sf_iqr", "h2s_lp"])
    graph_features(df, ["etoh_sf_iqr", "etoh_lp"])


if __name__ == "__main__":
    main()
