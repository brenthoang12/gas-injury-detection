import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, butter, filtfilt

H2S_SENSITIVITY_CODE = 216.09
H2S_TIA_GAIN = 49.9   

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

def detect_outliers_roc() -> bool:
    return False

def detect_outliers_zscore(series: pd.Series, window: int = 20, threshold: float = 3.0) -> pd.Series:
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


def handle_outliers(series: pd.Series, outlier_mask: pd.Series, 
                    method: str = "interpolate", max_gap: int = 30) -> pd.Series:
    """
    Replace outliers without discarding them.
    method="interpolate" : linear interpolation between clean neighbours
    method="ffill"       : carry forward the last clean reading (flat prediction)
    """
    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan

    if method == "interpolate":
        cleaned = cleaned.interpolate(method="linear", limit=max_gap, limit_direction="both")
    elif method == "ffill":
        cleaned = cleaned.ffill(limit=max_gap).bfill(limit=max_gap)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'interpolate' or 'ffill'.")

    return cleaned

def trimmed_data(df: pd.DataFrame, start_s: float = 0.0, end_s: float = None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    mask &= df["time_s"] >= start_s
    if end_s is not None:
        mask &= df["time_s"] <= end_s
    df = df[mask].reset_index(drop=True)  
    return df


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


def main():
    path = "testrun-h2s-st-2/readings_20260312_144332.csv"
    df = pd.read_csv(path)
    # df2 = pd.read_csv("testrun-h2s-st/readings_20260312_120603.csv")


    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()

    df = trimmed_data(df, 100)
    df = df.drop(columns=['h2s_ppm'])

    min_Vref = df['h2s_vref'].min()
    max_Vref = df['h2s_vref'].max()
    print(max_Vref)
    print(min_Vref)
    dif = (max_Vref - min_Vref) / max_Vref
    average_mm = (min_Vref + max_Vref) / 2
    average_true = df['h2s_vref'].sum()/len(df)
    print(average_mm)
    print(average_true)

    df['h2s_vref'] = average_true

    M_H2S = H2S_SENSITIVITY_CODE * H2S_TIA_GAIN * 1e-6

    df['h2s_ppm'] = (df['h2s_vgas'] - df['h2s_vref']) / M_H2S

    df['h2s_outlier_iqr'] = detect_outliers_iqr(df["h2s_ppm"])
    df['h2s_clean_iqr'] = handle_outliers(df["h2s_ppm"], df["h2s_outlier_iqr"], method="interpolate")
    df['h2s_sf_iqr'] = savgol_filter(df["h2s_clean_iqr"], window_length=51, polyorder=2)

    df['h2s_lp'] = lowpass_filter(df['h2s_clean_iqr'], cutoff_hz=0.01)

    graph_features(df, ["h2s_sf_iqr", "h2s_lp"])

    


if __name__ == "__main__":
    main()