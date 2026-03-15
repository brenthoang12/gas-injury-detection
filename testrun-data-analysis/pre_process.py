import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

def graph_feature(data: pd.DataFrame, col: str) -> None:
    plt.figure()
    plt.plot(data["time_s"], data[col])
    plt.xlabel("Time (s)")
    plt.ylabel(col)
    plt.title(col)
    plt.tight_layout()
    plt.show()

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

def handle_outliers(series: pd.Series, outlier_mask: pd.Series, method: str = "interpolate") -> pd.Series:
    """
    Replace outliers without discarding them.
    method="interpolate" : linear interpolation between clean neighbours
    method="ffill"       : carry forward the last clean reading (flat prediction)
    """
    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan

    if method == "interpolate":
        cleaned = cleaned.interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        cleaned = cleaned.ffill().bfill()  # bfill handles leading NaNs
    else:
        raise ValueError(f"Unknown method: {method}. Use 'interpolate' or 'ffill'.")

    return cleaned

def get_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Add time (s)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()

    # Smoothen H2S
    df["h2s_ra"] = df["h2s_ppm"].rolling(window=10, center=True).mean() # Rolling average
    df["h2s_sf"] = savgol_filter(df["h2s_ppm"], window_length=11, polyorder=2) # Savitzky-Golay Filter
    return df

def main():
    path = "testrun-h2s-st/readings_20260312_120603.csv"
    df = get_data(path)
    graph_feature(df, "h2s_sf")


if __name__ == "__main__":
    main()
