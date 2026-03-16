import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

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

def get_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Add time (s)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()

    # Outlier detection 
    df["h2s_outlier_z"]   = detect_outliers_zscore(df["h2s_ppm"])
    df["h2s_outlier_iqr"] = detect_outliers_iqr(df["h2s_ppm"])

    # Handle outlier
    df["h2s_clean_iqr"] = handle_outliers(df["h2s_ppm"], df["h2s_outlier_iqr"], method="interpolate")
    df["h2s_outlier_clean"] =  detect_outliers_iqr(df["h2s_clean_iqr"])

    # Smooth readings
    df["h2s_sf_iqr"] = savgol_filter(df["h2s_clean_iqr"], window_length=51, polyorder=2)

    return df

def get_trimmed_data(path: str, start_s: float = 0.0, end_s: float = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Add time (s)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()

    # Time filter
    mask = pd.Series(True, index=df.index)
    mask &= df["time_s"] >= start_s
    if end_s is not None:
        mask &= df["time_s"] <= end_s
    df = df[mask].reset_index(drop=True)  

    # Outlier detection
    df["h2s_outlier_z"]   = detect_outliers_zscore(df["h2s_ppm"])
    df["h2s_outlier_iqr"] = detect_outliers_iqr(df["h2s_ppm"])

    # Handle outliers
    df["h2s_clean_iqr"] = handle_outliers(df["h2s_ppm"], df["h2s_outlier_iqr"], method="interpolate")
    df["h2s_outlier_clean"] = detect_outliers_iqr(df["h2s_clean_iqr"])

    # Smooth readings
    df["h2s_sf_iqr"] = savgol_filter(df["h2s_clean_iqr"], window_length=51, polyorder=2)

    return df  


def graph_outlier_flag(data: pd.DataFrame, outlier_col: str) -> None:
    """Plot a binary yes/no outlier flag over time."""
    plt.figure()
    plt.plot(data["time_s"], data[outlier_col].astype(int), drawstyle="steps-post", linewidth=1)
    plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Time (s)")
    plt.ylabel("Outlier")
    plt.title(f"Outlier flag: {outlier_col}")
    plt.tight_layout()
    plt.show()

def main():
    path = "testrun-h2s-st-2/readings_20260312_144332.csv"
    df = get_trimmed_data(path, 0, 650)

    # print(df.head())

    for col in ["h2s_outlier_z", "h2s_outlier_iqr"]:
        count = df[col].sum()
        total = len(df)
        print(f"{col}: {count} outliers / {total} total ({100 * count / total:.1f}%)")
    
    graph_outlier_flag(df, "h2s_clean_iqr")
    graph_feature(df, "h2s_sf_iqr")

if __name__ == "__main__":
    main()
