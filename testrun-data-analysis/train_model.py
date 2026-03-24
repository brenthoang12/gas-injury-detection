import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from river import ensemble, metrics, preprocessing

from processing import (
    load_data,
    trimmed_data,
    detect_outliers_iqr,
    detect_outliers_roc,
    handle_outliers,
    lowpass_filter,
    M_H2S,
    M_ETOH,
    H2S_OFFSET_PPM,
    ETOH_OFFSET_PPM,
    H2S_SCALE,
)

from scipy.signal import savgol_filter

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_VOC = 0
LABEL_H2S = 1
LABEL_NAMES = {0: "VOC", 1: "H2S"}

# ── Stable windows (skip settling and clearing transitions) ───────────────────
SETTLE_S = 60   # seconds to skip at start of each concentration step
CLEAR_S  = 60   # seconds to skip at end of each concentration step

H2S_STABLE_WINDOWS = [
    {"start_s": 200  + SETTLE_S, "end_s": 650  - CLEAR_S, "label": LABEL_H2S, "conc": 1.5},
    {"start_s": 650  + SETTLE_S, "end_s": 1100 - CLEAR_S, "label": LABEL_H2S, "conc": 1.0},
    {"start_s": 1100 + SETTLE_S, "end_s": 1500 - CLEAR_S, "label": LABEL_H2S, "conc": 2.0},
]

VOC_STABLE_WINDOWS = [
    {"start_s": 0    + SETTLE_S, "end_s": 500  - CLEAR_S,  "label": LABEL_VOC, "conc": 2.0},
    {"start_s": 500  + SETTLE_S, "end_s": 1000 - CLEAR_S,  "label": LABEL_VOC, "conc": 7.0},
    {"start_s": 1000 + SETTLE_S, "end_s": None,             "label": LABEL_VOC, "conc": 10.0},
]

# ── Feature config ────────────────────────────────────────────────────────────
ROC_WINDOW    = 10   # rolling window for ROC features
ROLLING_WINDOW = 30  # rolling window for mean/std features


# ── Signal cleaning ───────────────────────────────────────────────────────────
def clean_h2s(df: pd.DataFrame) -> pd.DataFrame:
    df['h2s_vref'] = df['h2s_vref'].mean()
    df['h2s_ppm']  = ((df['h2s_vgas'] - df['h2s_vref']) / M_H2S - H2S_OFFSET_PPM).clip(lower=0)
    df['h2s_ppm']  = df['h2s_ppm'] * H2S_SCALE
    mask           = detect_outliers_iqr(df['h2s_ppm'])
    df['h2s_ppm']  = handle_outliers(df['h2s_ppm'], mask, method="interpolate")
    df['h2s_ppm']  = savgol_filter(df['h2s_ppm'], window_length=51, polyorder=2)
    df['h2s_ppm']  = lowpass_filter(df['h2s_ppm'], cutoff_hz=0.005)
    return df


def clean_etoh(df: pd.DataFrame) -> pd.DataFrame:
    df['etoh_vref'] = df['etoh_vref'].mean()
    df['etoh_ppm']  = ((df['etoh_vgas'] - df['etoh_vref']) / M_ETOH - ETOH_OFFSET_PPM).clip(lower=0)
    mask            = detect_outliers_iqr(df['etoh_ppm'])
    df['etoh_ppm']  = handle_outliers(df['etoh_ppm'], mask, method="interpolate")
    df['etoh_ppm']  = savgol_filter(df['etoh_ppm'], window_length=51, polyorder=2)
    df['etoh_ppm']  = lowpass_filter(df['etoh_ppm'], cutoff_hz=0.005)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from cleaned sensor readings.
    Each row = one timestep with temporal features baked in.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Raw cleaned values ──
    feat['h2s_ppm']  = df['h2s_ppm']
    feat['etoh_ppm'] = df['etoh_ppm']
    feat['voc_v']    = df['voc']
    feat['nh3_v']    = df['nh3']
    feat['hcho_v']   = df['hcho']
    feat['temp_c']   = df['temp_C']
    feat['rh_pct']   = df['rh_pct']

    # ── Rate of change — diff of cleaned values, smoothed over window ──
    # This is the column you were missing — absolute change per sample
    # then averaged over ROC_WINDOW to reduce noise
    for col in ['h2s_ppm', 'etoh_ppm', 'voc_v', 'nh3_v', 'hcho_v']:
        raw_diff = feat[col].diff().abs()  # |current - previous|
        feat[f'{col}_roc'] = raw_diff.rolling(
            window=ROC_WINDOW, center=True, min_periods=1).mean()

    # ── Rolling mean and std — computed from feat[col] not df[col] ──
    for col in ['h2s_ppm', 'etoh_ppm', 'voc_v', 'nh3_v', 'hcho_v']:
        feat[f'{col}_mean'] = feat[col].rolling(
            window=ROLLING_WINDOW, min_periods=1).mean()
        feat[f'{col}_std']  = feat[col].rolling(
            window=ROLLING_WINDOW, min_periods=1).std().fillna(0)

    # ── Rolling mean of ROC — captures trend in rate of change ──
    for col in ['h2s_ppm', 'etoh_ppm', 'voc_v', 'nh3_v', 'hcho_v']:
        feat[f'{col}_roc_mean'] = feat[f'{col}_roc'].rolling(
            window=ROLLING_WINDOW, min_periods=1).mean()

    # ── Cross-sensor ratios ──
    feat['h2s_to_voc']  = feat['h2s_ppm']  / (feat['voc_v']  + 1e-6)
    feat['nh3_to_hcho'] = feat['nh3_v']    / (feat['hcho_v'] + 1e-6)
    feat['h2s_to_nh3']  = feat['h2s_ppm']  / (feat['nh3_v']  + 1e-6)
    feat['etoh_to_voc'] = feat['etoh_ppm'] / (feat['voc_v']  + 1e-6)

    return feat.fillna(0)

# ── Labelling ─────────────────────────────────────────────────────────────────
def label_windows(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Assign labels to stable concentration windows only.
    Rows outside any window are dropped (transitions discarded).
    """
    labelled_chunks = []
    for w in windows:
        mask = df['time_s'] >= w['start_s']
        if w['end_s'] is not None:
            mask &= df['time_s'] < w['end_s']
        chunk = df[mask].copy()
        chunk['label'] = w['label']
        chunk['conc']  = w['conc']
        labelled_chunks.append(chunk)
    return pd.concat(labelled_chunks).sort_values('time_s').reset_index(drop=True)


# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(path_h2s: str, path_voc: str,
                  trim_voc_s: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    # Load and clean H2S run
    df_h2s = load_data(path_h2s)
    df_h2s = clean_h2s(df_h2s)
    df_h2s = clean_etoh(df_h2s)
    feat_h2s = engineer_features(df_h2s)
    feat_h2s['time_s'] = df_h2s['time_s']
    labelled_h2s = label_windows(feat_h2s, H2S_STABLE_WINDOWS)

    # Load and clean VOC run
    df_voc = load_data(path_voc)
    if trim_voc_s > 0:
        df_voc = trimmed_data(df_voc, start_s=trim_voc_s)
        df_voc['time_s'] = df_voc['time_s'] - df_voc['time_s'].iloc[0]
    df_voc = clean_h2s(df_voc)
    df_voc = clean_etoh(df_voc)
    feat_voc = engineer_features(df_voc)
    feat_voc['time_s'] = df_voc['time_s']
    labelled_voc = label_windows(feat_voc, VOC_STABLE_WINDOWS)

    # Combine
    combined = pd.concat([labelled_h2s, labelled_voc]).reset_index(drop=True)

    feature_cols = [c for c in combined.columns if c not in ['label', 'conc', 'time_s']]
    X = combined[feature_cols]
    y = combined['label']

    print(f"\n── Dataset summary ───────────────────────────────────")
    print(f"  Total samples : {len(combined)}")
    print(f"  H2S  (label=1): {(y == LABEL_H2S).sum()}")
    print(f"  VOC  (label=0): {(y == LABEL_VOC).sum()}")
    print(f"  Features      : {len(feature_cols)}")
    print(f"──────────────────────────────────────────────────────")

    return X, y, combined


# ── Online Random Forest training ─────────────────────────────────────────────
def train_online_rf(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Prequential evaluation — predict first, then learn.
    Returns trained model and running metrics.
    """
    model = preprocessing.StandardScaler() | ensemble.SRPClassifier(
        n_models=10,
        seed=42,
    )

    acc_metric    = metrics.Accuracy()
    cm_metric     = metrics.ConfusionMatrix()

    running_acc   = []
    predictions   = []
    ground_truths = []

    for i, (_, row) in enumerate(X.iterrows()):
        features = row.to_dict()
        label    = int(y.iloc[i])

        # Predict first (prequential)
        pred = model.predict_one(features)
        if pred is None:
            pred = LABEL_VOC  # default before first learn

        # Update metrics
        acc_metric.update(label, pred)
        cm_metric.update(label, pred)
        running_acc.append(acc_metric.get())
        predictions.append(pred)
        ground_truths.append(label)

        # Learn
        model.learn_one(features, label)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>5} / {len(X)}]  accuracy = {acc_metric.get():.3f}")

    print(f"\n── Training complete ─────────────────────────────────")
    print(f"  Final accuracy : {acc_metric.get():.4f}")
    print(f"\n  Confusion matrix:")
    print(f"  {cm_metric}")
    print(f"──────────────────────────────────────────────────────")

    return model, running_acc, predictions, ground_truths


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(running_acc: list, predictions: list,
                 ground_truths: list, combined: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Running accuracy ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(running_acc, color="#378ADD", lw=1.0)
    ax1.axhline(0.9, color="#D85A30", lw=0.8, ls="--", label="90% threshold")
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("accuracy")
    ax1.set_title("running accuracy (prequential)")
    ax1.set_ylim(0, 1.05)
    ax1.legend()

    # ── Predictions vs ground truth ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ground_truths, color="#cccccc", lw=0.5, label="ground truth", alpha=0.8)
    ax2.plot(predictions,   color="#D85A30", lw=0.5, label="predicted",    alpha=0.8)
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("label (0=VOC, 1=H2S)")
    ax2.set_title("predictions vs ground truth")
    ax2.legend(fontsize=8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["VOC", "H2S"])

    # ── Class distribution ──
    ax3 = fig.add_subplot(gs[1, 1])
    counts = pd.Series(ground_truths).value_counts().sort_index()
    ax3.bar([LABEL_NAMES[k] for k in counts.index], counts.values,
            color=["#D85A30", "#378ADD"], alpha=0.85)
    for i, v in enumerate(counts.values):
        ax3.text(i, v + 5, str(v), ha="center", fontsize=10)
    ax3.set_ylabel("sample count")
    ax3.set_title("class distribution")

    fig.suptitle("Online Random Forest — H2S vs VOC classification", fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    path_h2s = "testrun-h2s-st-2/readings_20260312_144332.csv"
    path_voc = "testrun-voc-st/readings_20260312_151908.csv"

    print("Building dataset...")
    X, y, combined = build_dataset(
        path_h2s  = path_h2s,
        path_voc  = path_voc,
        trim_voc_s = 1400,
    )

    print("\nTraining Online Random Forest...")
    model, running_acc, predictions, ground_truths = train_online_rf(X, y)

    plot_results(running_acc, predictions, ground_truths, combined)


if __name__ == "__main__":
    main()