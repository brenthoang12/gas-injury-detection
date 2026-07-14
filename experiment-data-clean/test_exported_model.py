"""Standalone tests for the exported XGBoost Run 6 model.

Run the export cells in combined_no_mix140.ipynb first so model_export/ contains:
  xgb_run6.json, export_meta.json, feature_cols.json,
  test_windows.pkl, test_windows_refpred.npy, persecond_mix_75_75_1.pkl

Test 1: reload the exported model and confirm it reproduces the notebook model's
        predictions on the held-out windows exactly (round-trip check).
Test 2: take one mix_75_75 session and run the model "every second" — at each
        second, build the trailing 60s window's features and predict — to see
        what the continuous, unsmoothed prediction stream looks like.

Run:  python test_exported_model.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, "model_export")
CLASS_NAMES = {0: "baseline", 1: "sweat", 2: "blood"}

# ── Load model + metadata ────────────────────────────────────────────────────
model = XGBClassifier()
model.load_model(f"{EXPORT}/xgb_run6.json")
meta = json.load(open(f"{EXPORT}/export_meta.json"))
RUN6_COLS = meta["run6_cols"]
WINDOW = meta["window_size"]
FEATURE_COLS = json.load(open(f"{EXPORT}/feature_cols.json"))


def extract_window_features(window_df, feature_cols):
    """Identical to the notebook's window aggregation: mean/std/max/slope per channel.
    NaN is left in place (XGBoost handles it natively, as it did during training)."""
    feats = {}
    t = np.arange(len(window_df))
    for col in feature_cols:
        vals = window_df[col].values
        feats[f"{col}_mean"] = np.nanmean(vals)
        feats[f"{col}_std"] = np.nanstd(vals)
        feats[f"{col}_max"] = np.nanmax(vals)
        feats[f"{col}_slope"] = np.polyfit(t, vals, 1)[0] if len(vals) > 1 else 0.0
    return feats


# ── Test 1: exact match against the notebook model's predictions ─────────────
def test_1_exact_match():
    df = pd.read_pickle(f"{EXPORT}/test_windows.pkl")
    ref = np.load(f"{EXPORT}/test_windows_refpred.npy", allow_pickle=True)
    pred = model.predict(df[RUN6_COLS])

    n_match = int((pred == ref).sum())
    print("=" * 60)
    print("[Test 1] Exported model vs notebook model on held-out windows")
    print("=" * 60)
    print(f"  windows:        {len(ref)}")
    print(f"  exact matches:  {n_match}/{len(ref)}  ({n_match / len(ref):.2%})")
    print(f"  accuracy vs true labels: {(pred == df['label'].values).mean():.2%}")
    assert n_match == len(ref), "reloaded model predictions differ from the reference!"
    print("  PASS: exported file reproduces the notebook model exactly.")


# ── Test 2: predict every second on one mix_75_75 session ────────────────────
def test_2_per_second(session=None):
    session = session or meta["sec_session"]
    df = pd.read_pickle(f"{EXPORT}/persecond_{session}.pkl").reset_index(drop=True)

    # slide a trailing 60s window, one prediction per second once history is full
    rows, times, truth = [], [], []
    for end in range(WINDOW, len(df) + 1):
        w = df.iloc[end - WINDOW:end]
        rows.append(extract_window_features(w, FEATURE_COLS))
        times.append(float(df["elapsed_s"].iloc[end - 1]))
        truth.append(int(df["label"].iloc[end - 1]))
    X = pd.DataFrame(rows)[RUN6_COLS]
    pred = model.predict(X)
    truth = np.array(truth)
    times = np.array(times)

    print("\n" + "=" * 60)
    print(f"[Test 2] Per-second prediction on {session}")
    print("=" * 60)
    print(f"  seconds predicted: {len(pred)}  (one per second after the first {WINDOW}s)")
    print("  predicted-class breakdown:")
    for k, name in CLASS_NAMES.items():
        c = int((pred == k).sum())
        print(f"    {name:<9} {c / len(pred):>6.1%}  ({c})")
    print(f"  accuracy vs per-second labels: {(pred == truth).mean():.1%}")

    # run-length timeline: collapse consecutive equal predictions into segments
    print("\n  prediction timeline (start-end seconds -> class):")
    start = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[start]:
            print(f"    {times[start]:>6.0f}s - {times[i - 1]:>6.0f}s   {CLASS_NAMES[int(pred[start])]}")
            start = i

    _plot_per_second(session, df, times, pred, truth)


def _plot_per_second(session, df, times, pred, truth):
    """Top: raw per-second prediction vs the true label. Bottom: the sensor
    channels, so the transition/settling region lines up with the signal."""
    tmin = times / 60.0
    ok = pred == truth
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.step(tmin, truth, where="post", color="0.7", lw=1.4, zorder=1, label="true label")
    ax1.scatter(tmin[ok], pred[ok], color="seagreen", s=20, zorder=3, label="correct")
    ax1.scatter(tmin[~ok], pred[~ok], color="crimson", s=30, marker="X", zorder=3, label="wrong")
    ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(["baseline", "sweat", "blood"])
    ax1.set_ylim(-0.5, 2.5); ax1.set_ylabel("prediction (per second)")
    ax1.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8, loc="center left", ncol=3)
    ax1.set_title(f"{session} — raw per-second prediction  (accuracy {(pred == truth).mean():.1%})",
                  fontsize=10, fontweight="bold")

    tfull = df["elapsed_s"].values / 60.0
    for ch in ["voc", "nh3", "hcho", "etoh_ppm"]:
        if ch in df.columns:
            ax2.plot(tfull, df[ch].values, lw=0.9, alpha=0.85, label=ch)
    ax2.set_ylabel("sensor value"); ax2.set_xlabel("elapsed time (min)")
    ax2.grid(True, alpha=0.25, linestyle="--"); ax2.legend(fontsize=8, ncol=4)

    plt.tight_layout()
    out = os.path.join(EXPORT, f"test2_per_second_{session}.png")
    fig.savefig(out, dpi=130)
    print(f"\n  saved figure: {out}")
    plt.show()


if __name__ == "__main__":
    test_1_exact_match()
    test_2_per_second()
