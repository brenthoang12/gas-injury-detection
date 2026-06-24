"""
Validate the exported XGBoost Run 3 model on the held-out test sessions.

Loads the artifacts in model_export/ (does NOT retrain), rebuilds the test windows
with the same feature pipeline as training, applies the model + causal smoothing,
and reports per-session accuracy. This is the offline sanity check before any
on-device / ESP32 work: confirm the exported model reproduces ~95.8%.

Run:  python test_exported_model.py
Needs: model_export/  and  processed/
"""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb

# Paths are anchored to this script's folder so it runs from any working directory
# (e.g. the outer repo root where the venv lives), not just experiment-data-clean/.
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT    = os.path.join(HERE, "model_export")
PROCESSED = os.path.join(HERE, "processed")
MEMS = ["voc", "nh3", "hcho"]; SPEC = ["h2s_ppm", "etoh_ppm"]; ENV = ["temp_C", "rh_pct"]
ALL = MEMS + SPEC + ENV
LB, LS, LBL = 0, 1, 2
CLASS_NAMES = {LB: "baseline", LS: "sweat", LBL: "blood"}

# Held-out test sessions and their labelled sample windows (from SAMPLE_WINDOWS)
TEST_WINDOWS = {
    "sweat_7":      [(1760, 3610, LS)],
    "blood_9":      [(2290, 4190, LBL)],
    "mix_75_75_3":  [(1390, None, LBL)],
    "mix_120_30_4": [(1675, None, LBL)],
}

# ── Load exported artifacts ───────────────────────────────────────────────────
cfg  = json.load(open(f"{EXPORT}/xgb_run3_config.json"))
cols = json.load(open(f"{EXPORT}/xgb_run3_features.json"))       # selected features, in order
booster = xgb.Booster(); booster.load_model(f"{EXPORT}/xgb_run3.json")

WIN      = cfg["window_size_s"]
CHANNELS = cfg["channels"]
ROLL     = cfg["roll_windows_s"]
SMOOTH   = cfg["smoothing"]
K        = SMOOTH["trailing_window"]
A        = SMOOTH["ema_alpha"]
print(f"Loaded model: {cfg['n_trees']} trees | {len(cols)} features | window {WIN}s | "
      f"smoothing causal EMA (alpha={A})")

# ── Load + merge the test sessions ────────────────────────────────────────────
def _load(prefix):
    out = {}
    for f in sorted(os.listdir(PROCESSED)):
        if f.startswith(prefix) and f.endswith(".pkl"):
            key = f[len(prefix):-4]
            if key in TEST_WINDOWS:
                out[key] = pd.read_pickle(os.path.join(PROCESSED, f))
    return out

mems, spec = _load("mems_"), _load("spec_")
test = {}
for k in TEST_WINDOWS:
    m, s = mems[k], spec[k]
    merged = pd.merge(m, s[["elapsed_s"] + SPEC], on="elapsed_s", how="inner")
    if len(merged) < max(len(m), len(s)):
        m_s = m.sort_values("elapsed_s").reset_index(drop=True)
        s_s = s[["elapsed_s"] + SPEC].sort_values("elapsed_s").reset_index(drop=True)
        merged = pd.merge_asof(m_s, s_s, on="elapsed_s", tolerance=0.5,
                               direction="nearest").dropna(subset=SPEC).reset_index(drop=True)
    test[k] = merged[["elapsed_s"] + ALL].reset_index(drop=True)

# ── Feature engineering + labelling (must match training exactly) ─────────────
def feateng(df):
    df = df.copy()
    for c in CHANNELS:
        df[f"{c}_roc"] = df[c].diff()
        df[f"{c}_acc"] = df[f"{c}_roc"].diff()
        for w in ROLL:
            df[f"{c}_roll_mean_{w}"] = df[c].rolling(w, min_periods=1).mean()
            df[f"{c}_roll_std_{w}"]  = df[c].rolling(w, min_periods=1).std().fillna(0)
            df[f"{c}_roll_roc_{w}"]  = df[f"{c}_roc"].abs().rolling(w, min_periods=1).mean()
    return df

EXC = {"elapsed_s", "label", "session", "temp_C"}
for k in test:
    df = feateng(test[k]); df["label"] = LB
    for t0, t1, lbl in TEST_WINDOWS[k]:
        t = df["elapsed_s"]
        mask = (t >= t0) & (t <= (t1 if t1 is not None else t.iloc[-1]))
        df.loc[mask, "label"] = lbl
    test[k] = df
FEAT = [c for c in next(iter(test.values())).columns if c not in EXC]

def window_features(w):
    f = {}; t = np.arange(len(w))
    for c in FEAT:
        v = w[c].values
        f[f"{c}_mean"]  = np.nanmean(v)
        f[f"{c}_std"]   = np.nanstd(v)
        f[f"{c}_max"]   = np.nanmax(v)
        f[f"{c}_slope"] = (np.polyfit(t, v, 1)[0] if len(v) > 1 else 0.0)
    return f

# ── Predict (loaded model) + causal smoothing, per session ────────────────────
print(f"\n{'Session':<16}{'Windows':>9}{'Raw':>9}{'Smoothed':>11}")
print("-" * 45)
all_true, all_raw, all_sm = [], [], []
for sess in TEST_WINDOWS:
    df = test[sess]; rows, meta = [], []
    for lbl in [LB, LS, LBL]:
        ld = df[df["label"] == lbl].reset_index(drop=True)
        for q in range(len(ld) // WIN):
            w = ld.iloc[q * WIN:(q + 1) * WIN]
            rows.append(window_features(w)); meta.append((w["elapsed_s"].iloc[0], lbl))
    W = pd.DataFrame(rows)
    order = np.argsort([m[0] for m in meta])                 # chronological
    yt    = np.array([meta[o][1] for o in order])
    dm    = xgb.DMatrix(W.iloc[order][cols], feature_names=cols)
    probas = booster.predict(dm)                              # (n, 3) class probabilities
    raw   = np.argmax(probas, axis=1)
    sm    = np.argmax(pd.DataFrame(probas).ewm(alpha=A, adjust=False).mean().values, axis=1)  # causal EMA
    all_true.extend(yt); all_raw.extend(raw); all_sm.extend(sm)
    print(f"{sess:<16}{len(yt):>9}{(raw==yt).mean():>8.1%}{(sm==yt).mean():>10.1%}")

print("-" * 45)
at, ar, asm = map(np.array, (all_true, all_raw, all_sm))
print(f"{'OVERALL':<16}{len(at):>9}{(ar==at).mean():>8.1%}{(asm==at).mean():>10.1%}")
print(f"\nDeployed (EMA-smoothed) accuracy: {(asm==at).mean():.1%}  — expect ~97.7%")
