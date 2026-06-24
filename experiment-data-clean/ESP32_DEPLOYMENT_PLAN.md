# ESP32 Deployment Plan — XGBoost Run 3

## 1. Exported artifacts (`model_export/`)

| File | What it is |
|---|---|
| `xgb_run3.json` | XGBoost native model (portable, human-readable) |
| `xgb_run3.ubj` | Same model, binary (smaller) |
| `xgb_run3_trees.json` | The 255 decision trees as JSON — source for C conversion |
| `xgb_run3_features.json` | The **230 selected feature names, in order** — defines the model's input vector |
| `xgb_run3_config.json` | Window size, channels, rolling windows, smoothing params, class labels |

**Model summary:** 3 classes (baseline / sweat / blood), **255 trees** (~85 boosting rounds × 3 classes), input is **230 features** (top 80% of the 288 window features), output is 3 raw margins → softmax → argmax.

## 2. On-device pipeline (runs at 1 Hz)

The model is the easy part. The real work is reproducing the **feature pipeline causally in C**.

1. **Sample sensors** (1 Hz): VOC, NH3, HCHO, H2S, EtOH, RH. (temperature is read but not used as a feature)
2. **Per-sample feature engineering** (incremental, causal) — for each of the 6 channels:
   - `roc = x - x_prev`, `acc = roc - roc_prev`
   - rolling mean / std / abs-roc at 15 / 30 / 60 s (ring buffers or running sums)
   - → 72 engineered columns per sample
3. **Window buffer**: keep a 60-sample ring buffer of the 72 columns.
4. **Every 60 s** (non-overlapping window): compute `mean / std / max / slope` of each column → 288 features → pick the **230** the model needs (order from `xgb_run3_features.json`).
5. **Inference**: evaluate the 255 trees, sum per-class margins, softmax → `P(baseline / sweat / blood)`.
6. **Causal smoothing**: EMA per class — `ema = α·p + (1-α)·ema_prev` (α = 0.4). `argmax(ema)` is the prediction. (EMA is preferred over the trailing mean on-device: one running value per class, no buffer.)
7. **Output**: LED / serial / display.

## 3. Resource budget (rough, ESP32 has ~320 KB usable RAM, 4 MB flash, 240 MHz)

- **Flash**: compiled trees ≈ 50–100 KB. Fine.
- **RAM**: 60×72 window buffer ≈ 17 KB + rolling buffers + 230-float feature vector ≈ well under 50 KB. Fine.
- **Timing**: 288 feature computations + 255 tree evaluations **once per 60 s** is trivial at 240 MHz. Real-time with huge margin.

## 4. Converting the model to C

`m2cgen` does **not** support XGBoost 3.x's classifier wrapper (the version used here). Three options:

- **(A) Quick demo** — make a throwaway venv with `pip install "xgboost<2.0"`, load `xgb_run3.json` (the native format is version-portable), then `m2cgen.export_to_c` → `xgb_run3.c`. Least code.
- **(B) Tree-walker** — write a ~100-line generic C interpreter that walks `xgb_run3_trees.json` converted to a flat struct array. Keeps the model as data; fully version-independent.
- **(C) treelite** (recommended for production) — compiles modern XGBoost models straight to C; supports XGBoost 3.x. Clean and fast.

## 5. The real risk: preprocessing parity

**This is the part most likely to break the demo.** The model was trained on data that went through the offline pipeline in `mems_filter` / `spec_filter`:

- **Filtering** (EMA / Savitzky-Golay / Butterworth) — causal versions exist, port them.
- **PPM computation** (H2S, EtOH from raw Vgas/Vref) — a fixed formula, easy to port (or read the sensor's computed ppm).
- **Baseline / drift correction** — **the hard one.** Training subtracted a per-session baseline measured from the pre-sample window. On-device there is no "whole session" to look back on. You need a real-time strategy: capture a baseline at startup (after warm-up) and subtract it, or re-train on data without offline baseline correction.
- **Warm-up** — the sensor's boot drift (high decaying NH3) must be handled: either wait for stabilization before predicting, or rely on the model (it learned warm-up is not blood from the flat VOC/HCHO).

**Action:** before trusting on-device predictions, feed one recorded session through both the Python pipeline and the C pipeline and confirm the **230 features match feature-by-feature**. That parity test is where bugs hide.

## 6. Suggested build order

1. **Offline sanity** — load `xgb_run3.json` in Python, confirm it reproduces the 95.8% held-out result.
2. **Model to C** — pick A / B / C, generate the inference function, unit-test it on a few exported feature vectors vs Python.
3. **Feature pipeline in C** — port engineering + window aggregation; validate feature-by-feature against Python on a recorded session.
4. **Baseline strategy** — implement startup-baseline capture (or decide to retrain without offline correction).
5. **Integrate** — sensors → pipeline → model → EMA → output.
6. **Live test** — run a known blood / sweat sample, confirm detection and latency.

## 7. Known limitations to design around

- **60 s latency**: predictions arrive once per non-overlapping window. For a snappier live demo, implement the sliding-window approach (see the notebook TODO).
- **Detection floor**: reliable down to 20 % blood; 6.7 % is below the sensor floor.
- **Onset lag**: causal smoothing commits a few windows after a sample is introduced — expected, and the only place the model errs.
