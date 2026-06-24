# ESP32 Blood Detection — XGBoost Run 3

Real-time, on-device version of the deployed model: a 1 Hz sensor loop that runs the
feature pipeline, the XGBoost tree model, and the causal EMA smoother entirely in C — no
external ML library on the device.

## Files

| File | What it is | Source |
|---|---|---|
| `main.cpp` | Main program (sensor loop → features → model → EMA → output) | hand-written |
| `xgb_model.h` | The 261 trees + `xgb_predict_proba()` | auto-generated from `model_export/` |
| `feature_spec.h` | Channel order, the 72 engineered columns, the 173 selected features | auto-generated |
| `features.h` | `feat_push()` / `feat_window()` — the causal feature pipeline | hand-written |
| `platformio.ini` | Build config | — |

## Verified correct

Both halves were checked on the host against the Python pipeline before shipping:

- **Model:** C `xgb_predict_proba` vs Python `predict_proba` → max difference **1.8e-07** (float precision).
- **Features:** C `feat_window` vs Python window features over a full session (67 windows) → max difference **4.6e-05**.

So if `read_sensors()` feeds the right values, the device reproduces the notebook's predictions.

## Sensors — already wired

`main.cpp` reads the 6 channels using the same pins, ADC setup, and HDC302x as the
data-collection firmware (`src/main.cpp`), in the model's channel order:

```
[0] voc   [1] nh3   [2] hcho   [3] h2s_ppm   [4] etoh_ppm   [5] rh_pct
```

Two corrections are applied so the values match how the model was trained:

- **SPEC PPM** (`h2s_ppm`, `etoh_ppm`) uses the **training** constants — including the H2S
  `×1.5` scale and EtOH `+0.100` offset that the logger's `calcSpecPPM` leaves out.
- **Baseline correction** — at button press in clean air, the clean-air reading of each gas
  channel is captured and subtracted from subsequent samples (response relative to baseline).

## Preprocessing parity (read this — still the real risk)

The model was trained on data filtered and baseline-corrected offline
(`mems_filter` / `spec_filter`). The two known approximations on-device:

- **Baseline correction** here is a single clean-air offset; training fit a *linear drift
  trend* over the whole pre-sample window. The constant-offset version is the best real-time
  approximation, but it is an approximation.
- **Filtering** — training applied EMA / Savitzky-Golay smoothing; this firmware feeds the
  64-sample-averaged reading without that extra stage.

**Before trusting the device:** record one session on the ESP32 (raw Vgas/Vref + MEMS
voltages), run it through the Python pipeline, and confirm the 173 features match — the host
parity test (`featverify`) is the template. If they diverge, the baseline/filtering
approximation is the place to look; the alternative is to retrain on minimally-processed data.

## Build

**PlatformIO**
```
cd esp32_blood_detect
pio run -t upload          # build + flash
pio device monitor         # 115200 baud
```

**Arduino IDE** — open `main.cpp` (folder name matches the sketch), select an
ESP32 board, upload, open Serial Monitor at 115200.

## Resource use

- Trees compile to ~60 KB flash; RAM for the feature buffers is ~20 KB. Comfortable on an ESP32.
- Inference (261 trees) + feature aggregation runs once per window — microseconds at 240 MHz.

## Output & latency

One prediction **per 60 s window**:
```
[window 7] blood    98%   (base 0.01 / sweat 0.01 / blood 0.98)
```

For a snappier live demo you can switch to a **sliding window** (predict every second using the
last 60 s): call `feat_window()` whenever `feat_count >= WIN_SIZE` instead of only on the
non-overlapping boundary. Note this produces many more windows, so retune `EMA_ALPHA` (the
current 0.4 was set for per-minute windows). This is the sliding-window item in the notebook TODO.

## Regenerating the model

`xgb_model.h` and `feature_spec.h` are generated from `../model_export/` (produced by the export
cell in `combined_no_mix140.ipynb`). Re-export there, then re-run the generator scripts to refresh
these headers, and re-run the host parity tests.
