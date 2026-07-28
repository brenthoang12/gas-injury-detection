# Gas Injury Detection

Detecting blood inside a prosthetic socket from the gases it gives off, using a low-cost
multi-channel gas sensor array on an ESP32. The system distinguishes three states —
**baseline** (nothing present), **sweat**, and **blood** — so that a wound inside a socket can
be flagged without removing the limb.

The hard part is not detecting blood against clean air; it is separating blood from sweat,
since sweat is always present in a socket and the two produce overlapping responses at low
concentrations.

## Sensor array

Six channels are logged at 1 Hz:

| Channel | Sensor type | Notes |
|---|---|---|
| `voc`, `nh3`, `hcho` | MEMS metal-oxide | Analog voltage; the primary discriminating channels |
| `h2s_ppm`, `etoh_ppm` | SPEC electrochemical | Converted to ppm from Vgas/Vref; **uncalibrated**, drifts between sessions |
| `rh_pct` (+ temp) | HDC302x (I2C) | Humidity is a model input; temperature is logged for reference |

Hardware is an ESP32 DOIT DevKit v1. Pins and the SPEC ppm conversion constants are defined in
[src/main.cpp](src/main.cpp).

## Repository layout

| Path | What it is |
|---|---|
| [src/main.cpp](src/main.cpp) | Data-collection firmware — reads all six channels at 1 Hz and logs over serial |
| `2026MMDD-experiment/` | Raw session captures (CSV + quick-look PNG), one folder per recording day |
| [experiment-data-clean/](experiment-data-clean/) | The analysis and modeling pipeline (see below) |
| [experiment-data-clean/esp32-blood-detect/](experiment-data-clean/esp32-blood-detect/) | Deployment firmware — runs the trained model on-device |
| [experiment-data-analysis/](experiment-data-analysis/) | Earlier exploratory scripts (superseded by the notebooks) |
| [model-creation/](model-creation/) | Early modeling experiments (superseded) |

## Pipeline

Raw capture → filtering → feature engineering → windowing → model → device.

1. **Filtering and baseline correction** — [mems_filter.ipynb](experiment-data-clean/mems_filter.ipynb)
   and [spec_filter.ipynb](experiment-data-clean/spec_filter.ipynb) clean each sensor family and
   write per-session pickles to `experiment-data-clean/processed/`. Session timing (warm-up,
   sample introduction) lives in these notebooks as `INTRO_TIMES` / `SAMPLE_WINDOWS`.
2. **Feature engineering** — each of the 6 channels expands to 12 engineered columns (raw, rate
   of change, acceleration, and rolling mean/std/rate-of-change at 15, 30, and 60 s) = **72
   columns**.
3. **Windowing** — non-overlapping 60 s windows, each summarized by 4 aggregates
   (mean, std, max, slope) = **288 features per window**. Windows never cross a label boundary.
4. **Modeling** — [combined_no_mix140.ipynb](experiment-data-clean/combined_no_mix140.ipynb)
   trains and compares Random Forest, XGBoost, a 1D CNN, and a dense neural network, with
   leave-one-session-out (LOSO) cross-validation and SHAP-based diagnostics.

## Dataset

28 sessions are used: 8 sweat, 10 blood, and 10 blood/sweat mixtures. Mixtures are named by
their sweat/blood split in hundredths of a millilitre, out of a 1.5 ml total:

| Group | Sweat / blood | Blood fraction | Sessions |
|---|---|---|---|
| `mix_75_75` | 0.75 / 0.75 ml | 50% | 5 |
| `mix_120_30` | 1.20 / 0.30 ml | 20% | 5 |
| `mix_140_10` | 1.40 / 0.10 ml | 6.7% | 5 (excluded) |

The most dilute group, `mix_140_10`, is **excluded** — at 6.7% blood it sits below the detection
floor and is not separable from sweat, which is why the main modeling notebook is named
`combined_no_mix140`.

## Results

Mean per-session accuracy under leave-one-session-out cross-validation (28 folds, each model
trained without the held-out session):

| Model | LOSO accuracy |
|---|---|
| Random Forest | 94.4% |
| **XGBoost (deployed)** | **94.9%** |
| 1D CNN | 97.3% |
| Dense NN | 95.8% |

Most sessions classify near-perfectly; errors concentrate in a few detection-floor sessions.
The tree models fail on unusually weak sweat by falling back to baseline, and SHAP traces the
remaining blood-side errors to the uncalibrated electrochemical channels. Sensor calibration,
not further modeling, is the clearest way to raise the ceiling.

**XGBoost is deployed** despite the CNN's marginally higher score: it is interpretable through
SHAP (which is how the sensor problem was found), exposes direct controls for down-weighting
the unreliable channels, and has a small fixed footprint suited to the ESP32.

## Deployment

The trained model is exported from the modeling notebook to `experiment-data-clean/model_export/`,
then compiled to plain C headers for the device.

```bash
# 1. Validate the exported model in Python (round-trip + per-second simulation)
cd experiment-data-clean
python test_exported_model.py

# 2. Regenerate the C headers from the export (includes a host parity check)
cd esp32-blood-detect
../../.venv/bin/python3 gen_headers.py

# 3. Build and flash
pio run -t upload
pio device monitor        # 115200 baud
```

The device runs the full feature pipeline, the tree model, and a causal EMA smoother in C with
no ML library on board, producing one prediction per second from the trailing 60 s window. See
[experiment-data-clean/esp32-blood-detect/README.md](experiment-data-clean/esp32-blood-detect/README.md)
for wiring, verification, and the known preprocessing caveats.

## Build (data collection)

```bash
pio run -t upload         # flashes src/main.cpp, the logging firmware
pio device monitor
```

## Known limitations

- **Uncalibrated electrochemical sensors.** EtOH and H2S drift between sessions and are the
  single largest source of cross-session error. Calibration requires a rig that is not yet built.
- **Detection floor.** During onset, dilute blood overlaps sweat in both level and rate of
  change. Mixtures at 20% blood are classified reliably; at 6.7% they are not, so the floor for
  this sensor array lies somewhere between those two dilutions.
- **Controlled conditions.** All data was collected at room temperature and 20–60% RH, using a
  Ziplock bag as the sample chamber. An inert (PTFE/polypropylene) chamber and a simulated
  in-socket environment are planned.
