// ─────────────────────────────────────────────────────────────────────────────
// ESP32 real-time blood detection — XGBoost Run 6  (PlatformIO / Arduino framework)
//
// Always sampling at 1 Hz. Every second: read sensors -> causal filters -> print.
// History is kept continuously, so the button can look BACKWARD in time.
//
//   Button press  -> fit the baseline from the history already collected:
//                    MEMS gets a linear drift trend, SPEC a constant offset.
//                    Classification starts once a baseline exists.
//   Button hold*   -> (short press again) re-fit the baseline from newer history.
//
// Filtering mirrors the offline training pipeline (see filters.h for what matches
// exactly and what is approximated). The model (xgb_model.h) and feature pipeline
// (features.h) are generated and verified against Python.
//
// USAGE: power on and leave it in CLEAN AIR. Readings print immediately. Once the
//        trace looks settled (a couple of minutes at least), press the button to
//        capture the baseline, then introduce the sample.
// ─────────────────────────────────────────────────────────────────────────────
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_HDC302x.h>

#include "feature_spec.h"   // channel order, feature layout
#include "filters.h"        // causal filters + baseline capture
#include "features.h"       // feat_reset / feat_push / feat_window
#include "xgb_model.h"      // xgb_predict_proba

// ── Pins (from src/main.cpp) ──────────────────────────────────────────────────
#define PIN_BUTTON    4
#define PIN_LED       2
#define PIN_HCHO      39
#define PIN_VOC       35
#define PIN_NH3       34
#define PIN_ETOH_GAS  25
#define PIN_ETOH_REF  33
#define PIN_H2S_GAS   27
#define PIN_H2S_REF   26

// ── SPEC PPM constants — TRAINING values (match spec_filter, not the logger) ──
// ppm = clip((Vgas + offset) - Vref, 0) / m * scale ,  m = sensCode * tiaGain * 1e-6
static const float M_H2S       = 216.09f * 49.9f  * 1e-6f;
static const float H2S_OFFSET  = 0.0f;
static const float H2S_SCALE   = 1.5f;     // <- the logger's calcSpecPPM omits this
static const float M_ETOH      = 21.5f  * 249.0f * 1e-6f;
static const float ETOH_OFFSET = 0.100f;   // <- and omits this
static const float ETOH_SCALE  = 1.0f;

// ── Model / smoothing ─────────────────────────────────────────────────────────
static const char* CLASS_NAME[XGB_N_CLASSES] = { "baseline", "sweat", "blood" };
// Per-second smoothing. Raw per-second predictions were already ~100% in test 2
// with only a 1 s blip at the transition, so a light EMA is enough. Tune 0.1-0.3:
// lower = steadier but slower to switch, higher = snappier but more jitter.
static const float EMA_ALPHA = 0.2f;

Adafruit_HDC302x hdc = Adafruit_HDC302x();
bool  hdcReady   = false;
bool  lastButton = HIGH;

float ema[XGB_N_CLASSES] = { 0 };
bool  ema_init = false;
unsigned long lastMeasure = 0;
long  t_sec = 0;                 // seconds since boot — the clock for the drift fit

// ── Helpers (from src/main.cpp) ──────────────────────────────────────────────
static float getAverageVoltage(int pin) {
    long sum = 0;
    for (int i = 0; i < 64; i++) { sum += analogRead(pin); delay(2); }
    return (sum / 64.0f) * (3.3f / 4095.0f);
}
static float calcSpecPPM(float vgas, float vref, float m, float offset, float scale) {
    float ppm = ((vgas + offset) - vref) / m;
    if (ppm < 0) ppm = 0;
    return ppm * scale;
}

// Raw (unfiltered, uncorrected) channel reads, in the model's channel order.
static void readRaw(float* r) {
    r[CH_VOC]  = getAverageVoltage(PIN_VOC);                             // voc  (V)
    r[CH_NH3]  = getAverageVoltage(PIN_NH3);                             // nh3  (V)
    r[CH_HCHO] = getAverageVoltage(PIN_HCHO);                            // hcho (V)
    r[CH_H2S]  = calcSpecPPM(getAverageVoltage(PIN_H2S_GAS),  getAverageVoltage(PIN_H2S_REF),
                             M_H2S,  H2S_OFFSET,  H2S_SCALE);            // h2s_ppm
    r[CH_ETOH] = calcSpecPPM(getAverageVoltage(PIN_ETOH_GAS), getAverageVoltage(PIN_ETOH_REF),
                             M_ETOH, ETOH_OFFSET, ETOH_SCALE);           // etoh_ppm
    double t, rh; rh = 0;
    if (hdcReady && hdc.readTemperatureHumidityOnDemand(t, rh, TRIGGERMODE_LP0)) {}
    r[CH_RH] = (float)rh;                                                // rh_pct
}

// Button: capture (or re-capture) the baseline from the history already collected.
static void handleButton() {
    bool b = digitalRead(PIN_BUTTON);
    if (lastButton == HIGH && b == LOW) {
        delay(50);                                       // debounce
        Serial.println();
        if (!filters_capture_baseline()) {
            Serial.printf("!! baseline needs %d s of history, only have %d s — keep waiting\n",
                          BASELINE_MIN_S, hist_n);
        } else {
            Serial.printf("=== BASELINE CAPTURED === from %d s of history\n", hist_n);
            Serial.printf("  drift (V/s): voc=%+.3e nh3=%+.3e hcho=%+.3e\n",
                          base_slope[CH_VOC], base_slope[CH_NH3], base_slope[CH_HCHO]);
            Serial.printf("  level  @now: voc=%.4f nh3=%.4f hcho=%.4f  offset: h2s=%.3f etoh=%.3f\n",
                          base_slope[CH_VOC] * t_sec + base_icept[CH_VOC],
                          base_slope[CH_NH3] * t_sec + base_icept[CH_NH3],
                          base_slope[CH_HCHO] * t_sec + base_icept[CH_HCHO],
                          base_icept[CH_H2S], base_icept[CH_ETOH]);
            Serial.println("  classifying — first prediction after 60 s of corrected data");
            feat_reset();                                // window must be built from corrected data
            ema_init = false;
        }
        Serial.println();
    }
    lastButton = b;
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pinMode(PIN_LED, OUTPUT);
    digitalWrite(PIN_LED, LOW);
    Wire.begin(21, 22);
    hdcReady = hdc.begin(0x44, &Wire);
    filters_reset();
    feat_reset();
    Serial.println();
    Serial.println("ESP32 blood-detect ready");
    Serial.printf("model: %d trees | %d features | %d s window | EMA alpha=%.2f | HDC %s\n",
                  XGB_N_TREES, XGB_N_FEATURES, WIN_SIZE, EMA_ALPHA, hdcReady ? "ok" : "MISSING");
    Serial.printf("filters: MEMS EMA span %d | SPEC Butterworth fc 0.005 Hz | RH EMA span %d\n",
                  MEMS_EMA_SPAN, RH_EMA_SPAN);
    Serial.printf("Reading now. Press the button after >=%d s in clean air to set the baseline.\n",
                  BASELINE_MIN_S);
    Serial.println("t,voc,nh3,hcho,h2s,etoh,rh   (filtered; corrected once baseline is set)");
}

void loop() {
    handleButton();

    if (millis() - lastMeasure < 1000) { delay(5); return; }   // 1 Hz
    lastMeasure = millis();

    float raw[N_CHANNELS], filt[N_CHANNELS];
    readRaw(raw);
    filters_apply(raw, filt);                    // causal filters — always running
    filters_push_history(filt, t_sec);           // keep history so the button can look back
    t_sec++;

    // Always print the current reading.
    Serial.printf("[%lds] %.4f %.4f %.4f %.3f %.3f %.1f",
                  t_sec, filt[CH_VOC], filt[CH_NH3], filt[CH_HCHO],
                  filt[CH_H2S], filt[CH_ETOH], filt[CH_RH]);

    if (!baseline_ready) {                       // no baseline yet: monitor only
        Serial.printf("   | no baseline (%d s history)\n", hist_n);
        return;
    }

    float corr[N_CHANNELS];
    filters_correct(filt, t_sec, corr);          // subtract drift trend / offset, clip at 0
    feat_push(corr);

    if (feat_count < WIN_SIZE) {                 // trailing window not yet full
        Serial.printf("   | filling window %ld/%d\n", feat_count, WIN_SIZE);
        return;
    }

    float feat[N_SELECTED];
    feat_window(feat);                           // aggregate the trailing 60 s -> 288 features

    float proba[XGB_N_CLASSES];
    xgb_predict_proba(feat, proba);

    if (!ema_init) { for (int c = 0; c < XGB_N_CLASSES; c++) ema[c] = proba[c]; ema_init = true; }
    else { for (int c = 0; c < XGB_N_CLASSES; c++) ema[c] = EMA_ALPHA * proba[c] + (1 - EMA_ALPHA) * ema[c]; }

    int pred = 0;
    for (int c = 1; c < XGB_N_CLASSES; c++) if (ema[c] > ema[pred]) pred = c;

    Serial.printf("   | %-8s %3.0f%%  (b %.2f / s %.2f / bl %.2f)\n",
                  CLASS_NAME[pred], ema[pred] * 100.0f, ema[0], ema[1], ema[2]);

    digitalWrite(PIN_LED, pred == 2 ? HIGH : LOW);   // LED solid while blood is the call
}
