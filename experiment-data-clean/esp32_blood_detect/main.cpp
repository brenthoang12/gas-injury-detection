// ─────────────────────────────────────────────────────────────────────────────
// ESP32 real-time blood detection — XGBoost Run 3  (PlatformIO / Arduino framework)
//
// 1 Hz: read sensors -> baseline-correct -> feature pipeline -> 60 s window
//       -> tree model -> causal EMA smoothing -> class output (Serial + LED)
//
// Sensor wiring / ADC setup / SPEC PPM math are taken from the data-collection
// firmware (../../src/main.cpp). The model (xgb_model.h) and feature pipeline
// (features.h) are auto-generated and verified bit-for-bit against Python.
//
// USAGE: power on, let sensors warm up, place in CLEAN AIR, press the button to
//        capture baselines and start classifying, then introduce the sample.
// ─────────────────────────────────────────────────────────────────────────────
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_HDC302x.h>

#include "feature_spec.h"   // channel order, feature layout
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
static const float EMA_ALPHA      = 0.4f;
static const int   WARMUP_WINDOWS = 1;

Adafruit_HDC302x hdc = Adafruit_HDC302x();
bool  hdcReady    = false;
bool  measuring   = false;
bool  lastButton  = HIGH;

// Baselines captured in clean air at button press (approximates the offline
// per-session drift correction with a constant offset — see README).
float v0_voc = 0, v0_nh3 = 0, v0_hcho = 0, b0_h2s = 0, b0_etoh = 0;

float ema[XGB_N_CLASSES] = { 0 };
bool  ema_init   = false;
int   window_idx = 0;
unsigned long lastMeasure = 0;

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

// Raw (uncorrected) channel reads, in the model's channel order.
static void readRaw(float* r) {
    r[0] = getAverageVoltage(PIN_VOC);                                   // voc  (V)
    r[1] = getAverageVoltage(PIN_NH3);                                   // nh3  (V)
    r[2] = getAverageVoltage(PIN_HCHO);                                  // hcho (V)
    r[3] = calcSpecPPM(getAverageVoltage(PIN_H2S_GAS),  getAverageVoltage(PIN_H2S_REF),
                       M_H2S,  H2S_OFFSET,  H2S_SCALE);                  // h2s_ppm
    r[4] = calcSpecPPM(getAverageVoltage(PIN_ETOH_GAS), getAverageVoltage(PIN_ETOH_REF),
                       M_ETOH, ETOH_OFFSET, ETOH_SCALE);                 // etoh_ppm
    double t, rh; rh = 0;
    if (hdcReady && hdc.readTemperatureHumidityOnDemand(t, rh, TRIGGERMODE_LP0)) {}
    r[5] = (float)rh;                                                    // rh_pct
}

// Baseline-corrected reads fed to the model (response relative to clean-air baseline).
static void read_sensors(float* out) {
    float r[N_CHANNELS]; readRaw(r);
    out[0] = r[0] - v0_voc;                       // MEMS: subtract clean-air baseline
    out[1] = r[1] - v0_nh3;
    out[2] = r[2] - v0_hcho;
    out[3] = r[3] - b0_h2s; if (out[3] < 0) out[3] = 0;   // SPEC: offset-corrected, clipped
    out[4] = r[4] - b0_etoh; if (out[4] < 0) out[4] = 0;
    out[5] = r[5];                                 // humidity used as-is
}

static void captureBaselines() {
    Serial.println("Calibrating baselines in clean air...");
    float r[N_CHANNELS]; readRaw(r);
    v0_voc = r[0]; v0_nh3 = r[1]; v0_hcho = r[2]; b0_h2s = r[3]; b0_etoh = r[4];
    Serial.printf("  v0: voc=%.3f nh3=%.3f hcho=%.3f  b0: h2s=%.2f etoh=%.2f\n",
                  v0_voc, v0_nh3, v0_hcho, b0_h2s, b0_etoh);
}

static void handleButton() {
    bool b = digitalRead(PIN_BUTTON);
    if (lastButton == HIGH && b == LOW) {
        delay(50);                                 // debounce
        measuring = !measuring;
        if (measuring) {
            captureBaselines();
            feat_reset(); ema_init = false; window_idx = 0; lastMeasure = 0;
            digitalWrite(PIN_LED, HIGH);
            Serial.println("=== MEASURING ===  (one prediction per 60 s)");
        } else {
            digitalWrite(PIN_LED, LOW);
            Serial.println("=== STOPPED ===  press button to recalibrate + start");
        }
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
    Serial.println();
    Serial.println("ESP32 blood-detect ready");
    Serial.printf("model: %d trees | %d features | %d s window | EMA alpha=%.2f | HDC %s\n",
                  XGB_N_TREES, XGB_N_FEATURES, WIN_SIZE, EMA_ALPHA, hdcReady ? "ok" : "MISSING");
    Serial.println("Warm up, place in clean air, then press the button to start.");
}

void loop() {
    handleButton();
    if (!measuring) { delay(20); return; }

    if (millis() - lastMeasure < 1000) { delay(5); return; }   // 1 Hz
    lastMeasure = millis();

    float raw[N_CHANNELS];
    read_sensors(raw);

    if (feat_push(raw)) {                          // non-overlapping 60 s window done
        window_idx++;
        if (window_idx <= WARMUP_WINDOWS) {
            Serial.printf("[window %d] warm-up — skipped\n", window_idx);
            return;
        }
        float feat[N_SELECTED];
        feat_window(feat);

        float proba[XGB_N_CLASSES];
        xgb_predict_proba(feat, proba);

        if (!ema_init) { for (int c = 0; c < XGB_N_CLASSES; c++) ema[c] = proba[c]; ema_init = true; }
        else { for (int c = 0; c < XGB_N_CLASSES; c++) ema[c] = EMA_ALPHA * proba[c] + (1 - EMA_ALPHA) * ema[c]; }

        int pred = 0;
        for (int c = 1; c < XGB_N_CLASSES; c++) if (ema[c] > ema[pred]) pred = c;

        Serial.printf("[window %d] %-8s  %3.0f%%   (base %.2f / sweat %.2f / blood %.2f)\n",
                      window_idx, CLASS_NAME[pred], ema[pred] * 100.0f, ema[0], ema[1], ema[2]);

        // Blink LED twice when blood is detected
        if (pred == 2) { for (int i = 0; i < 4; i++) { digitalWrite(PIN_LED, i & 1); delay(80); } digitalWrite(PIN_LED, HIGH); }
    }
}
