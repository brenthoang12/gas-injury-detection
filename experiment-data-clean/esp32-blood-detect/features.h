// Causal feature pipeline — mirrors the Python training pipeline.
// Per 1 Hz sample: update raw history, compute the 72 engineered columns.
// Every WIN_SIZE samples (non-overlapping window): aggregate to the 173 model features.
#ifndef FEATURES_H
#define FEATURES_H
#include <math.h>
#include "feature_spec.h"

static float raw_hist[N_CHANNELS][MAX_ROLL];   // last MAX_ROLL raw values per channel
static float roc_hist[N_CHANNELS][MAX_ROLL];   // last MAX_ROLL rate-of-change values
static float eng_hist[N_ENG][WIN_SIZE];        // last WIN_SIZE engineered-column values
static float prev_raw[N_CHANNELS];
static float prev_roc[N_CHANNELS];
static long  feat_count = 0;

static void feat_reset(void) {
    feat_count = 0;
    for (int c = 0; c < N_CHANNELS; c++) {
        prev_raw[c] = 0; prev_roc[c] = 0;
        for (int i = 0; i < MAX_ROLL; i++) { raw_hist[c][i] = 0; roc_hist[c][i] = 0; }
    }
    for (int e = 0; e < N_ENG; e++)
        for (int i = 0; i < WIN_SIZE; i++) eng_hist[e][i] = 0;
}

// helpers over the last `nav` entries (tail) of a length-`len` buffer
static float tail_mean(const float* b, int len, int nav) {
    float s = 0; for (int i = len - nav; i < len; i++) s += b[i]; return s / nav;
}
static float tail_absmean(const float* b, int len, int nav) {
    float s = 0; for (int i = len - nav; i < len; i++) s += fabsf(b[i]); return s / nav;
}
static float tail_std(const float* b, int len, int nav) {     // sample std, ddof=1 (pandas rolling.std)
    if (nav < 2) return 0.0f;
    float m = tail_mean(b, len, nav), s = 0;
    for (int i = len - nav; i < len; i++) { float d = b[i] - m; s += d * d; }
    return sqrtf(s / (nav - 1));
}

// Push one 1 Hz sample (raw[N_CHANNELS], channel order per feature_spec.h).
// Returns 1 when a non-overlapping window has completed (call feat_window next).
static int feat_push(const float* raw) {
    float roc[N_CHANNELS];
    for (int c = 0; c < N_CHANNELS; c++) {
        roc[c] = (feat_count == 0) ? 0.0f : raw[c] - prev_raw[c];
        for (int i = 0; i < MAX_ROLL - 1; i++) {            // shift histories left
            raw_hist[c][i] = raw_hist[c][i + 1];
            roc_hist[c][i] = roc_hist[c][i + 1];
        }
        raw_hist[c][MAX_ROLL - 1] = raw[c];
        roc_hist[c][MAX_ROLL - 1] = roc[c];
    }

    int navail = (feat_count + 1 < MAX_ROLL) ? (int)(feat_count + 1) : MAX_ROLL;
    float engval[N_ENG];
    for (int e = 0; e < N_ENG; e++) {
        EngSpec sp = ENG_SPEC[e];
        int c = sp.ch, w = sp.win;
        int navw = (w == 0) ? navail : (navail < w ? navail : w);
        switch (sp.kind) {
            case 0: engval[e] = raw[c];                                   break; // raw
            case 1: engval[e] = roc[c];                                   break; // roc
            case 2: engval[e] = roc[c] - prev_roc[c];                     break; // acc
            case 3: engval[e] = tail_mean(raw_hist[c], MAX_ROLL, navw);   break; // roll_mean
            case 4: engval[e] = tail_std (raw_hist[c], MAX_ROLL, navw);   break; // roll_std
            case 5: engval[e] = tail_absmean(roc_hist[c], MAX_ROLL, navw); break;// roll_roc
        }
    }
    for (int c = 0; c < N_CHANNELS; c++) { prev_roc[c] = roc[c]; prev_raw[c] = raw[c]; }

    for (int e = 0; e < N_ENG; e++) {                       // push engineered values
        for (int i = 0; i < WIN_SIZE - 1; i++) eng_hist[e][i] = eng_hist[e][i + 1];
        eng_hist[e][WIN_SIZE - 1] = engval[e];
    }
    feat_count++;
    return (feat_count >= WIN_SIZE && feat_count % WIN_SIZE == 0) ? 1 : 0;
}

// Aggregate the current window into the 173-feature model input (out[N_SELECTED]).
static void feat_window(float* out) {
    for (int k = 0; k < N_SELECTED; k++) {
        SelSpec s = SEL_SPEC[k];
        const float* v = eng_hist[s.eng];
        int n = WIN_SIZE;
        float val;
        if (s.agg == 0) {                                   // mean
            float sum = 0; for (int i = 0; i < n; i++) sum += v[i]; val = sum / n;
        } else if (s.agg == 1) {                            // std (population, ddof=0 — np.nanstd)
            float m = 0; for (int i = 0; i < n; i++) m += v[i]; m /= n;
            float ss = 0; for (int i = 0; i < n; i++) { float d = v[i] - m; ss += d * d; }
            val = sqrtf(ss / n);
        } else if (s.agg == 2) {                            // max
            float mx = v[0]; for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i]; val = mx;
        } else {                                            // slope (least-squares over t=0..n-1)
            float st = 0, sv = 0, stv = 0, stt = 0;
            for (int i = 0; i < n; i++) { st += i; sv += v[i]; stv += (float)i * v[i]; stt += (float)i * i; }
            float den = (float)n * stt - st * st;
            val = (den != 0.0f) ? ((float)n * stv - st * sv) / den : 0.0f;
        }
        out[k] = val;
    }
}
#endif
