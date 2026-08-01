// Real-time (causal) replicas of the offline training filters, plus baseline capture.
//
// Training chain (mems_filter.ipynb / spec_filter.ipynb) and what runs here:
//
//   MEMS voc/nh3/hcho : EMA span 30                      -> exact match (EMA is causal)
//   SPEC h2s/etoh     : Savitzky-Golay(121,3) -> Butterworth(order 4, fc 0.005 Hz)
//                                                        -> Butterworth only (see note)
//   rh_pct            : EMA span 60                      -> exact match
//
// NOTE on SPEC: the Savitzky-Golay stage is a *centered* filter — it needs 60 future
// samples, so it cannot run in real time. It is omitted here. The Butterworth stage is
// far more aggressive (fc 0.005 Hz ~ 200 s time constant) and dominates the result, so
// the causal output tracks the training output closely after settling. The Butterworth
// itself is reproduced exactly: same order-4 sections, applied forward only (lfilter),
// which is what the notebook does.
//
// Baseline (button press) mirrors the offline correction:
//   MEMS : least-squares linear trend over the captured history window, subtracted and
//          clipped at 0 — the drift correction from mems_filter.ipynb.
//   SPEC : constant offset (mean of the tail of the history), subtracted and clipped at 0
//          — the stabilization-point offset from spec_filter.ipynb.
//   rh   : uncorrected, used as-is.
#ifndef FILTERS_H
#define FILTERS_H
#include <math.h>
#include "feature_spec.h"

// ── Channel indices (must match feature_spec.h channel order) ────────────────
#define CH_VOC   0
#define CH_NH3   1
#define CH_HCHO  2
#define CH_H2S   3
#define CH_ETOH  4
#define CH_RH    5

// ── Tunables ─────────────────────────────────────────────────────────────────
#define MEMS_EMA_SPAN     30      // mems_filter FILTER_CONFIG "span"
#define RH_EMA_SPAN       60      // spec_filter RH_EMA_SPAN
#define BASELINE_HIST_S   600     // history kept for the baseline fit (10 min)
#define BASELINE_MIN_S    120     // refuse to capture a baseline with less than this
#define SPEC_OFFSET_TAIL  120     // SPEC offset = mean of the last N s of history

static const float MEMS_EMA_ALPHA = 2.0f / (MEMS_EMA_SPAN + 1.0f);   // 0.06452
static const float RH_EMA_ALPHA   = 2.0f / (RH_EMA_SPAN   + 1.0f);   // 0.03279

// ── Butterworth order 4, fc = 0.005 Hz, fs = 1 Hz, as 2 biquad sections ──────
// scipy.signal.butter(4, 0.005/0.5, btype="low", output="sos"). Double state: the
// first section's gain is ~5.8e-08 and float32 loses it.
typedef struct { double b0, b1, b2, a1, a2; } Biquad;
static const Biquad BW_SOS[2] = {
    {5.8451424331e-08, 1.1690284866e-07, 5.8451424331e-08, -1.9426382305, 0.94359727847},
    {1.0,              2.0,              1.0,              -1.9752696349, 0.97624479236},
};
typedef struct { double z1[2], z2[2]; } BwState;   // transposed direct-form II, per section

static void bw_reset(BwState* st, double x0) {
    // Seed to steady state at x0 so the filter starts at the signal level instead of 0
    // (equivalent to scipy's lfilter_zi * x[0] initialization used in the notebook).
    for (int s = 0; s < 2; s++) {
        const Biquad* q = &BW_SOS[s];
        double g = (q->b0 + q->b1 + q->b2) / (1.0 + q->a1 + q->a2);   // DC gain
        double y = g * x0;
        st->z1[s] = q->b1 * x0 - q->a1 * y + q->b2 * x0 - q->a2 * y;
        st->z2[s] = q->b2 * x0 - q->a2 * y;
        x0 = y;
    }
}

static double bw_step(BwState* st, double x) {
    for (int s = 0; s < 2; s++) {
        const Biquad* q = &BW_SOS[s];
        double y = q->b0 * x + st->z1[s];
        st->z1[s] = q->b1 * x - q->a1 * y + st->z2[s];
        st->z2[s] = q->b2 * x - q->a2 * y;
        x = y;
    }
    return x;
}

// ── Filter state ─────────────────────────────────────────────────────────────
static float   ema_state[N_CHANNELS];      // MEMS + rh
static BwState bw_state[2];                // [0]=h2s, [1]=etoh
static bool    filt_primed = false;

// ── Baseline history (ring buffer of filtered values) ────────────────────────
static float hist_buf[N_CHANNELS][BASELINE_HIST_S];
static long  hist_t0[BASELINE_HIST_S];     // sample time (s) for each slot
static int   hist_head = 0;                // next write slot
static int   hist_n    = 0;                // valid samples (saturates at BASELINE_HIST_S)

// ── Captured baseline ────────────────────────────────────────────────────────
static bool  baseline_ready = false;
static float base_slope[N_CHANNELS];       // MEMS: V per second
static float base_icept[N_CHANNELS];       // MEMS: V at t=0;  SPEC: constant offset

static void filters_reset(void) {
    filt_primed = false;
    hist_head = hist_n = 0;
    baseline_ready = false;
    for (int c = 0; c < N_CHANNELS; c++) {
        ema_state[c] = 0; base_slope[c] = 0; base_icept[c] = 0;
    }
}

// Filter one raw sample in place: raw[N_CHANNELS] -> filtered values.
static void filters_apply(const float* raw, float* out) {
    if (!filt_primed) {                                    // seed from the first sample
        for (int c = 0; c < N_CHANNELS; c++) ema_state[c] = raw[c];
        bw_reset(&bw_state[0], raw[CH_H2S]);
        bw_reset(&bw_state[1], raw[CH_ETOH]);
        filt_primed = true;
    }
    // MEMS: causal EMA, span 30
    for (int c = CH_VOC; c <= CH_HCHO; c++) {
        ema_state[c] += MEMS_EMA_ALPHA * (raw[c] - ema_state[c]);
        out[c] = ema_state[c];
    }
    // SPEC: causal Butterworth
    out[CH_H2S]  = (float)bw_step(&bw_state[0], raw[CH_H2S]);
    out[CH_ETOH] = (float)bw_step(&bw_state[1], raw[CH_ETOH]);
    // Humidity: causal EMA, span 60
    ema_state[CH_RH] += RH_EMA_ALPHA * (raw[CH_RH] - ema_state[CH_RH]);
    out[CH_RH] = ema_state[CH_RH];
}

// Append a filtered sample (taken at time t_s seconds) to the baseline history.
static void filters_push_history(const float* filt, long t_s) {
    for (int c = 0; c < N_CHANNELS; c++) hist_buf[c][hist_head] = filt[c];
    hist_t0[hist_head] = t_s;
    hist_head = (hist_head + 1) % BASELINE_HIST_S;
    if (hist_n < BASELINE_HIST_S) hist_n++;
}

// Fit the baseline from the history collected so far. Returns 0 if there is not
// enough history yet (nothing is changed in that case).
static int filters_capture_baseline(void) {
    if (hist_n < BASELINE_MIN_S) return 0;

    // MEMS: least-squares line over the whole history window (drift trend)
    for (int c = CH_VOC; c <= CH_HCHO; c++) {
        double st = 0, sv = 0, stv = 0, stt = 0;
        for (int i = 0; i < hist_n; i++) {
            int k = (hist_head - hist_n + i + BASELINE_HIST_S) % BASELINE_HIST_S;
            double t = (double)hist_t0[k], v = hist_buf[c][k];
            st += t; sv += v; stv += t * v; stt += t * t;
        }
        double den = (double)hist_n * stt - st * st;
        double a = (den != 0.0) ? ((double)hist_n * stv - st * sv) / den : 0.0;
        double b = (sv - a * st) / (double)hist_n;
        base_slope[c] = (float)a;
        base_icept[c] = (float)b;
    }
    // SPEC: constant offset = mean of the last SPEC_OFFSET_TAIL samples
    int tail = (hist_n < SPEC_OFFSET_TAIL) ? hist_n : SPEC_OFFSET_TAIL;
    for (int c = CH_H2S; c <= CH_ETOH; c++) {
        double s = 0;
        for (int i = 0; i < tail; i++) {
            int k = (hist_head - tail + i + BASELINE_HIST_S) % BASELINE_HIST_S;
            s += hist_buf[c][k];
        }
        base_slope[c] = 0.0f;
        base_icept[c] = (float)(s / tail);
    }
    base_slope[CH_RH] = 0.0f;                  // humidity used as-is
    base_icept[CH_RH] = 0.0f;
    baseline_ready = true;
    return 1;
}

// Subtract the captured baseline from a filtered sample taken at time t_s.
static void filters_correct(const float* filt, long t_s, float* out) {
    for (int c = 0; c < N_CHANNELS; c++) {
        if (c == CH_RH) { out[c] = filt[c]; continue; }        // humidity uncorrected
        float base = base_slope[c] * (float)t_s + base_icept[c];
        float v = filt[c] - base;
        out[c] = (v < 0.0f) ? 0.0f : v;                        // clip(lower=0), as in training
    }
}
#endif
