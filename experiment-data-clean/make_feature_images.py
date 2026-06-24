"""
Convert each 60 s window into a 2-D greyscale image (TODO: 2-D image models).

Layout: rows = sensor channels, columns = time (seconds within the window).
Each pixel is the channel value, normalized to a grey level [0, 255].

Normalization is GLOBAL per channel (fit across all windows) so that the same
physical level maps to the same grey value in every image — this preserves the
absolute-level differences between classes (e.g. NH3 is higher for blood).
An optional signed-log step compresses the wide dynamic range so small responses
(low blood content) are not crushed by large pure-blood values.

Output:
  feature_images/<label>/<session>_w<idx>.png   one image per window
  feature_images/dataset.npz                     X (N, C, WIN) uint8, y, sessions, channels

Run:  python make_feature_images.py
Needs: processed/   (and combined_no_mix140.ipynb for the label windows)
"""
import os, json
import numpy as np
import pandas as pd
from PIL import Image

HERE      = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "processed")
OUTDIR    = os.path.join(HERE, "feature_images")

MEMS = ["voc", "nh3", "hcho"]; SPEC = ["h2s_ppm", "etoh_ppm"]; ENV = ["temp_C", "rh_pct"]
ALL  = MEMS + SPEC + ENV
EXCLUDE = ("mix_140_10",)
LB, LS, LBL = 0, 1, 2
LABEL_NAME = {LB: "baseline", LS: "sweat", LBL: "blood"}

WIN_SIZE    = 60                                   # seconds per window
IMAGE_COLS  = ["voc", "nh3", "hcho", "h2s_ppm", "etoh_ppm", "rh_pct"]   # image rows
SIGNED_LOG  = True                                 # compress dynamic range before normalizing
UPSCALE     = 4                                    # nearest-neighbour upscale for saved PNGs (viewing only)

def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))

# ── Load + merge all sessions (excluding mix_140_10) ──────────────────────────
def _load(prefix):
    out = {}
    for f in sorted(os.listdir(PROCESSED)):
        if f.startswith(prefix) and f.endswith(".pkl") and not any(x in f for x in EXCLUDE):
            out[f[len(prefix):-4]] = pd.read_pickle(os.path.join(PROCESSED, f))
    return out

mems, spec = _load("mems_"), _load("spec_")
datasets = {}
for k in sorted(mems):
    if k not in spec: continue
    m, s = mems[k], spec[k]
    merged = pd.merge(m, s[["elapsed_s"] + SPEC], on="elapsed_s", how="inner")
    if len(merged) < max(len(m), len(s)):
        m_s = m.sort_values("elapsed_s").reset_index(drop=True)
        s_s = s[["elapsed_s"] + SPEC].sort_values("elapsed_s").reset_index(drop=True)
        merged = pd.merge_asof(m_s, s_s, on="elapsed_s", tolerance=0.5,
                               direction="nearest").dropna(subset=SPEC).reset_index(drop=True)
    datasets[k] = merged[["elapsed_s"] + ALL].reset_index(drop=True)

# ── Labels (from the notebook's SAMPLE_WINDOWS) ───────────────────────────────
nb = json.load(open(os.path.join(HERE, "combined_no_mix140.ipynb")))
src = next("".join(c["source"]) for c in nb["cells"] if c.get("id") == "label_code")
i = src.index("SAMPLE_WINDOWS"); j = src.index("}", i) + 1
ns = {"LABEL_BASELINE": LB, "LABEL_SWEAT": LS, "LABEL_BLOOD": LBL}; exec(src[i:j], ns)
SW = ns["SAMPLE_WINDOWS"]
for k, df in datasets.items():
    df["label"] = LB
    for t0, t1, lbl in SW.get(k, []):
        t = df["elapsed_s"]
        df.loc[(t >= t0) & (t <= (t1 if t1 is not None else t.iloc[-1])), "label"] = lbl

# ── Cut non-overlapping windows -> (C, WIN) arrays + labels ───────────────────
images, labels, tags = [], [], []
for sess, df in datasets.items():
    for lbl in [LB, LS, LBL]:
        seg = df[df["label"] == lbl].reset_index(drop=True)
        for q in range(len(seg) // WIN_SIZE):
            w = seg.iloc[q * WIN_SIZE:(q + 1) * WIN_SIZE]
            arr = w[IMAGE_COLS].to_numpy(dtype=np.float32).T   # (C, WIN)
            images.append(arr); labels.append(lbl); tags.append(f"{sess}_w{q}")
X = np.stack(images)                                            # (N, C, WIN)
y = np.array(labels)
print(f"Windows: {len(X)}  | image shape (C x WIN): {X.shape[1]} x {X.shape[2]}")

# ── Global per-channel normalization (preserves cross-window level) ───────────
Xn = signed_log(X) if SIGNED_LOG else X.copy()
# robust min/max per channel via 1st/99th percentile over all windows+time
lo = np.percentile(Xn, 1,  axis=(0, 2), keepdims=True)
hi = np.percentile(Xn, 99, axis=(0, 2), keepdims=True)
Xn = np.clip((Xn - lo) / (hi - lo + 1e-9), 0, 1)
Ximg = (Xn * 255).astype(np.uint8)                             # (N, C, WIN) greyscale

# ── Save ──────────────────────────────────────────────────────────────────────
for name in LABEL_NAME.values():
    os.makedirs(os.path.join(OUTDIR, name), exist_ok=True)
for img, lbl, tag in zip(Ximg, y, tags):
    im = Image.fromarray(img, mode="L")
    if UPSCALE > 1:
        im = im.resize((img.shape[1] * UPSCALE, img.shape[0] * UPSCALE), Image.NEAREST)
    im.save(os.path.join(OUTDIR, LABEL_NAME[lbl], f"{tag}.png"))
np.savez_compressed(os.path.join(OUTDIR, "dataset.npz"),
                    X=Ximg, y=y, sessions=np.array(tags), channels=np.array(IMAGE_COLS))

n_by = {LABEL_NAME[l]: int((y == l).sum()) for l in LABEL_NAME}
print(f"Saved {len(Ximg)} images to {OUTDIR}/  ({n_by})")
print(f"  channels (rows): {IMAGE_COLS}")
print(f"  dataset.npz: X{Ximg.shape} uint8, y, sessions, channels")
