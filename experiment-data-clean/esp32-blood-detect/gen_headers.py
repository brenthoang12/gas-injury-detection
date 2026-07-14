#!/usr/bin/env python3
"""Generate feature_spec.h and xgb_model.h for the ESP32 firmware from the
deployed XGBoost Run 6 export in ../model_export/.

Reproducible: re-export from combined_no_mix140.ipynb, then re-run this. It also
runs a host parity check of the emitted C logic against xgboost predict_proba.

Run:  ../../.venv/bin/python3 gen_headers.py
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, "..", "model_export")

# ── Load the deployed model + export metadata ────────────────────────────────
model = XGBClassifier()
model.load_model(os.path.join(EXPORT, "xgb_run6.json"))
bst = model.get_booster()
meta = json.load(open(os.path.join(EXPORT, "export_meta.json")))
RUN6_COLS = meta["run6_cols"]
WIN = meta["window_size"]
FEATURE_COLS = json.load(open(os.path.join(EXPORT, "feature_cols.json")))
name2idx = {n: i for i, n in enumerate(RUN6_COLS)}
NF = len(RUN6_COLS)
N_CLASSES = 3

# best_iteration: predict_proba only uses trees up to here, so the C model must too
best_it = None
for obj in (model, bst):
    bi = getattr(obj, "best_iteration", None)
    if bi is not None:
        best_it = int(bi)
        break
print(f"features={NF}  engineered={len(FEATURE_COLS)}  best_iteration={best_it}")

# ── Engineered-column spec (channel, kind, window) from the FEATURE_COLS names ─
CH = ["voc", "nh3", "hcho", "h2s_ppm", "etoh_ppm", "rh_pct"]
ch2i = {c: i for i, c in enumerate(CH)}
ROLL_W = [15, 30, 60]
KIND = {"raw": 0, "roc": 1, "acc": 2, "roll_mean": 3, "roll_std": 4, "roll_roc": 5}


def parse_eng(name):
    for c in sorted(CH, key=len, reverse=True):
        if name == c:
            return ch2i[c], KIND["raw"], 0
        if name.startswith(c + "_"):
            suf = name[len(c) + 1:]
            if suf == "roc":
                return ch2i[c], KIND["roc"], 0
            if suf == "acc":
                return ch2i[c], KIND["acc"], 0
            for w in ROLL_W:
                if suf == f"roll_mean_{w}":
                    return ch2i[c], KIND["roll_mean"], w
                if suf == f"roll_std_{w}":
                    return ch2i[c], KIND["roll_std"], w
                if suf == f"roll_roc_{w}":
                    return ch2i[c], KIND["roll_roc"], w
    raise ValueError(f"cannot parse engineered column: {name}")


ENG = [parse_eng(n) for n in FEATURE_COLS]

# ── Selected-feature spec (engineered index, aggregate) from RUN6_COLS names ──
AGG = {"mean": 0, "std": 1, "max": 2, "slope": 3}


def parse_sel(col):
    for a, ai in AGG.items():
        if col.endswith("_" + a):
            eng = col[:-(len(a) + 1)]
            return FEATURE_COLS.index(eng), ai
    raise ValueError(f"cannot parse feature column: {col}")


SEL = [parse_sel(c) for c in RUN6_COLS]

# ── Parse the boosted trees (only the rounds predict_proba actually uses) ─────
dump = bst.get_dump(dump_format="json")
n_trees = (best_it + 1) * N_CLASSES if best_it is not None else len(dump)
dump = dump[:n_trees]

all_nodes = []   # per-tree list of node dicts, in local-index order (root = 0)
offsets = [0]
for tj in dump:
    tree = json.loads(tj)
    flat = {}

    def collect(nd):
        flat[nd["nodeid"]] = nd
        for ch in nd.get("children", []):
            collect(ch)

    collect(tree)

    order, idmap = [], {}

    def dfs(nid):
        idmap[nid] = len(order)
        order.append(nid)
        nd = flat[nid]
        if "children" in nd:
            dfs(nd["yes"])
            dfs(nd["no"])

    dfs(0)

    nodes = []
    for nid in order:
        nd = flat[nid]
        if "leaf" in nd:
            nodes.append(dict(feature=-1, thr=0.0, yes=0, no=0, miss=0,
                              leaf=float(nd["leaf"]), is_leaf=1))
        else:
            sp = nd["split"]
            fi = name2idx.get(sp)
            if fi is None:
                fi = int(sp[1:]) if sp[0] == "f" else name2idx[sp]
            nodes.append(dict(feature=fi, thr=float(nd["split_condition"]),
                              yes=idmap[nd["yes"]], no=idmap[nd["no"]],
                              miss=idmap[nd["missing"]], leaf=0.0, is_leaf=0))
    all_nodes.append(nodes)
    offsets.append(offsets[-1] + len(nodes))

N_NODES = offsets[-1]
print(f"trees={len(all_nodes)}  nodes={N_NODES}")

# ── Host replica of the C traversal, to fix the base margins and check parity ─
def treesum(x):
    marg = np.zeros(N_CLASSES)
    for t, nodes in enumerate(all_nodes):
        k = 0
        while not nodes[k]["is_leaf"]:
            nd = nodes[k]
            xi = x[nd["feature"]]
            if np.isnan(xi):
                k = nd["miss"]
            elif xi < nd["thr"]:
                k = nd["yes"]
            else:
                k = nd["no"]
        marg[t % N_CLASSES] += nodes[k]["leaf"]
    return marg


tw = pd.read_pickle(os.path.join(EXPORT, "test_windows.pkl"))
Xdf = tw[RUN6_COLS]
X = Xdf.to_numpy(dtype=np.float32)

ts = np.array([treesum(row) for row in X])
it_range = (0, best_it + 1) if best_it is not None else (0, 0)
marg_xgb = bst.predict(xgb.DMatrix(Xdf), output_margin=True, iteration_range=it_range)
BASE = (marg_xgb - ts).mean(axis=0)


def softmax(m):
    e = np.exp(m - m.max())
    return e / e.sum()


my_proba = np.array([softmax(treesum(row) + BASE) for row in X])
ref_proba = model.predict_proba(Xdf)
maxdiff = np.abs(my_proba - ref_proba).max()
print(f"PARITY  max|C_proba - xgb_proba| = {maxdiff:.3e}   base={BASE}")
assert maxdiff < 1e-4, "parity check failed — do not ship these headers"

# ── Emit feature_spec.h ──────────────────────────────────────────────────────
def ff(v):
    s = f"{float(v):.9g}"                       # compact but full float32 precision
    if not any(ch in s for ch in ".eEnN"):      # "59" -> "59.0" so "59f" is not invalid C
        s += ".0"
    return s + "f"


fs = []
fs.append("// Auto-generated feature spec — XGBoost Run 6. Do not edit.")
fs.append("#ifndef FEATURE_SPEC_H")
fs.append("#define FEATURE_SPEC_H\n")
fs.append(f"#define N_CHANNELS {len(CH)}")
fs.append(f"#define N_ENG {len(FEATURE_COLS)}        // engineered columns per sample")
fs.append(f"#define N_SELECTED {NF}   // model input length (Run 6 uses all features)")
fs.append(f"#define WIN_SIZE {WIN}")
fs.append(f"#define MAX_ROLL {max(ROLL_W)}\n")
fs.append("// channel order: " + ", ".join(CH))
fs.append(f"static const int ROLL_W[{len(ROLL_W)}] = {{{','.join(map(str, ROLL_W))}}};\n")
fs.append("// engineered column: {channel_index, kind, window}")
fs.append("//   kind: 0=raw 1=roc 2=acc 3=roll_mean 4=roll_std 5=roll_roc")
fs.append("typedef struct { unsigned char ch; unsigned char kind; unsigned short win; } EngSpec;")
fs.append("static const EngSpec ENG_SPEC[N_ENG] = {")
fs.append(",\n".join(f"  {{{c},{k},{w}}}" for (c, k, w) in ENG))
fs.append("};\n")
fs.append("// selected feature: {engineered_col_index, aggregate}  agg: 0=mean 1=std 2=max 3=slope")
fs.append("typedef struct { unsigned short eng; unsigned char agg; } SelSpec;")
fs.append("static const SelSpec SEL_SPEC[N_SELECTED] = {")
fs.append(",\n".join(f"  {{{e},{a}}}" for (e, a) in SEL))
fs.append("};\n")
fs.append("#endif")
open(os.path.join(HERE, "feature_spec.h"), "w").write("\n".join(fs) + "\n")

# ── Emit xgb_model.h ─────────────────────────────────────────────────────────
node_lines = []
for nodes in all_nodes:
    for n in nodes:
        if n["is_leaf"]:
            node_lines.append(f"  {{-1,0.0f,0,0,0,1,{ff(n['leaf'])}}}")
        else:
            node_lines.append(f"  {{{n['feature']},{ff(n['thr'])},"
                              f"{n['yes']},{n['no']},{n['miss']},0,0.0f}}")

mh = []
mh.append("// Auto-generated from model_export/ — XGBoost Run 6. Do not edit by hand.")
mh.append("#ifndef XGB_MODEL_H")
mh.append("#define XGB_MODEL_H")
mh.append("#include <math.h>\n")
mh.append(f"#define XGB_N_FEATURES {NF}")
mh.append(f"#define XGB_N_CLASSES {N_CLASSES}")
mh.append(f"#define XGB_N_TREES {len(all_nodes)}")
mh.append(f"#define XGB_N_NODES {N_NODES}\n")
mh.append("typedef struct { short feature; float threshold; short yes; short no; "
          "short miss; unsigned char is_leaf; float leaf; } XgbNode;\n")
mh.append("static const XgbNode XGB_NODES[XGB_N_NODES] = {")
mh.append(",\n".join(node_lines))
mh.append("};\n")
mh.append("static const int XGB_TREE_OFFSET[XGB_N_TREES+1] = {" +
          ",".join(map(str, offsets)) + "};\n")
mh.append("static const float XGB_BASE[XGB_N_CLASSES] = {" +
          ",".join(ff(v) for v in BASE) + "};\n")
mh.append("""// Fill `proba` (length XGB_N_CLASSES) from feature vector `x` (length XGB_N_FEATURES).
static void xgb_predict_proba(const float* x, float* proba) {
    float margin[XGB_N_CLASSES];
    for (int c = 0; c < XGB_N_CLASSES; c++) margin[c] = XGB_BASE[c];
    for (int t = 0; t < XGB_N_TREES; t++) {
        int i = XGB_TREE_OFFSET[t];
        for (;;) {
            const XgbNode* n = &XGB_NODES[i];
            if (n->is_leaf) { margin[t % XGB_N_CLASSES] += n->leaf; break; }
            float xi = x[n->feature];
            short j = isnan(xi) ? n->miss : (xi < n->threshold ? n->yes : n->no);
            i = XGB_TREE_OFFSET[t] + j;
        }
    }
    float m = margin[0]; for (int c = 1; c < XGB_N_CLASSES; c++) if (margin[c] > m) m = margin[c];
    float s = 0.0f;
    for (int c = 0; c < XGB_N_CLASSES; c++) { proba[c] = expf(margin[c] - m); s += proba[c]; }
    for (int c = 0; c < XGB_N_CLASSES; c++) proba[c] /= s;
}
#endif""")
open(os.path.join(HERE, "xgb_model.h"), "w").write("\n".join(mh) + "\n")

print("wrote feature_spec.h and xgb_model.h")
