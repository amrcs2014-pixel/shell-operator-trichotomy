"""Paper A - first three-class experiment (bot / human / agent) on the public corpora.

Fairness controls applied here:
  * FAIR feature set only: features computable identically from every source
    (command timestamps + command strings). Source-availability indicators
    (pre, has_pre, has_outlen, corr_iv_prevout) are excluded - they identify the
    corpus, not the behaviour.
  * Every session truncated to the same first K commands, so turn caps (DTU 20/30)
    cannot leak through n_cmd.
  * Grouped splits: bot=source IP, human=trainee, agent=LLM model. Holding out whole
    models makes the agent test "unseen LLM".
"""
import json, os, sys, collections
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from a_data import read, seq_hash
import a_feat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
CLASSES = ["bot", "human", "agent"]
SOURCE_LEAKS = {"pre", "has_pre", "has_outlen", "corr_iv_prevout"}
K = int(os.environ.get("K", 10))          # commands per session (truncation)
MINC = K                                   # require at least K commands
RNG = np.random.default_rng(0)


def tpr_at_fpr(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def load(cap_agent=6000, cap_bot=6000):
    S = []
    for name, lab in (("cyberlab", "bot"), ("kypo", "human"), ("dtu", "agent"), ("aipot", "agent")):
        for s in read(name):
            if len(s["cmds"]) >= MINC:
                s["y"] = lab; S.append(s)
    by = collections.defaultdict(list)
    for s in S:
        by[s["y"]].append(s)
    out = []
    for lab, lst in by.items():
        cap = cap_agent if lab == "agent" else cap_bot if lab == "bot" else len(lst)
        if len(lst) > cap:
            idx = RNG.choice(len(lst), cap, replace=False); lst = [lst[i] for i in idx]
        out += lst
    return out


def fair_matrix(S):
    X, names = a_feat.matrix(S, early=K)
    keep = [i for i, n in enumerate(names) if n not in SOURCE_LEAKS]
    return X[:, keep], [names[i] for i in keep]


def run():
    S = load()
    y = np.array([CLASSES.index(s["y"]) for s in S])
    g = np.array([f'{s["y"]}:{s["group"]}' for s in S])
    print("sessions per class:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
          "| groups:", {c: len(set(g[y == i])) for i, c in enumerate(CLASSES)}, flush=True)
    X, names = fair_matrix(S)
    print(f"features ({len(names)}, K={K}):", names, flush=True)

    res = {"K": K, "n": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}, "features": names, "models": {}}
    gkf = GroupKFold(n_splits=5)
    for mname in ("xgb", "logreg"):
        P = np.zeros((len(y), 3))
        for tr, te in gkf.split(X, y, g):
            w = np.ones(len(tr))
            for i in range(3):                                   # inverse-frequency class weights
                w[y[tr] == i] = len(tr) / (3 * max(1, (y[tr] == i).sum()))
            if mname == "xgb":
                m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                                      colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist", n_jobs=8, random_state=0)
                m.fit(X[tr], y[tr], sample_weight=w)
            else:
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
                m.fit(X[tr], y[tr], logisticregression__sample_weight=w)
            P[te] = m.predict_proba(X[te])
        pred = P.argmax(1)
        r = {"macro_f1": float(f1_score(y, pred, average="macro"))}
        for i, c in enumerate(CLASSES):
            yy = (y == i).astype(int)
            r[f"auc_{c}"] = float(roc_auc_score(yy, P[:, i]))
            r[f"tpr1_{c}"] = tpr_at_fpr(yy, P[:, i], 0.01)
        cm = np.zeros((3, 3), int)
        for a, b in zip(y, pred):
            cm[a, b] += 1
        r["confusion"] = cm.tolist()
        res["models"][mname] = r
        print(f"\n[{mname}] grouped 5-fold (unseen IP / trainee / LLM):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    # feature importance (xgb, full fit) for a first look at what drives it
    w = np.ones(len(y))
    for i in range(3):
        w[y == i] = len(y) / (3 * (y == i).sum())
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                          eval_metric="mlogloss", tree_method="hist", n_jobs=8, random_state=0).fit(X, y, sample_weight=w)
    imp = sorted(zip(names, m.feature_importances_), key=lambda x: -x[1])
    res["importance"] = [(n, float(v)) for n, v in imp]
    print("\ntop features (gain):", [(n, round(float(v), 3)) for n, v in imp[:12]], flush=True)

    # ablation by feature family: timing-only vs structure-only
    TIMING = {n for n in names if n.startswith(("iv_", "frac_iv", "dur", "rate", "corr_iv"))}
    for fam, sel in (("timing_only", [i for i, n in enumerate(names) if n in TIMING]),
                     ("structure_only", [i for i, n in enumerate(names) if n not in TIMING])):
        P = np.zeros((len(y), 3))
        for tr, te in gkf.split(X, y, g):
            ww = np.ones(len(tr))
            for i in range(3):
                ww[y[tr] == i] = len(tr) / (3 * max(1, (y[tr] == i).sum()))
            mm = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                                   eval_metric="mlogloss", tree_method="hist", n_jobs=8, random_state=0).fit(X[tr][:, sel], y[tr], sample_weight=ww)
            P[te] = mm.predict_proba(X[te][:, sel])
        r = {"macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
             "auc_agent": float(roc_auc_score((y == 2).astype(int), P[:, 2]))}
        res[fam] = r
        print(f"[{fam}] ({len(sel)} feats):", {k: round(v, 3) for k, v in r.items()}, flush=True)

    json.dump(res, open(os.path.join(RES, f"a_clf_K{K}.json"), "w"), indent=1)
    print("\nsaved", os.path.join(RES, f"a_clf_K{K}.json"))


if __name__ == "__main__":
    run()
