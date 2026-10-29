"""Paper A - diagnose source confounding in the naive three-class result.

Tests (all with the FAIR feature set, K-command truncation):
  T1  per-corpus distributions of the top features (frac_chain, frac_pipe, iv_min, frac_iv_lt1)
  T2  source identification WITHIN one class: DTU-agent vs AIPot-agent. Both are agents;
      high AUROC here means the features encode the corpus/harness, not agent behaviour.
  T3  cross-corpus transfer: train bot/human/agent with agents from DTU only, test the
      agent class on AIPot (frontier models). Collapse => the 'agent' signal does not transfer.
  T4  harness floor: share of inter-command intervals below the DTU fixed read-timeout.
"""
import json, os, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from a_data import read
import a_feat
from a_clf import SOURCE_LEAKS, K, tpr_at_fpr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RNG = np.random.default_rng(1)


def get(name, cap=None):
    L = [s for s in read(name) if len(s["cmds"]) >= K]
    if cap and len(L) > cap:
        L = [L[i] for i in RNG.choice(len(L), cap, replace=False)]
    return L


def fm(S):
    X, names = a_feat.matrix(S, early=K)
    keep = [i for i, n in enumerate(names) if n not in SOURCE_LEAKS]
    return X[:, keep], [names[i] for i in keep]


def XGB():
    return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                             eval_metric="logloss", tree_method="hist", n_jobs=8, random_state=0)


out = {}
C = {"cyberlab(bot)": get("cyberlab", 3000), "kypo(human)": get("kypo"), "dtu(agent)": get("dtu", 3000), "aipot(agent)": get("aipot")}

# ---- T1 ----
print("T1 per-corpus feature medians [q10, q90]:")
t1 = {}
for name, S in C.items():
    X, names = fm(S)
    row = {}
    for f in ("frac_chain", "frac_pipe", "iv_min", "iv_med", "frac_iv_lt1", "len_mean"):
        v = X[:, names.index(f)]
        row[f] = [round(float(np.percentile(v, 10)), 3), round(float(np.median(v)), 3), round(float(np.percentile(v, 90)), 3)]
    t1[name] = row
    print(f"  {name:14s}", row)
out["T1"] = t1

# ---- T4 harness floor ----
dtu_iv = np.concatenate([np.diff([c[0] for c in s["cmds"][:K]]) for s in C["dtu(agent)"]])
dtu_exec = np.array([c[3] - c[0] for s in C["dtu(agent)"] for c in s["cmds"][:K] if c[3] is not None])
out["T4"] = {"dtu_exec_median_s": float(np.median(dtu_exec)), "dtu_iv_q01": float(np.percentile(dtu_iv, 1)),
             "dtu_share_iv_below_1p2s": float(np.mean(dtu_iv < 1.2)),
             "bot_share_iv_below_1p2s": float(np.mean(np.concatenate([np.diff([c[0] for c in s["cmds"][:K]]) for s in C["cyberlab(bot)"]]) < 1.2))}
print("\nT4 harness floor:", {k: round(v, 3) for k, v in out["T4"].items()})

# ---- T2 source identification within the agent class ----
A, B = C["dtu(agent)"], C["aipot(agent)"]
S = A + B
X, names = fm(S)
y = np.array([0] * len(A) + [1] * len(B))
g = np.array([s["group"] for s in S])           # hold out whole models
P = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, y, g):
    if len(set(y[tr])) < 2:
        continue
    w = np.where(y[tr] == 1, len(tr) / (2 * y[tr].sum()), len(tr) / (2 * (1 - y[tr]).sum()))
    P[te] = XGB().fit(X[tr], y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
auc_src = float(roc_auc_score(y, P))
out["T2"] = {"dtu_vs_aipot_auc_leave_model_out": auc_src}
print(f"\nT2 DTU-agent vs AIPot-agent (both agents), leave-model-out AUROC = {auc_src:.3f}"
      "   (0.5 would mean the features ignore the corpus)")

# ---- T3 cross-corpus transfer of the agent class ----
train = C["cyberlab(bot)"] + C["kypo(human)"] + C["dtu(agent)"]
lab = {"cyberlab": 0, "kypo": 1, "dtu": 2}
Xtr, _ = fm(train); ytr = np.array([lab[s["corpus"]] for s in train])
w = np.ones(len(ytr))
for i in range(3):
    w[ytr == i] = len(ytr) / (3 * (ytr == i).sum())
m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8,
                      eval_metric="mlogloss", tree_method="hist", n_jobs=8, random_state=0).fit(Xtr, ytr, sample_weight=w)
Xa, _ = fm(B)
pa = m.predict_proba(Xa)
pred = pa.argmax(1)
dist = {c: float(np.mean(pred == i)) for i, c in enumerate(["bot", "human", "agent"])}
out["T3"] = {"aipot_predicted_as": dist, "aipot_recall_agent": dist["agent"]}
print(f"\nT3 trained with DTU agents only -> AIPot agents predicted as: { {k: round(v,3) for k,v in dist.items()} }")
by_model = collections.defaultdict(list)
for s, p in zip(B, pred):
    by_model[s["group"]].append(p == 2)
out["T3"]["recall_by_model"] = {k: float(np.mean(v)) for k, v in by_model.items()}
print("   agent recall by AIPot model:", {k: round(float(np.mean(v)), 3) for k, v in by_model.items()})

json.dump(out, open(os.path.join(RES, "a_confound.json"), "w"), indent=1)
print("\nsaved a_confound.json")
