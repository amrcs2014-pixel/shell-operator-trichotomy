"""Paper A - revised analyses (Oct 2026).

All detection results under ONE consistent, leak-free protocol. a_eval.json is left untouched.
  R0  benchmark sample identical to a_eval.load() (K=10, >=10 cmds, cap 4000/corpus, same seed)
  R1  feature inventory: every computed feature, the 4 excluded source features, constant n_cmd under K=10
  R2  the single-feature rule (median inter-command interval): learned 3-class decision regions
  R3  full LOCO table: {logistic, GBDT, GRU, rule1, TF-IDF, Reworr} x held agent corpus {dtu, aipot, own}.
      Non-agent sessions are split into 5 grouped folds; each fold is scored by a model that never saw it,
      so the 1%-FPR threshold is calibrated on held-out negatives (fixes negative-class leakage).
      Metrics: 3-class argmax agent recall and agent-vs-rest TPR@1%FPR, cluster-bootstrap CIs, plus the
      in-distribution grouped-CV reference for every model.
  R4  OOD probes for single-source negatives: novelty detector fit on bot+human only; unseen human
      sub-population (KYPO compromised-server shells) and unseen bot period (CyberLab 2020) held out.
  R5  scale sweep + adaptive attacks under ONE cross-fitted detector; white-box grid contains pure scaling.
  R6  mechanistic think-time table on the benchmark sample (incl. own), K-truncated, per model.
  R7  own-corpus prompting conditions (base / deceive / mimic / jitter).
  R8  timestamp resolution per corpus + robustness with all timestamps coarsened to 1 s.
Usage: python a_revision.py            (QUICK=1 for a fast smoke test on a subsample)
"""
import collections, functools, json, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import torch, torch.nn as nn

import a_feat
a_feat._lev_ratio = functools.lru_cache(maxsize=None)(a_feat._lev_ratio)          # timing transforms keep texts
a_feat._charclass_entropy = functools.lru_cache(maxsize=None)(a_feat._charclass_entropy)
import a_eval
from a_eval import XGB, weights, tpr_at, cluster_boot, SOURCE_LEAK, CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
K = a_eval.K
QUICK = os.environ.get("QUICK") == "1"
HELD = ["dtu", "aipot", "own"]
R = np.random.default_rng(7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------- R0 sample
S = a_eval.load()                                   # first consumer of a_eval.RNG -> identical sample
if QUICK:
    keep = sorted(R.choice(len(S), 1500, replace=False))
    S = [S[i] for i in keep]
y = np.array([CLASSES.index(s["y"]) for s in S])
corpus = np.array([s["corpus"] for s in S])
grp = np.array([f'{s["y"]}:{s["group"]}' for s in S])
X, names = a_eval.feat(S)
IV = names.index("iv_med")
out = {"K": K, "n_sessions": len(S), "per_corpus": {k: int(v) for k, v in collections.Counter(corpus).items()}}
log("sample", out["per_corpus"])

# ------------------------------------------------------------------------------------------- R1 features
allnames = list(a_feat.FEATURE_NAMES) if a_feat.FEATURE_NAMES else []
a_feat.matrix(S[:5], early=K); allnames = list(a_feat.FEATURE_NAMES)
const = [n for i, n in enumerate(names) if np.std(X[:, i]) == 0]
out["R1_features"] = {"n_computed": len(allnames), "computed": allnames, "excluded_source": sorted(SOURCE_LEAK),
                      "n_kept": len(names), "constant_under_K": const,
                      "n_informative": len(names) - len(const)}
log("R1", {k: v for k, v in out["R1_features"].items() if k.startswith("n_") or k == "constant_under_K"})


# ------------------------------------------------------------------------------------------- per-command sequences (GRU)
FLAG, TOKEN = a_feat.FLAG, a_feat.TOKEN
ASSIGN = re.compile(r"^\s*[A-Za-z_]\w*=")


def seq_of(s):
    cm = s["cmds"][:K]; t = [c[0] for c in cm]; tx = [c[1] or "" for c in cm]; rows = []
    for i in range(K):
        x = tx[i]; prev = tx[i - 1] if i else None
        iv = t[i] - t[i - 1] if i else 0.0
        rows.append([np.log1p(max(iv, 0.0)), len(x), len(TOKEN.findall(x)), "|" in x,
                     (";" in x or "&&" in x or "||" in x), (">" in x or "<" in x), bool(FLAG.search(x)),
                     len(FLAG.findall(x)), ("$(" in x or "`" in x), "/" in x, bool(ASSIGN.match(x)),
                     len(x.strip()) == 0, a_feat._charclass_entropy(x),
                     prev is not None and x == prev, a_feat._lev_ratio(prev, x) if prev is not None else 0.0])
    return np.array(rows, np.float32)


SEQ = np.stack([seq_of(s) for s in S])
log("sequences", SEQ.shape)


class GRUNet(nn.Module):
    def __init__(self, d, h=32, ncls=3):
        super().__init__(); self.gru = nn.GRU(d, h, batch_first=True); self.out = nn.Linear(h, ncls)

    def forward(self, x):
        _, hN = self.gru(x); return self.out(hN[-1])


class GRUClf:
    """Small GRU over the per-command feature stream; standardised inputs, class-weighted CE, fixed epochs."""
    def __init__(self, ncls=3, epochs=40, seed=0):
        self.ncls, self.epochs, self.seed = ncls, epochs, seed

    def fit(self, Xs, yy):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        d = Xs.shape[2]; flat = Xs.reshape(-1, d)
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        xt = torch.tensor((Xs - self.mu) / self.sd, dtype=torch.float32, device=DEV)
        yt = torch.tensor(yy, dtype=torch.long, device=DEV)
        cw = torch.tensor([len(yy) / (self.ncls * max(1, (yy == c).sum())) for c in range(self.ncls)],
                          dtype=torch.float32, device=DEV)
        self.net = GRUNet(d, ncls=self.ncls).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=cw); n = len(yy); g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            for b in torch.randperm(n, generator=g).split(256):
                opt.zero_grad(); loss = lossf(self.net(xt[b.to(DEV)]), yt[b.to(DEV)]); loss.backward(); opt.step()
        return self

    def predict_proba(self, Xs):
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor((Xs - self.mu) / self.sd, dtype=torch.float32, device=DEV)
            return torch.softmax(self.net(xt), 1).cpu().numpy()


DOCS = [" ".join((c[1] or "") for c in s["cmds"][:K]) for s in S]


class TfidfClf:
    def fit(self, idx, yy):
        self.v = TfidfVectorizer(max_features=3000, token_pattern=r"[^\s]+")
        Xt = self.v.fit_transform([DOCS[i] for i in idx])
        self.m = LogisticRegression(max_iter=2000).fit(Xt, yy, sample_weight=weights(yy, 3)); return self

    def predict_proba(self, idx):
        return self.m.predict_proba(self.v.transform([DOCS[i] for i in idx]))


def fit_predict(kind, tr, te_list):
    """Fit model `kind` on rows `tr` (3-class) and return predict_proba for each index array in te_list."""
    yt = y[tr]
    if kind == "gbdt":
        m = XGB().fit(X[tr], yt, sample_weight=weights(yt, 3)); return [m.predict_proba(X[te]) for te in te_list]
    if kind in ("logistic", "rule1"):
        cols = list(range(X.shape[1])) if kind == "logistic" else [IV]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(X[tr][:, cols], yt, logisticregression__sample_weight=weights(yt, 3))
        return [m.predict_proba(X[te][:, cols]) for te in te_list]
    if kind == "gru":
        m = GRUClf().fit(SEQ[tr], yt); return [m.predict_proba(SEQ[te]) for te in te_list]
    if kind == "tfidf":
        m = TfidfClf().fit(tr, yt); return [m.predict_proba(te) for te in te_list]
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------- R2 single-feature rule
m1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
m1.fit(X[:, [IV]], y, logisticregression__sample_weight=weights(y, 3))
grid = np.logspace(-3, 3, 4000)
lab = m1.predict(grid[:, None])
cuts = [{"from_s": float(grid[i - 1]), "to_s": float(grid[i]), "from": CLASSES[lab[i - 1]], "to": CLASSES[lab[i]]}
        for i in range(1, len(grid)) if lab[i] != lab[i - 1]]
regions = []
start = grid[0]
for i in range(1, len(grid) + 1):
    if i == len(grid) or lab[i] != lab[i - 1]:
        regions.append({"class": CLASSES[lab[i - 1]], "from_s": float(start), "to_s": float(grid[i - 1])}); start = grid[i] if i < len(grid) else None
out["R2_rule1"] = {"feature": "iv_med (median inter-command interval, s)",
                   "form": "standardised multinomial logistic regression, inverse-frequency class weights, argmax",
                   "regions_full_data": regions, "n_cmd_constant": "n_cmd" in const}
log("R2 rule1 regions", [(r["class"], round(r["from_s"], 3), round(r["to_s"], 3)) for r in regions])

# ------------------------------------------------------------------------------------------- R3 leak-free LOCO table
NEG = np.where(y != 2)[0]
neg_folds = list(GroupKFold(5).split(NEG, groups=grp[NEG]))
KINDS = ["logistic", "gbdt", "gru", "rule1", "tfidf"]
loco = {k: {} for k in KINDS + ["reworr"]}
own_probs = {}
for held in HELD:
    te = np.where(corpus == held)[0]
    if len(te) == 0:
        continue
    agent_tr = np.where((y == 2) & (corpus != held))[0]
    g_te = np.array([S[i]["group"] for i in te])
    for kind in KINDS:
        P_te = np.zeros((len(te), 3)); neg_score = np.zeros(len(NEG)); neg_pred = np.zeros(len(NEG), int)
        for ntr, nte in neg_folds:
            tr = np.r_[agent_tr, NEG[ntr]]
            Pa, Pn = fit_predict(kind, tr, [te, NEG[nte]])
            P_te += Pa / len(neg_folds); neg_score[nte] = Pn[:, 2]; neg_pred[nte] = Pn.argmax(1)
        hit = (P_te.argmax(1) == 2).astype(float)
        thr = float(np.quantile(neg_score, 0.99))
        det = (P_te[:, 2] > thr).astype(float)
        yy = np.r_[np.ones(len(te)), np.zeros(len(NEG))]; ss = np.r_[P_te[:, 2], neg_score]
        lo, hi = cluster_boot(lambda idx: float(np.mean(hit[idx])), g_te, B=600)
        tlo, thi = cluster_boot(lambda idx: float(np.mean(det[idx])), g_te, B=600)
        loco[kind][held] = {"n": int(len(te)), "n_groups": int(len(set(g_te))),
                            "argmax_recall": float(hit.mean()), "argmax_ci": [lo, hi],
                            "tpr1": tpr_at(yy, ss), "tpr1_thr": float(det.mean()), "tpr1_ci": [tlo, thi],
                            "neg_fpr_argmax": float(np.mean(neg_pred == 2)),
                            "agent_called_human": float(np.mean(P_te.argmax(1) == 1)),
                            "agent_called_bot": float(np.mean(P_te.argmax(1) == 0))}
        if held == "own" and kind == "gbdt":
            own_probs = {"idx": te.tolist(), "p_agent": P_te[:, 2].tolist(), "argmax": P_te.argmax(1).tolist(), "thr": thr}
        log("R3", held, kind, round(hit.mean(), 3), round(loco[kind][held]["tpr1"], 3))
    med = X[te][:, IV]; rh = ((med >= 1.0) & (med <= 2.0)).astype(float)
    rn = X[NEG][:, IV]
    lo, hi = cluster_boot(lambda idx: float(np.mean(rh[idx])), g_te, B=600)
    loco["reworr"][held] = {"n": int(len(te)), "argmax_recall": float(rh.mean()), "argmax_ci": [lo, hi],
                            "rule_fpr_on_negatives": float(np.mean((rn >= 1.0) & (rn <= 2.0)))}
for kind, d in loco.items():
    hs = [h for h in HELD if h in d]
    if hs:
        ns = np.array([d[h]["n"] for h in hs])
        for met in ("argmax_recall", "tpr1"):
            vals = np.array([d[h][met] for h in hs if met in d[h]])
            if len(vals) == len(ns):
                d[f"mean_{met}"] = float(vals.mean()); d[f"wmean_{met}"] = float((vals * ns).sum() / ns.sum())
out["R3_loco"] = loco

# in-distribution reference (grouped CV over everything)
idref = {}
id_folds = list(GroupKFold(5).split(X, y, grp))
OOF = {}
for kind in KINDS:
    P = np.zeros((len(y), 3))
    for tr, te in id_folds:
        P[te] = fit_predict(kind, tr, [te])[0]
    OOF[kind] = P
    ya = (y == 2).astype(int)
    idref[kind] = {"macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
                   "agent_argmax_recall": float(np.mean(P[y == 2].argmax(1) == 2)),
                   "agent_auc": float(roc_auc_score(ya, P[:, 2])), "agent_tpr1": tpr_at(ya, P[:, 2])}
    log("R3 in-dist", kind, {k: round(v, 3) for k, v in idref[kind].items()})
iv_all = X[:, IV]; rh = (iv_all >= 1.0) & (iv_all <= 2.0)
idref["reworr"] = {"agent_argmax_recall": float(rh[y == 2].mean()), "rule_fpr_on_negatives": float(rh[y != 2].mean())}
out["R3_in_distribution"] = idref

# ------------------------------------------------------------------------------------------- R4 OOD probes
ood = {}
# (a) novelty detector trained on bot+human only: does "not a known negative" reproduce LOCO agent detection?
Xs = StandardScaler().fit(X[NEG]).transform(X)
nov_neg = np.zeros(len(NEG)); nov_agent = collections.defaultdict(lambda: np.zeros(0))
agent_scores = {h: np.zeros(int((corpus == h).sum())) for h in HELD if (corpus == h).any()}
for ntr, nte in neg_folds:
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Xs[NEG[ntr]])
    nov_neg[nte] = -iso.score_samples(Xs[NEG[nte]])
    for h in agent_scores:
        agent_scores[h] += -iso.score_samples(Xs[corpus == h]) / len(neg_folds)
thr = float(np.quantile(nov_neg, 0.99))
ood["novelty_isoforest"] = {h: {"tpr1": tpr_at(np.r_[np.ones(len(v)), np.zeros(len(nov_neg))], np.r_[v, nov_neg]),
                                "above_thr": float(np.mean(v > thr))} for h, v in agent_scores.items()}
log("R4a novelty", {h: round(v["tpr1"], 3) for h, v in ood["novelty_isoforest"].items()})


def holdout_subpop(mask, label):
    tr = np.where(~mask)[0]; te = np.where(mask)[0]
    m = XGB().fit(X[tr], y[tr], sample_weight=weights(y[tr], 3))
    p = m.predict_proba(X[te]).argmax(1)
    return {"n": int(len(te)), "true_class": label,
            "called_bot": float(np.mean(p == 0)), "called_human": float(np.mean(p == 1)), "called_agent": float(np.mean(p == 2))}


server = np.array([s["corpus"] == "kypo" and s["sub"].get("env") == "victim" for s in S])
if server.sum() >= 10:
    ood["unseen_human_subpop_kypo_server"] = holdout_subpop(server, "human")
y2020 = np.array([s["corpus"] == "cyberlab" and str(s["sub"].get("day", ""))[:4] == "2020" for s in S])
if y2020.sum() >= 10:
    ood["unseen_bot_period_cyberlab_2020"] = holdout_subpop(y2020, "bot")
out["R4_ood"] = ood
log("R4", {k: v for k, v in ood.items() if k != "novelty_isoforest"})

# ------------------------------------------------------------------------------------------- R5 scale + adaptive (one cross-fitted detector)
AG = np.where(y == 2)[0]
HUMAN_IV = [np.diff([c[0] for c in S[i]["cmds"][:K]]) for i in np.where(y == 1)[0]]
SCALES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0]
WB = [(f, c) for f in (1.0, 1.5, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0) for c in (0.0, 1.0, 5.0, 15.0) if not (f == 1.0 and c == 0.0)]


def retime(s, newt):
    cm = [list(c) for c in s["cmds"][:K]]
    for k, t in enumerate(newt):
        cm[k][0] = float(t)
    return {**s, "cmds": cm}


def affine(s, f, c):
    t = [x[0] for x in s["cmds"][:K]]; nt = [t[0]]
    for k in range(1, len(t)):
        nt.append(nt[-1] + f * (t[k] - t[k - 1]) + c)
    return retime(s, nt)


def human_replay(s, rng):
    t = [x[0] for x in s["cmds"][:K]]; hv = HUMAN_IV[rng.integers(len(HUMAN_IV))]; nt = [t[0]]
    for k in range(1, len(t)):
        nt.append(nt[-1] + max(t[k] - t[k - 1], float(hv[(k - 1) % len(hv)])))
    return retime(s, nt)


def add_only(s, rng):
    t = [x[0] for x in s["cmds"][:K]]; nt = [t[0]]
    for k in range(1, len(t)):
        nt.append(nt[-1] + (t[k] - t[k - 1]) + abs(rng.normal(3, 2)))
    return retime(s, nt)


def dechain(s, rng):
    cm = [list(c) for c in s["cmds"][:K]]
    for c in cm:
        c[1] = (c[1] or "").split(";")[0]
    return {**s, "cmds": cm}


def dur(s):
    return max(1e-6, s["cmds"][:K][-1][0] - s["cmds"][0][0])


ya = (y == 2).astype(int)
neg_oof = np.zeros(len(y)); agent_clean = np.zeros(len(y)); argmax_clean = np.zeros(len(y), int)
sc_bin = {a: np.zeros(len(y)) for a in SCALES}; sc_arg = {a: np.zeros(len(y), int) for a in SCALES}
wb_best = np.full(len(y), np.inf); wb_infl = np.ones(len(y)); wb10_best = np.full(len(y), np.inf); wb10_infl = np.ones(len(y))
att = {k: np.zeros(len(y)) for k in ("human_replay", "add_only_delay", "structure_dechain")}
att_infl = {k: np.ones(len(y)) for k in att}
rng5 = np.random.default_rng(11)
for fold, (tr, te) in enumerate(id_folds):
    det = XGB().fit(X[tr], ya[tr], sample_weight=weights(ya[tr], 2))
    m3 = XGB().fit(X[tr], y[tr], sample_weight=weights(y[tr], 3))
    neg_oof[te] = det.predict_proba(X[te])[:, 1]
    ag = te[y[te] == 2]
    agent_clean[ag] = neg_oof[ag]; argmax_clean[ag] = m3.predict_proba(X[ag]).argmax(1)
    base = [S[i] for i in ag]; d0 = np.array([dur(s) for s in base])
    for a in SCALES:
        Xt, _ = a_eval.feat([affine(s, a, 0.0) for s in base])
        sc_bin[a][ag] = det.predict_proba(Xt)[:, 1]; sc_arg[a][ag] = m3.predict_proba(Xt).argmax(1)
    for f, c in WB:
        tr_s = [affine(s, f, c) for s in base]; Xt, _ = a_eval.feat(tr_s)
        sc = det.predict_proba(Xt)[:, 1]; infl = np.array([dur(s) for s in tr_s]) / d0
        better = sc < wb_best[ag]; wb_best[ag] = np.where(better, sc, wb_best[ag]); wb_infl[ag] = np.where(better, infl, wb_infl[ag])
        ok = infl <= 10.0
        b10 = ok & (sc < wb10_best[ag]); wb10_best[ag] = np.where(b10, sc, wb10_best[ag]); wb10_infl[ag] = np.where(b10, infl, wb10_infl[ag])
    for nm, fn in (("human_replay", human_replay), ("add_only_delay", add_only), ("structure_dechain", dechain)):
        tr_s = [fn(s, rng5) for s in base]; Xt, _ = a_eval.feat(tr_s)
        att[nm][ag] = det.predict_proba(Xt)[:, 1]; att_infl[nm][ag] = np.array([dur(s) for s in tr_s]) / d0
    log("R5 fold", fold, "done")
negs = neg_oof[y != 2]


def tpr_vs_neg(scores):
    return tpr_at(np.r_[np.ones(len(scores)), np.zeros(len(negs))], np.r_[scores, negs])


thr_oof = float(np.quantile(negs, 0.99))
wb10_best = np.where(np.isfinite(wb10_best), wb10_best, agent_clean)
out["R5_adaptive"] = {
    "detector": "agent-vs-rest GBDT, 5-fold grouped cross-fitting; negatives scored out-of-fold; TPR@1%FPR",
    "clean_agent_tpr1": tpr_vs_neg(agent_clean[AG]),
    "clean_agent_argmax_recall_3class": float(np.mean(argmax_clean[AG] == 2)),
    "scale_tpr1": {str(a): tpr_vs_neg(sc_bin[a][AG]) for a in SCALES},
    "scale_argmax_recall_3class": {str(a): float(np.mean(sc_arg[a][AG] == 2)) for a in SCALES},
    "scale_argmax_called_human_3class": {str(a): float(np.mean(sc_arg[a][AG] == 1)) for a in SCALES},
    "whitebox_grid": [list(p) for p in WB],
    "whitebox_addonly": {"residual_tpr1": tpr_vs_neg(wb_best[AG]), "median_inflation": float(np.median(wb_infl[AG]))},
    "whitebox_addonly_infl_le_10": {"residual_tpr1": tpr_vs_neg(wb10_best[AG]), "median_inflation": float(np.median(wb10_infl[AG]))},
}
for nm in att:
    out["R5_adaptive"][nm] = {"residual_tpr1": tpr_vs_neg(att[nm][AG]), "median_inflation": float(np.median(att_infl[nm][AG])),
                              "mean_inflation": float(np.mean(att_infl[nm][AG]))}
log("R5", {k: v for k, v in out["R5_adaptive"].items() if k not in ("whitebox_grid", "detector")})

# ------------------------------------------------------------------------------------------- R6 mechanistic table (benchmark sample)


def think_series(s):
    c = s["cmds"][:K]
    if s["corpus"] in ("dtu", "aipot", "own"):
        th = [x[4] for x in c[1:]]; ln = [len(x[1] or "") for x in c[1:]]
    else:
        th = list(np.diff([x[0] for x in c])); ln = [len(x[1] or "") for x in c[1:]]
    keep = [i for i, t in enumerate(th) if t is not None and t >= 0]
    return np.array([th[i] for i in keep], float), np.array([ln[i] for i in keep], float)


def srckey(s):
    if s["corpus"] in ("dtu", "aipot", "own"):
        return f'{s["corpus"]}:{s["sub"]["model"]}'
    if s["corpus"] == "kypo":
        return "kypo:" + ("server" if s["sub"].get("env") == "victim" else "attacker-host")
    return s["corpus"]


THINK_DEF = {"dtu": "command time minus completion time of the previous output (harness read-timeout removed)",
             "aipot": "harness-reported model inference seconds (API round-trip for hosted models)",
             "own": "harness-reported local llama.cpp inference seconds",
             "kypo": "raw inter-command interval (no harness; reading and typing included)",
             "cyberlab": "raw inter-command interval (no harness)"}
rows = collections.defaultdict(lambda: {"n_sessions": 0, "rho": [], "cv": [], "med": [], "provider": None})
for s in S:
    k = srckey(s); r = rows[k]; r["n_sessions"] += 1
    r["provider"] = s["sub"].get("provider") or r["provider"]
    th, ln = think_series(s)
    if len(th) >= 4 and np.std(th) > 0 and np.std(ln) > 0:
        r["rho"].append(float(spearmanr(th, ln).correlation))
    if len(th):
        r["cv"].append(float(np.std(th) / (np.mean(th) + 1e-9))); r["med"].append(float(np.median(th)))
mech = {}
for k in sorted(rows):
    r = rows[k]; rho = np.array(r["rho"])
    mech[k] = {"n_sessions": r["n_sessions"], "n_valid_rho": int(len(rho)),
               "rho_med": float(np.median(rho)) if len(rho) else None,
               "rho_q25": float(np.percentile(rho, 25)) if len(rho) else None,
               "rho_q75": float(np.percentile(rho, 75)) if len(rho) else None,
               "cv_med": float(np.median(r["cv"])) if r["cv"] else None,
               "think_med_s": float(np.median(r["med"])) if r["med"] else None,
               "provider": r["provider"],
               "serving": ("local" if k.split(":")[0] in ("dtu", "own") else
                           ("local" if str(r["provider"]).lower() in ("ollama", "local", "vllm") else "API/hosted")
                           if k.startswith("aipot:") else "n/a (no LLM)")}
out["R6_mech"] = {"think_definition": THINK_DEF, "rows": mech,
                  "sum_sessions": int(sum(v["n_sessions"] for v in mech.values()))}
log("R6", {k: (v["n_sessions"], None if v["rho_med"] is None else round(v["rho_med"], 2)) for k, v in mech.items()})

# ------------------------------------------------------------------------------------------- R7 own-corpus conditions
cond = {}
own_idx = np.where(corpus == "own")[0]
if len(own_idx) and own_probs:
    lo_p = dict(zip(own_probs["idx"], own_probs["p_agent"])); lo_a = dict(zip(own_probs["idx"], own_probs["argmax"]))
    base_med = None
    for cnd in ("base", "deceive", "mimic", "jitter"):
        ii = [i for i in own_idx if S[i]["sub"].get("cond") == cnd]
        if not ii:
            continue
        sc_id = agent_clean[ii]
        cond[cnd] = {"n": len(ii), "n_models": len({S[i]["sub"]["model"] for i in ii}),
                     "median_iv_s": float(np.median(X[ii][:, IV])),
                     "median_duration_s": float(np.median([dur(S[i]) for i in ii])),
                     "in_dist_tpr1": tpr_vs_neg(sc_id),
                     "in_dist_above_thr": float(np.mean(sc_id > thr_oof)),
                     "loco_own_argmax_recall": float(np.mean([lo_a[i] == 2 for i in ii])),
                     "loco_own_above_thr": float(np.mean([lo_p[i] > own_probs["thr"] for i in ii])),
                     "loco_own_called_human": float(np.mean([lo_a[i] == 1 for i in ii]))}
        if cnd == "base":
            base_med = cond[cnd]["median_duration_s"]
    for cnd in cond:
        cond[cnd]["duration_vs_base"] = float(cond[cnd]["median_duration_s"] / base_med) if base_med else None
out["R7_own_conditions"] = cond
log("R7", {k: (v["n"], round(v["in_dist_tpr1"], 3), round(v["loco_own_argmax_recall"], 3)) for k, v in cond.items()})

# ------------------------------------------------------------------------------------------- R8 timestamp resolution + 1 s robustness
res = {}
for name in ("cyberlab", "kypo", "dtu", "aipot", "own"):
    ss = [s for s in S if s["corpus"] == name]
    if not ss:
        continue
    allint = [all(abs(c[0] - round(c[0])) < 1e-9 for c in s["cmds"][:K]) for s in ss]
    fr = np.array([abs(c[0] - round(c[0])) > 1e-9 for s in ss for c in s["cmds"][1:K]])
    ivs = np.concatenate([np.diff([c[0] for c in s["cmds"][:K]]) for s in ss])
    pos = ivs[ivs > 0]
    res[name] = {"frac_sessions_integer_seconds": float(np.mean(allint)), "frac_nonint_timestamps": float(fr.mean()),
                 "min_pos_interval_s": float(pos.min()) if len(pos) else None,
                 "p1_pos_interval_s": float(np.percentile(pos, 1)) if len(pos) else None,
                 "frac_iv_lt0p1": float(np.mean(ivs < 0.1))}
rr = np.random.default_rng(5)


def coarsen(s):
    u = rr.random(); t = [c[0] for c in s["cmds"][:K]]
    return retime(s, [np.floor(x + u) - np.floor(u) for x in t])


Sc = [coarsen(s) for s in S]
Xc, _ = a_eval.feat(Sc)
rob = {}
P = np.zeros((len(y), 3))
for tr, te in id_folds:
    P[te] = XGB().fit(Xc[tr], y[tr], sample_weight=weights(y[tr], 3)).predict_proba(Xc[te])
rob["in_dist_macro_f1"] = float(f1_score(y, P.argmax(1), average="macro"))
rob["in_dist_agent_tpr1"] = tpr_at(ya, P[:, 2])
for held in HELD:
    te = np.where(corpus == held)[0]
    if not len(te):
        continue
    agent_tr = np.where((y == 2) & (corpus != held))[0]
    Pt = np.zeros((len(te), 3)); ns = np.zeros(len(NEG))
    for ntr, nte in neg_folds:
        tr = np.r_[agent_tr, NEG[ntr]]
        m = XGB().fit(Xc[tr], y[tr], sample_weight=weights(y[tr], 3))
        Pt += m.predict_proba(Xc[te]) / len(neg_folds); ns[nte] = m.predict_proba(Xc[NEG[nte]])[:, 2]
    rob[f"loco_{held}_argmax_recall"] = float(np.mean(Pt.argmax(1) == 2))
    rob[f"loco_{held}_tpr1"] = tpr_at(np.r_[np.ones(len(te)), np.zeros(len(NEG))], np.r_[Pt[:, 2], ns])
out["R8_timestamps"] = {"per_corpus": res, "coarsened_1s": rob,
                        "coarsening": "t' = floor(t + u) - floor(u), u ~ U(0,1) per session (emulates 1-s absolute logging)"}
log("R8", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in res.items()})
log("R8 robustness", {k: round(v, 3) for k, v in rob.items()})

fn = "a_revision_quick.json" if QUICK else "a_revision.json"
json.dump(out, open(os.path.join(RES, fn), "w"), indent=1)
log("saved", fn)
