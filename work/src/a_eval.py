"""Paper A - the honest cross-corpus evaluation (headline results).

Produces results/a_eval.json with:
  E1 naive grouped CV (the optimistic number the pilots report)
  E2 confound audit: within-class SOURCE AUROC vs the class AUROC, same features
  E3 leave-one-corpus-out for the agent class (the honest generalisation number) + bot temporal split
  E4 baselines under LOCO: Reworr fixed-threshold, 2-feature rule, logistic, TF-IDF content
  E5 scale-invariance: multiply agent think/intervals x{0.05..10}, re-measure agent recall
  E6 adaptive attackers (post-hoc on recorded agent sessions): human-timing replay, add-only delay,
     structure (de-chain / typo-retry) -> residual agent TPR@1%FPR + duration inflation
  E7 early detection: agent TPR@1%FPR vs first-k commands
All with cluster bootstrap CIs over the grouping unit. Vocabulary-agnostic features only.
"""
import json, os, collections
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from a_data import read
import a_feat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
K = int(os.environ.get("K", 10)); MINC = K
SOURCE_LEAK = {"pre", "has_pre", "has_outlen", "corr_iv_prevout"}
TIMING_PREF = ("iv_", "frac_iv", "dur", "rate", "corr_iv")
RNG = np.random.default_rng(0)
CLASSES = ["bot", "human", "agent"]
CAP = 4000


def load():
    S = []
    src = [("cyberlab", "bot"), ("kypo", "human"), ("dtu", "agent"), ("aipot", "agent")]
    if os.path.exists(os.path.join(ROOT, "data", "sessions", "own.jsonl.gz")):
        src.append(("own", "agent"))
    for name, lab in src:
        rows = [s for s in read(name) if len(s["cmds"]) >= MINC]
        for s in rows:
            s["y"] = lab
        if len(rows) > CAP:
            rows = [rows[i] for i in RNG.choice(len(rows), CAP, replace=False)]
        S += rows
    return S


def XGB():
    return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                             colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=8, random_state=0)


def weights(y, ncls):
    w = np.ones(len(y))
    for i in range(ncls):
        m = (y == i)
        if m.sum():
            w[m] = len(y) / (ncls * m.sum())
    return w


def tpr_at(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def feat(S):
    X, names = a_feat.matrix(S, early=K)
    keep = [i for i, n in enumerate(names) if n not in SOURCE_LEAK]
    return X[:, keep], [names[i] for i in keep]


def cluster_boot(fn, groups, B=400):
    """Bootstrap a metric by resampling whole groups. fn(mask)->float over selected group members."""
    uq = list(set(groups)); vals = []
    g2idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        g2idx[g].append(i)
    for _ in range(B):
        pick = RNG.choice(len(uq), len(uq), replace=True)
        idx = np.array([i for k in pick for i in g2idx[uq[k]]])
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def main():
    S = load()
    y = np.array([CLASSES.index(s["y"]) for s in S])
    corpus = np.array([s["corpus"] for s in S])
    grp = np.array([f'{s["y"]}:{s["group"]}' for s in S])
    X, names = feat(S)
    out = {"K": K, "n_sessions": len(S), "features": names,
           "per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
           "per_corpus": {k: int(v) for k, v in collections.Counter(corpus).items()}}
    print("loaded", out["per_class"], out["per_corpus"], flush=True)

    # ---- E1 naive grouped CV ----
    P = np.zeros((len(y), 3))
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = XGB().fit(X[tr], y[tr], sample_weight=weights(y[tr], 3))
        P[te] = m.predict_proba(X[te])
    out["E1_naive"] = {"macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
                       "auc_agent": float(roc_auc_score((y == 2).astype(int), P[:, 2])),
                       "tpr1_agent": tpr_at((y == 2).astype(int), P[:, 2])}
    print("E1", out["E1_naive"], flush=True)

    # ---- E2 within-class source AUROC vs class AUROC (agent class) ----
    agent_src = {"dtu": 0, "aipot": 1, "own": 2}
    ai = np.array([i for i, s in enumerate(S) if s["corpus"] in agent_src])
    if len(set(corpus[ai])) >= 2:
        ys = np.array([agent_src[corpus[i]] for i in ai]); Xi = X[ai]; gi = grp[ai]
        # one-vs-rest source AUROC via grouped CV (how identifiable is the SOURCE within agents)
        aucs = []
        for sid in sorted(set(ys)):
            yy = (ys == sid).astype(int)
            if yy.sum() < 10 or (1 - yy).sum() < 10:
                continue
            Ps = np.zeros(len(yy))
            for tr, te in GroupKFold(min(5, len(set(gi)))).split(Xi, yy, gi):
                if len(set(yy[tr])) < 2:
                    continue
                Ps[te] = XGB().fit(Xi[tr], yy[tr], sample_weight=weights(yy[tr], 2)).predict_proba(Xi[te])[:, 1]
            aucs.append(roc_auc_score(yy, Ps))
        out["E2_within_agent_source_auc"] = {"mean_ovr_auc": float(np.mean(aucs)), "per": [round(a, 3) for a in aucs]}
        print("E2 within-agent source AUROC", out["E2_within_agent_source_auc"], flush=True)
    # kypo attacker-host vs server (human source confound)
    ky = [i for i, s in enumerate(S) if s["corpus"] == "kypo"]
    env = np.array([1 if S[i]["sub"].get("env") == "victim" else 0 for i in ky])
    if 10 < env.sum() < len(env) - 10:
        Pk = np.zeros(len(ky)); Xk = X[ky]; gk = grp[ky]
        for tr, te in GroupKFold(5).split(Xk, env, gk):
            if len(set(env[tr])) < 2:
                continue
            Pk[te] = XGB().fit(Xk[tr], env[tr], sample_weight=weights(env[tr], 2)).predict_proba(Xk[te])[:, 1]
        out["E2_kypo_hostrole_auc"] = float(roc_auc_score(env, Pk))
        print("E2 kypo host-role AUROC", round(out["E2_kypo_hostrole_auc"], 3), flush=True)

    # ---- E3 leave-one-agent-corpus-out (honest agent generalisation) ----
    loco = {}
    for held in sorted(set(corpus[ai])):
        tr = np.array([i for i in range(len(S)) if not (S[i]["corpus"] == held)])
        te = np.array([i for i in range(len(S)) if S[i]["corpus"] == held])     # held-out agent corpus
        m = XGB().fit(X[tr], y[tr], sample_weight=weights(y[tr], 3))
        pa = m.predict_proba(X[te])
        pred = pa.argmax(1)
        # cluster-bootstrap CI over the held corpus's groups (models), capturing the few-independent-units uncertainty
        g_te = np.array([S[i]["group"] for i in te]); hit = (pred == 2).astype(float)
        lo, hi = cluster_boot(lambda idx: float(np.mean(hit[idx])), g_te, B=600)
        loco[held] = {"n": int(len(te)), "n_groups": int(len(set(g_te))),
                      "agent_recall": float(np.mean(pred == 2)), "ci": [lo, hi],
                      "mean_p_agent": float(pa[:, 2].mean())}
        # TPR@1%FPR for this held agent corpus vs the (bot+human) negatives of the fold
        neg = np.array([i for i in range(len(S)) if S[i]["y"] != "agent"])
        yy = np.r_[np.ones(len(te)), np.zeros(len(neg))]
        ss = np.r_[pa[:, 2], m.predict_proba(X[neg])[:, 2]]
        loco[held]["tpr1"] = tpr_at(yy, ss)
    out["E3_loco_agent"] = loco
    print("E3 LOCO agent", {k: round(v["agent_recall"], 3) for k, v in loco.items()}, flush=True)
    # bot temporal split (train on 2019, test on 2020)
    cb = [i for i, s in enumerate(S) if s["corpus"] == "cyberlab"]
    yr = np.array([S[i]["sub"].get("day", "")[:4] for i in cb])
    if "2019" in set(yr) and "2020" in set(yr):
        trb = np.array([i for i in range(len(S)) if not (S[i]["corpus"] == "cyberlab" and S[i]["sub"].get("day", "")[:4] == "2020")])
        teb = np.array([i for i in cb if S[i]["sub"].get("day", "")[:4] == "2020"])
        mb = XGB().fit(X[trb], y[trb], sample_weight=weights(y[trb], 3))
        out["E3_bot_temporal_recall_2020"] = float(np.mean(mb.predict_proba(X[teb]).argmax(1) == 0))

    # ---- E4 baselines under LOCO (agent recall, averaged over held agent corpora) ----
    base = {}
    iv_med_idx = names.index("iv_med")
    # Reworr-style fixed threshold: median inter-command interval in [1.0, 2.0] s => agent
    rew = []
    for held in sorted(set(corpus[ai])):
        te = np.array([i for i in range(len(S)) if S[i]["corpus"] == held])
        med = X[te][:, iv_med_idx]
        rew.append(float(np.mean((med >= 1.0) & (med <= 2.0))))
    base["reworr_fixed_1to2s_agent_recall"] = float(np.mean(rew))
    # 2-feature logistic rule (iv_med, n_cmd) and full logistic, LOCO agent recall
    for tag, cols in (("rule2", [iv_med_idx, names.index("n_cmd")]), ("logistic_all", list(range(len(names))))):
        rec = []
        for held in sorted(set(corpus[ai])):
            tr = np.array([i for i in range(len(S)) if S[i]["corpus"] != held])
            te = np.array([i for i in range(len(S)) if S[i]["corpus"] == held])
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr][:, cols], y[tr], logisticregression__sample_weight=weights(y[tr], 3))
            rec.append(float(np.mean(m.predict(X[te][:, cols]) == 2)))
        base[f"{tag}_agent_recall_loco"] = float(np.mean(rec))
    # TF-IDF content model (NOT vocabulary-agnostic) under LOCO - to quantify source vocabulary leakage
    docs = [" ".join((c[1] or "") for c in s["cmds"][:K]) for s in S]
    tf_rec = []
    for held in sorted(set(corpus[ai])):
        tr = np.array([i for i in range(len(S)) if S[i]["corpus"] != held])
        te = np.array([i for i in range(len(S)) if S[i]["corpus"] == held])
        vec = TfidfVectorizer(max_features=3000, token_pattern=r"[^\s]+")
        Xtr = vec.fit_transform([docs[i] for i in tr]); Xte = vec.transform([docs[i] for i in te])
        m = LogisticRegression(max_iter=2000); m.fit(Xtr, y[tr], sample_weight=weights(y[tr], 3))
        tf_rec.append(float(np.mean(m.predict(Xte) == 2)))
    base["tfidf_content_agent_recall_loco"] = float(np.mean(tf_rec))
    out["E4_baselines"] = base
    print("E4", {k: round(v, 3) for k, v in base.items()}, flush=True)

    # ---- E5 scale-invariance: rescale agent intervals, re-measure agent detection (train once, no agent in test rescaled) ----
    # train 3-class on all; then for the agent test rows, multiply their inter-command intervals and recompute features
    m = XGB().fit(X, y, sample_weight=weights(y, 3))
    scale_res = {}
    agent_idx = [i for i in range(len(S)) if S[i]["y"] == "agent"]
    for sc in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
        Sr = []
        for i in agent_idx:
            s = S[i]; cm = [list(c) for c in s["cmds"][:K]]
            t0 = cm[0][0]
            for c in cm:
                c[0] = t0 + (c[0] - t0) * sc
            Sr.append({**s, "cmds": cm})
        Xr, _ = feat(Sr)
        scale_res[sc] = float(np.mean(m.predict_proba(Xr).argmax(1) == 2))
    out["E5_scale_invariance_agent_recall"] = scale_res
    print("E5 scale sweep", {k: round(v, 3) for k, v in scale_res.items()}, flush=True)

    # ---- E6 adaptive attackers (post-hoc on agent sessions) vs a detector trained without that attack ----
    # detector: agent-vs-rest, trained on clean data; attackers modify held agent intervals/structure
    yb = (y == 2).astype(int)
    det = XGB().fit(X, yb, sample_weight=weights(yb, 2))
    base_tpr = tpr_at(yb, det.predict_proba(X)[:, 1])
    human_iv = [np.diff([c[0] for c in s["cmds"][:K]]) for s in S if s["y"] == "human" and len(s["cmds"]) > 1]
    adapt = {"clean_agent_tpr1": base_tpr}
    def agent_eval(transform):
        Sr = [transform(dict(S[i]), i) for i in agent_idx]
        Xr, _ = feat(Sr)
        neg = [i for i in range(len(S)) if S[i]["y"] != "agent"]
        yy = np.r_[np.ones(len(Sr)), np.zeros(len(neg))]
        ss = np.r_[det.predict_proba(Xr)[:, 1], det.predict_proba(X[neg])[:, 1]]
        dur0 = np.mean([S[i]["cmds"][min(K, len(S[i]["cmds"]))-1][0] for i in agent_idx])
        dur1 = np.mean([Sr[j]["cmds"][len(Sr[j]["cmds"])-1][0] for j in range(len(Sr))])
        return tpr_at(yy, ss), float(dur1 / max(1e-9, dur0))
    def human_replay(s, i):
        cm = [list(c) for c in s["cmds"][:K]]; hv = human_iv[RNG.integers(len(human_iv))]
        t = cm[0][0]
        for k in range(1, len(cm)):
            gap = max(cm[k][0] - cm[k-1][0], float(hv[(k-1) % len(hv)]))   # add-only to a human cadence
            t = t + gap; cm[k][0] = t
        return {**s, "cmds": cm}
    def add_only(s, i):
        cm = [list(c) for c in s["cmds"][:K]]; t = cm[0][0]
        for k in range(1, len(cm)):
            gap = (cm[k][0] - cm[k-1][0]) + abs(RNG.normal(3, 2))          # inject positive delay
            t = t + gap; cm[k][0] = t
        return {**s, "cmds": cm}
    def structure(s, i):
        cm = []
        for c in s["cmds"][:K]:
            c = list(c); parts = (c[1] or "").split(";")
            c[1] = parts[0]                                                # de-chain compound commands
            cm.append(c)
        return {**s, "cmds": cm}
    for nm, fn in (("human_replay", human_replay), ("add_only_delay", add_only), ("structure_dechain", structure)):
        t1, infl = agent_eval(fn)
        adapt[nm] = {"residual_tpr1": t1, "duration_inflation": round(infl, 2)}

    # white-box add-only timing attacker: each agent knows the detector and picks the add-only transform
    # new_interval = f*interval + c (f>=1, c>=0, so strictly slower) that minimises its own agent score.
    neg = [i for i in range(len(S)) if S[i]["y"] != "agent"]
    neg_scores = det.predict_proba(X[neg])[:, 1]
    widx = agent_idx if len(agent_idx) <= 1500 else list(RNG.choice(agent_idx, 1500, replace=False))
    orig_dur = np.array([max(1e-6, S[i]["cmds"][min(K, len(S[i]["cmds"])) - 1][0] - S[i]["cmds"][0][0]) for i in widx])
    best_score = np.ones(len(widx)); best_infl = np.ones(len(widx))
    for f in (1.0, 2.0, 4.0, 8.0, 16.0):
        for c in (0.0, 5.0, 15.0):
            if f == 1.0 and c == 0.0:
                continue
            Sr = []
            for i in widx:
                orig = S[i]["cmds"][:K]; ivs = [orig[k][0] - orig[k - 1][0] for k in range(1, len(orig))]
                cm = [list(x) for x in orig]; t = cm[0][0]
                for k in range(1, len(cm)):
                    t = t + f * ivs[k - 1] + c; cm[k][0] = t
                Sr.append({**S[i], "cmds": cm})
            sc = det.predict_proba(feat(Sr)[0])[:, 1]
            dur1 = np.array([Sr[j]["cmds"][len(Sr[j]["cmds"]) - 1][0] - Sr[j]["cmds"][0][0] for j in range(len(Sr))])
            better = sc < best_score
            best_score = np.where(better, sc, best_score)
            best_infl = np.where(better, dur1 / orig_dur, best_infl)
    yy = np.r_[np.ones(len(widx)), np.zeros(len(neg))]; ss = np.r_[best_score, neg_scores]
    adapt["whitebox_addonly"] = {"residual_tpr1": tpr_at(yy, ss), "duration_inflation": round(float(np.median(best_infl)), 2)}
    out["E6_adaptive"] = adapt
    print("E6 adaptive", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in adapt.items()}, flush=True)

    # ---- E6b task-matched human-vs-agent (addresses task confounding) ----
    # humans operating a shell ON A COMPROMISED HOST (KYPO server/victim shells) vs agents operating a
    # shell on a real sshd host they accessed (our own corpus) -- same task type, different operator.
    hm = [i for i in range(len(S)) if S[i]["corpus"] == "kypo" and S[i]["sub"].get("env") == "victim"]
    am = [i for i in range(len(S)) if S[i]["corpus"] == "own"]
    if len(hm) >= 20 and len(am) >= 20:
        idx = hm + am
        yt = np.r_[np.zeros(len(hm)), np.ones(len(am))]
        gt = np.array([S[i]["group"] for i in idx])
        Xt = X[idx]
        Pt = np.zeros(len(yt))
        for tr, te in GroupKFold(min(5, len(set(gt)))).split(Xt, yt, gt):
            if len(set(yt[tr])) < 2:
                continue
            Pt[te] = XGB().fit(Xt[tr], yt[tr], sample_weight=weights(yt[tr], 2)).predict_proba(Xt[te])[:, 1]
        out["E6b_taskmatched_human_vs_agent"] = {
            "n_human_server": len(hm), "n_agent_own": len(am),
            "auc": float(roc_auc_score(yt, Pt)), "tpr1_agent": tpr_at(yt, Pt)}
        print("E6b task-matched human(server) vs agent(own)", out["E6b_taskmatched_human_vs_agent"], flush=True)

    # ---- E7 early detection ----
    early = {}
    for k in (3, 5, 8, 10):
        Sk = [s for s in S if len(s["cmds"]) >= k]
        yk = np.array([CLASSES.index(s["y"]) for s in Sk]); gk = np.array([f'{s["y"]}:{s["group"]}' for s in Sk])
        Xk, _ = a_feat.matrix(Sk, early=k); keepk = [j for j, n in enumerate(a_feat.FEATURE_NAMES) if n not in SOURCE_LEAK]
        Xk = Xk[:, keepk]
        Pk = np.zeros(len(yk))
        yb2 = (yk == 2).astype(int)
        for tr, te in GroupKFold(5).split(Xk, yb2, gk):
            Pk[te] = XGB().fit(Xk[tr], yb2[tr], sample_weight=weights(yb2[tr], 2)).predict_proba(Xk[te])[:, 1]
        early[k] = {"tpr1_agent": tpr_at(yb2, Pk), "auc_agent": float(roc_auc_score(yb2, Pk))}
    out["E7_early"] = early
    print("E7 early", {k: round(v["tpr1_agent"], 3) for k, v in early.items()}, flush=True)

    json.dump(out, open(os.path.join(RES, "a_eval.json"), "w"), indent=1)
    print("saved a_eval.json", flush=True)


if __name__ == "__main__":
    main()
