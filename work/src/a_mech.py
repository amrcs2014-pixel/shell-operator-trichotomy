"""Paper A - is there a mechanistic, scale-invariant latency cue?

Autoregressive decoding: latency ~ a + b*(generated tokens) + c*(context tokens). If agents follow this,
per-session Spearman rho(think_i, len(cmd_i)) should be high and tight for agents. Spearman is invariant
to ANY monotone rescaling of time, so it cannot come from GPU speed or a fixed timeout floor.

think time: DTU = command time - previous output-complete time (removes the 1.2 s read-timeout);
            AIPot = harness inference_seconds;
            KYPO / CyberLab = inter-command interval (no output timing exists; for humans it includes
            reading + typing, which also grows with command length - the comparison that matters).
"""
import collections, json, os
import numpy as np
from scipy.stats import spearmanr
from a_data import read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MIN = 8


def think_series(s):
    c = s["cmds"]
    if s["corpus"] in ("dtu", "aipot", "own"):
        th = [x[4] for x in c[1:]]                       # think for commands 2..n (first has no prior output)
        ln = [len(x[1]) for x in c[1:]]
        po = [c[i][2] for i in range(len(c) - 1)]        # previous output length
    else:
        th = list(np.diff([x[0] for x in c]))
        ln = [len(x[1]) for x in c[1:]]
        po = [None] * len(th)
    keep = [i for i, t in enumerate(th) if t is not None and t >= 0]
    return np.array([th[i] for i in keep], float), np.array([ln[i] for i in keep], float), [po[i] for i in keep]


def rho(a, b):
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).correlation)


rows = collections.defaultdict(lambda: collections.defaultdict(list))
for name in ("dtu", "aipot", "kypo", "cyberlab"):
    for s in read(name):
        if len(s["cmds"]) < MIN:
            continue
        th, ln, po = think_series(s)
        if len(th) < MIN - 1:
            continue
        key = name
        if name == "dtu":
            key = "dtu:" + s["sub"]["model"]
        elif name == "aipot":
            key = "aipot:" + s["sub"]["model"]
        elif name == "kypo":
            key = "kypo:" + ("server" if s["sub"]["env"] == "victim" else "attacker-host")
        r = rows[key]
        r["rho_len"].append(rho(th, ln))
        if po and po[0] is not None:
            r["rho_prevout"].append(rho(th, np.array([p or 0 for p in po], float)))
        r["cv"].append(float(np.std(th) / (np.mean(th) + 1e-9)))
        r["med_think"].append(float(np.median(th)))
        r["min_think"].append(float(np.min(th)))

out = {}
print(f"{'source':38s} {'n':>5s}  {'rho(think,len) med [IQR]':>28s}  {'rho(think,prevout)':>20s}  {'CV med':>7s}  {'think med s':>11s}  {'think min':>9s}")
for k in sorted(rows, key=lambda x: (x.split(':')[0], x)):
    r = rows[k]
    rl = np.array([v for v in r["rho_len"] if np.isfinite(v)])
    rp = np.array([v for v in r["rho_prevout"] if np.isfinite(v)])
    q = lambda v: f"{np.median(v):+.2f} [{np.percentile(v,25):+.2f},{np.percentile(v,75):+.2f}]" if len(v) else "-"
    print(f"{k:38s} {len(rl):5d}  {q(rl):>28s}  {q(rp) if len(rp) else '-':>20s}  {np.median(r['cv']):7.2f}  {np.median(r['med_think']):11.2f}  {np.median(r['min_think']):9.2f}")
    out[k] = {"n": int(len(rl)), "rho_len_med": float(np.median(rl)) if len(rl) else None,
              "rho_len_q25": float(np.percentile(rl, 25)) if len(rl) else None, "rho_len_q75": float(np.percentile(rl, 75)) if len(rl) else None,
              "rho_prevout_med": float(np.median(rp)) if len(rp) else None,
              "cv_med": float(np.median(r["cv"])), "think_med": float(np.median(r["med_think"])), "think_min_med": float(np.median(r["min_think"]))}
json.dump(out, open(os.path.join(RES, "a_mech.json"), "w"), indent=1)
print("\nsaved a_mech.json")
