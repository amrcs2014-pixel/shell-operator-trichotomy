"""Paper A - content-agnostic session features.

No command vocabulary and no canary strings are used: only inter-command timing, the
structural shape of each command line, and session-level counts. A separate TF-IDF
content baseline lives in a_clf.py purely to quantify how much source vocabulary leaks.
"""
import math, re
import numpy as np

FLAG = re.compile(r"(^|\s)-{1,2}[A-Za-z]")
TOKEN = re.compile(r"\S+")


def _entropy(xs, bins=12):
    if len(xs) < 2:
        return 0.0
    h, _ = np.histogram(xs, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def _charclass_entropy(s):
    if not s:
        return 0.0
    c = {"l": 0, "u": 0, "d": 0, "p": 0, "s": 0}
    for ch in s:
        c["d" if ch.isdigit() else "l" if ch.islower() else "u" if ch.isupper() else "s" if ch.isspace() else "p"] += 1
    p = np.array([v for v in c.values() if v]) / len(s)
    return float(-(p * np.log(p)).sum())


def _lev_ratio(a, b):
    """Normalised Levenshtein distance between two short strings (1 = completely different)."""
    a, b = a[:60], b[:60]
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n] / max(m, n, 1)


def features(session, early=None):
    """Return an ordered dict of content-agnostic features for one session.
    `early`: if set, use only the first `early` commands (for early-detection curves)."""
    cmds = session["cmds"][:early] if early else session["cmds"]
    t = [c[0] for c in cmds]
    texts = [c[1] or "" for c in cmds]
    n = len(cmds)
    f = {}
    f["n_cmd"] = n

    # ---- timing ----
    iv = np.diff(t) if n > 1 else np.array([])
    logiv = np.log1p(np.clip(iv, 0, None))
    f["iv_med"] = float(np.median(iv)) if len(iv) else 0.0
    f["iv_mean"] = float(np.mean(iv)) if len(iv) else 0.0
    f["iv_std"] = float(np.std(iv)) if len(iv) else 0.0
    f["iv_iqr"] = float(np.percentile(iv, 75) - np.percentile(iv, 25)) if len(iv) else 0.0
    f["iv_min"] = float(np.min(iv)) if len(iv) else 0.0
    f["iv_max"] = float(np.max(iv)) if len(iv) else 0.0
    f["iv_cv"] = float(np.std(iv) / (np.mean(iv) + 1e-9)) if len(iv) else 0.0
    mu, sd = (np.mean(iv), np.std(iv)) if len(iv) else (0, 0)
    f["iv_burst"] = float((sd - mu) / (sd + mu + 1e-9)) if len(iv) else 0.0   # Goh-Barabasi burstiness
    f["iv_logent"] = _entropy(logiv)
    f["frac_iv_lt0p1"] = float(np.mean(iv < 0.1)) if len(iv) else 0.0
    f["frac_iv_lt1"] = float(np.mean(iv < 1.0)) if len(iv) else 0.0
    f["frac_iv_lt2"] = float(np.mean(iv < 2.0)) if len(iv) else 0.0
    f["frac_iv_gt10"] = float(np.mean(iv > 10.0)) if len(iv) else 0.0
    f["dur"] = float(t[-1] - t[0]) if n > 1 else 0.0
    f["rate"] = n / (f["dur"] + 1e-9) if f["dur"] > 0 else 0.0
    if len(iv) > 2:
        a = iv[:-1] - iv[:-1].mean(); b = iv[1:] - iv[1:].mean()
        f["iv_acf1"] = float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        f["iv_trend"] = float(np.polyfit(np.arange(len(iv)), iv, 1)[0])
    else:
        f["iv_acf1"] = 0.0; f["iv_trend"] = 0.0
    f["pre"] = float(session["pre"]) if session.get("pre") is not None else -1.0
    f["has_pre"] = 1.0 if session.get("pre") is not None else 0.0

    # ---- command structure ----
    lens = np.array([len(x) for x in texts], float)
    ntok = np.array([len(TOKEN.findall(x)) for x in texts], float)
    f["len_mean"] = float(lens.mean()); f["len_std"] = float(lens.std()); f["len_med"] = float(np.median(lens)); f["len_max"] = float(lens.max())
    f["tok_mean"] = float(ntok.mean()); f["tok_std"] = float(ntok.std()); f["tok_max"] = float(ntok.max())
    f["frac_pipe"] = float(np.mean(["|" in x for x in texts]))
    f["frac_chain"] = float(np.mean([(";" in x or "&&" in x or "||" in x) for x in texts]))
    f["frac_redir"] = float(np.mean([(">" in x or "<" in x) for x in texts]))
    f["frac_flag"] = float(np.mean([bool(FLAG.search(x)) for x in texts]))
    f["nflag_mean"] = float(np.mean([len(FLAG.findall(x)) for x in texts]))
    f["frac_subsh"] = float(np.mean([("$(" in x or "`" in x) for x in texts]))
    f["frac_path"] = float(np.mean([("/" in x) for x in texts]))
    f["frac_assign"] = float(np.mean([bool(re.match(r"^\s*[A-Za-z_]\w*=", x)) for x in texts]))
    f["frac_empty"] = float(np.mean([len(x.strip()) == 0 for x in texts]))
    f["cc_ent_mean"] = float(np.mean([_charclass_entropy(x) for x in texts]))

    # ---- sequence redundancy ----
    f["uniq_ratio"] = len(set(texts)) / n
    f["repeat_rate"] = float(np.mean([texts[i] == texts[i - 1] for i in range(1, n)])) if n > 1 else 0.0
    if n > 1:
        eds = [_lev_ratio(texts[i - 1], texts[i]) for i in range(1, n)]
        f["edit_mean"] = float(np.mean(eds)); f["edit_min"] = float(np.min(eds))
        f["frac_near_dup"] = float(np.mean([e < 0.2 for e in eds]))      # typo-and-retry
    else:
        f["edit_mean"] = 0.0; f["edit_min"] = 0.0; f["frac_near_dup"] = 0.0

    # ---- think-time dependence (needs timing + output lengths; else neutral 0) ----
    outs = [c[2] for c in cmds]
    if len(iv) > 2:
        cl = lens[1:]
        f["corr_iv_cmdlen"] = float(np.corrcoef(iv, cl)[0, 1]) if np.std(cl) > 0 else 0.0
        prev_out = np.array([outs[i] if outs[i] is not None else 0 for i in range(len(iv))], float)
        f["corr_iv_prevout"] = float(np.corrcoef(iv, prev_out)[0, 1]) if np.std(prev_out) > 0 else 0.0
    else:
        f["corr_iv_cmdlen"] = 0.0; f["corr_iv_prevout"] = 0.0
    f["has_outlen"] = 1.0 if any(o is not None for o in outs) else 0.0
    return f


FEATURE_NAMES = None


def matrix(sessions, early=None):
    rows = [features(s, early) for s in sessions]
    global FEATURE_NAMES
    FEATURE_NAMES = list(rows[0].keys())
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], float)
    X[~np.isfinite(X)] = 0.0
    return X, FEATURE_NAMES
