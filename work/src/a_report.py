"""Paper A - build figures, number macros and tables into paperA/.

Revised analyses come from results/a_revision.json (leak-free LOCO table, OOD probes, cross-fitted adaptive
attacks, mechanistic table on the benchmark sample, own-corpus conditions, timestamp robustness). The remaining
analyses keep their a_eval.json / a_confound.json source (within-source AUROC, task-matched control, early
detection, harness floor)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "paperA"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
C = {"bot": "#4C78A8", "human": "#F58518", "agent": "#E45756", "g1": "#54A24B", "g2": "#B279A2", "grey": "#9D9D9D"}
HELD = ["dtu", "aipot", "own"]
HNAME = {"dtu": "DTU", "aipot": "AIPot", "own": "Own"}


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def pct(x, nd=1):
    return f"{100*x:.{nd}f}"


def _box(ax, x, y, w, h, title, body, fc="#F4F6F8", ec="#444444", lw=0.9, tfs=6.6, bfs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=tfs, fontweight="bold")
    ax.text(x + w / 2, y + 1.6, body, ha="center", va="bottom", fontsize=bfs, linespacing=1.3)


def _arrow(ax, p, q, color="#444444", lw=1.0, style="-|>", cs="arc3,rad=0.0", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=cs, linestyle=ls, shrinkA=0, shrinkB=0))


def fig_pipeline():
    """Methodology diagram: sources -> unified sessions -> vocabulary-agnostic features -> classifiers,
    then the audit (pooled / confound / leave-one-corpus-out) and the stress tests (scale / mechanism / adaptive)."""
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
    Y1, H1 = 51, 14
    _box(ax, 1, Y1, 23, H1, "Wild bots", "CyberLab Cowrie\nSSH/Telnet honeypot", fc="#E8F1FA", ec="#3B6FA0")
    _box(ax, 26, Y1, 20, H1, "Human operators", "KYPO cyber-range\nshell commands", fc="#FFF4E5", ec="#C98A1E")
    _box(ax, 48, Y1, 25, H1, "Agent corpora", "DTU 'Honey for the Agent'\n+ AIPot pentest logs", fc="#FBE9E9", ec="#C0504D")
    _box(ax, 75, Y1, 24, H1, "Generated agents", "ReAct on real sshd\n4 models via llama.cpp", fc="#FBE9E9", ec="#C0504D")
    _box(ax, 26, 34, 48, 11, "Unified session representation", "timestamped commands, first $K{=}10$ turns,\nsingle idle-gap segmentation", fc="#F4F6F8")
    for cx in (12.5, 36, 60.5, 87):
        _arrow(ax, (cx, Y1), (cx if cx in (36, 60.5) else (36 if cx < 50 else 60.5), 45))
    _box(ax, 1, 18, 44, 12, "Vocabulary-agnostic features", "40 timing / structure / redundancy features\n(no command tokens, no canaries; source features excluded)", fc="#E9F5E9", ec="#3E8E41")
    _box(ax, 51, 18, 48, 12, "Classifiers", "logistic $\\cdot$ gradient-boosted trees $\\cdot$ GRU\n+ single-feature rule, TF-IDF, fixed threshold", fc="#F4F6F8")
    _arrow(ax, (38, 34), (23, 30)); _arrow(ax, (45, 24), (51, 24))   # starts at the box edge, not across its text
    _box(ax, 1, 2, 40, 12, "Cross-corpus audit", "pooled grouped-CV $\\cdot$ source confound\nleave-one-corpus-out $\\cdot$ OOD probes", fc="#EFEAF4", ec="#7E5BA6")
    _box(ax, 44, 2, 30, 12, "Stress tests", "think-time rescaling $\\cdot$ mechanistic cue\n$\\cdot$ adaptive attackers", fc="#EFEAF4", ec="#7E5BA6")
    _box(ax, 77, 3.5, 22, 9, "What transfers?", "coarse 'slow & deliberate',\nnot a fingerprint of agency", fc="#FFFFFF", ec="#222222", lw=1.3, tfs=6.8)
    _arrow(ax, (21, 18), (21, 14), color="#7E5BA6"); _arrow(ax, (70, 18), (59, 14), color="#7E5BA6")
    _arrow(ax, (41, 8), (44, 8), color="#7E5BA6"); _arrow(ax, (74, 8), (77, 8), color="#222222", lw=1.2)
    fig.savefig(os.path.join(OUT, "a_fig_pipeline.pdf"), bbox_inches="tight"); plt.close(fig)


# feature dictionary for the appendix table: name -> (group, definition, status)
FEAT_DEF = [
    ("n_cmd", "count", "number of commands in the truncated session", "constant ($=K$)"),
    ("iv_med", "timing", "median inter-command interval (s)", "used"),
    ("iv_mean", "timing", "mean inter-command interval", "used"),
    ("iv_std", "timing", "standard deviation of intervals", "used"),
    ("iv_iqr", "timing", "inter-quartile range of intervals", "used"),
    ("iv_min", "timing", "minimum interval", "used"),
    ("iv_max", "timing", "maximum interval", "used"),
    ("iv_cv", "timing", "coefficient of variation, $\\sigma/\\mu$", "used"),
    ("iv_burst", "timing", "Goh--Barab\\'asi burstiness $(\\sigma-\\mu)/(\\sigma+\\mu)$", "used"),
    ("iv_logent", "timing", "entropy of the 12-bin histogram of $\\log(1+\\Delta t)$", "used"),
    ("frac_iv_lt0p1", "timing", "fraction of intervals $<0.1$\\,s", "used"),
    ("frac_iv_lt1", "timing", "fraction of intervals $<1$\\,s", "used"),
    ("frac_iv_lt2", "timing", "fraction of intervals $<2$\\,s", "used"),
    ("frac_iv_gt10", "timing", "fraction of intervals $>10$\\,s", "used"),
    ("dur", "timing", "session duration, last minus first command time (s)", "used"),
    ("rate", "timing", "commands per second, $n/\\mathrm{dur}$", "used"),
    ("iv_acf1", "timing", "lag-1 autocorrelation of the interval series", "used"),
    ("iv_trend", "timing", "least-squares slope of interval against position", "used"),
    ("pre", "timing", "login-to-first-command delay", "excluded (source)"),
    ("has_pre", "timing", "whether that delay is recorded", "excluded (source)"),
    ("len_mean", "structure", "mean command length (characters)", "used"),
    ("len_std", "structure", "standard deviation of command length", "used"),
    ("len_med", "structure", "median command length", "used"),
    ("len_max", "structure", "maximum command length", "used"),
    ("tok_mean", "structure", "mean whitespace-token count", "used"),
    ("tok_std", "structure", "standard deviation of token count", "used"),
    ("tok_max", "structure", "maximum token count", "used"),
    ("frac_pipe", "structure", "fraction of commands containing a pipe", "used"),
    ("frac_chain", "structure", "fraction containing \\texttt{;}, \\texttt{\\&\\&} or \\texttt{||}", "used"),
    ("frac_redir", "structure", "fraction containing a redirect", "used"),
    ("frac_flag", "structure", "fraction containing a short or long flag", "used"),
    ("nflag_mean", "structure", "mean number of flags per command", "used"),
    ("frac_subsh", "structure", "fraction containing a command substitution", "used"),
    ("frac_path", "structure", "fraction containing a path separator", "used"),
    ("frac_assign", "structure", "fraction starting with a variable assignment", "used"),
    ("frac_empty", "structure", "fraction of empty command lines", "used"),
    ("cc_ent_mean", "structure", "mean character-class entropy", "used"),
    ("uniq_ratio", "redundancy", "distinct commands divided by $n$", "used"),
    ("repeat_rate", "redundancy", "fraction identical to the previous command", "used"),
    ("edit_mean", "redundancy", "mean normalised Levenshtein distance between consecutive commands", "used"),
    ("edit_min", "redundancy", "minimum of that distance", "used"),
    ("frac_near_dup", "redundancy", "fraction of consecutive pairs with distance $<0.2$ (typo-and-retry)", "used"),
    ("corr_iv_cmdlen", "think-time", "Pearson correlation of interval with the length of the next command", "used"),
    ("corr_iv_prevout", "think-time", "correlation of interval with the previous output length", "excluded (source)"),
    ("has_outlen", "think-time", "whether output lengths are recorded", "excluded (source)"),
]


def main():
    fig_pipeline()
    ev, conf, rv = J("a_eval.json"), J("a_confound.json"), J("a_revision.json")
    M = {}

    def mac(name, val):
        M[name] = val

    # ---- counts ----
    if ev:
        mac("nSessions", f"{ev['n_sessions']:,}".replace(",", "{,}"))
        for c in ("bot", "human", "agent"):
            mac("n" + c.capitalize(), f"{ev['per_class'][c]:,}".replace(",", "{,}"))
        for corp, v in ev["per_corpus"].items():
            mac("n" + corp, f"{v:,}".replace(",", "{,}"))

    # ---- features (R1) ----
    if rv:
        f = rv["R1_features"]
        mac("nFeatComputed", str(f["n_computed"])); mac("nFeatKept", str(f["n_kept"])); mac("nFeatInf", str(f["n_informative"]))
        groups = {}
        for n, g, _, st in FEAT_DEF:
            if st == "used":
                groups[g] = groups.get(g, 0) + 1
        mac("nFeatTiming", str(groups.get("timing", 0))); mac("nFeatStruct", str(groups.get("structure", 0)))
        mac("nFeatRedund", str(groups.get("redundancy", 0))); mac("nFeatThink", str(groups.get("think-time", 0)))

    # ---- single-feature rule (R2) ----
    if rv:
        reg = rv["R2_rule1"]["regions_full_data"]
        by = {r["class"]: r for r in reg}
        if "bot" in by:
            mac("ruleBotHi", f"{by['bot']['to_s']:.1f}")
        if "human" in by:
            mac("ruleHumanLo", f"{by['human']['from_s']:.1f}")

    # ---- in-distribution reference (R3) ----
    if rv:
        idr = rv["R3_in_distribution"]
        g = idr["gbdt"]
        mac("naiveMacroF", f"{g['macro_f1']:.3f}"); mac("naiveAucAgent", f"{g['agent_auc']:.3f}")
        mac("naiveTprAgent", pct(g["agent_tpr1"])); mac("naiveRecallAgent", pct(g["agent_argmax_recall"]))
        for k, tag in (("logistic", "Log"), ("gru", "Gru"), ("rule1", "Rule"), ("tfidf", "Tfidf")):
            mac("id" + tag + "Tpr", pct(idr[k]["agent_tpr1"])); mac("id" + tag + "Recall", pct(idr[k]["agent_argmax_recall"]))
            mac("id" + tag + "MacroF", f"{idr[k]['macro_f1']:.3f}")

    # ---- leak-free LOCO (R3) ----
    if rv:
        L = rv["R3_loco"]
        gb = L["gbdt"]
        for h in HELD:
            if h in gb:
                v = gb[h]
                mac("loco" + h, pct(v["argmax_recall"]))
                mac("loco" + h + "Lo", pct(v["argmax_ci"][0])); mac("loco" + h + "Hi", pct(v["argmax_ci"][1]))
                mac("loco" + h + "Tpr", pct(v["tpr1"]))
                mac("loco" + h + "TprLo", pct(v["tpr1_ci"][0])); mac("loco" + h + "TprHi", pct(v["tpr1_ci"][1]))
                mac("loco" + h + "Human", pct(v["agent_called_human"]))
                mac("loco" + h + "N", str(v["n"])); mac("loco" + h + "G", str(v["n_groups"]))
        rec = [gb[h]["argmax_recall"] for h in HELD if h in gb]; tp = [gb[h]["tpr1"] for h in HELD if h in gb]
        mac("locoMin", pct(min(rec))); mac("locoMax", pct(max(rec)))
        mac("locoTprMin", pct(min(tp))); mac("locoTprMax", pct(max(tp)))
        for k, tag in (("gbdt", "Gbdt"), ("logistic", "Log"), ("gru", "Gru"), ("rule1", "Rule"), ("tfidf", "Tfidf"), ("reworr", "Reworr")):
            if L.get(k) and "mean_argmax_recall" in L[k]:
                mac("locoMean" + tag, pct(L[k]["mean_argmax_recall"])); mac("locoWMean" + tag, pct(L[k]["wmean_argmax_recall"]))
            if L.get(k) and "mean_tpr1" in L[k]:
                mac("locoTprMean" + tag, pct(L[k]["mean_tpr1"])); mac("locoTprWMean" + tag, pct(L[k]["wmean_tpr1"]))
        for k, tag in (("gbdt", "Gbdt"), ("logistic", "Log"), ("gru", "Gru"), ("rule1", "Rule"), ("tfidf", "Tfidf"), ("reworr", "Reworr")):
            for h in HELD:
                if h in L.get(k, {}):
                    mac("loco" + tag + h, pct(L[k][h]["argmax_recall"]))
                    if "tpr1" in L[k][h]:
                        mac("loco" + tag + h + "Tpr", pct(L[k][h]["tpr1"]))
        rw = [L["reworr"][h]["rule_fpr_on_negatives"] for h in HELD if h in L["reworr"]]
        if rw:
            mac("reworrFpr", pct(rw[0]))
        hum = [gb[h]["agent_called_human"] for h in HELD if h in gb]
        mac("locoHumanMin", pct(min(hum))); mac("locoHumanMax", pct(max(hum)))

    # ---- OOD probes (R4) ----
    if rv:
        o = rv["R4_ood"]
        for h, v in o.get("novelty_isoforest", {}).items():
            mac("nov" + h, pct(v["tpr1"]))
        if "unseen_human_subpop_kypo_server" in o:
            v = o["unseen_human_subpop_kypo_server"]
            mac("oodServerN", str(v["n"])); mac("oodServerAgent", pct(v["called_agent"])); mac("oodServerHuman", pct(v["called_human"]))
            mac("oodServerBot", pct(v["called_bot"]))
        if "unseen_bot_period_cyberlab_2020" in o:
            v = o["unseen_bot_period_cyberlab_2020"]
            mac("oodBotN", f"{v['n']:,}".replace(",", "{,}")); mac("oodBotAgent", pct(v["called_agent"])); mac("oodBotBot", pct(v["called_bot"]))

    # ---- confound (unchanged sources) ----
    if conf:
        mac("srcDtuAipotAuc", f"{conf['T2']['dtu_vs_aipot_auc_leave_model_out']:.3f}")
        mac("harnessFloor", f"{conf['T4']['dtu_exec_median_s']:.2f}")
        mac("botFastShare", pct(conf["T4"]["bot_share_iv_below_1p2s"]))
    if ev:
        if "E2_within_agent_source_auc" in ev:
            mac("srcThreeway", f"{ev['E2_within_agent_source_auc']['mean_ovr_auc']:.3f}")
        if "E2_kypo_hostrole_auc" in ev:
            mac("kypoHostAuc", f"{ev['E2_kypo_hostrole_auc']:.3f}")
        if "E6b_taskmatched_human_vs_agent" in ev:
            tm = ev["E6b_taskmatched_human_vs_agent"]
            mac("taskMatchAuc", f"{tm['auc']:.3f}"); mac("taskMatchTpr", pct(tm["tpr1_agent"]))
            mac("taskMatchNh", str(tm["n_human_server"])); mac("taskMatchNa", str(tm["n_agent_own"]))
        if "E3_bot_temporal_recall_2020" in ev:
            mac("botTemporalRecall", pct(ev["E3_bot_temporal_recall_2020"]))
        e = ev["E7_early"]
        mac("earlyThree", pct(e["3"]["tpr1_agent"])); mac("earlyTen", pct(e["10"]["tpr1_agent"]))

    # ---- adaptive (R5) ----
    if rv:
        a = rv["R5_adaptive"]
        mac("advClean", pct(a["clean_agent_tpr1"])); mac("advCleanArgmax", pct(a["clean_agent_argmax_recall_3class"]))
        st, sa, sh = a["scale_tpr1"], a["scale_argmax_recall_3class"], a["scale_argmax_called_human_3class"]
        mac("scaleFastTpr", pct(st["0.05"])); mac("scaleSlowTpr", pct(st["10.0"]))
        mac("scaleFastArgmax", pct(sa["0.05"])); mac("scaleSlowArgmax", pct(sa["10.0"])); mac("scaleSlowHuman", pct(sh["10.0"]))
        mac("scaleTwoTpr", pct(st["2.0"])); mac("scaleHalfTpr", pct(st["0.5"]))
        mac("advWhitebox", pct(a["whitebox_addonly"]["residual_tpr1"])); mac("advWhiteboxInfl", f"{a['whitebox_addonly']['median_inflation']:.1f}")
        mac("advWhiteboxTen", pct(a["whitebox_addonly_infl_le_10"]["residual_tpr1"]))
        mac("advWhiteboxTenInfl", f"{a['whitebox_addonly_infl_le_10']['median_inflation']:.1f}")
        mac("advHuman", pct(a["human_replay"]["residual_tpr1"])); mac("advHumanInfl", f"{a['human_replay']['median_inflation']:.1f}")
        mac("advAddOnly", pct(a["add_only_delay"]["residual_tpr1"])); mac("advAddOnlyInfl", f"{a['add_only_delay']['median_inflation']:.1f}")
        mac("advStruct", pct(a["structure_dechain"]["residual_tpr1"]))
        mac("wbGridF", str(len({p[0] for p in a["whitebox_grid"]}))); mac("wbGridN", str(len(a["whitebox_grid"])))

    # ---- mechanism (R6) ----
    if rv:
        rows = rv["R6_mech"]["rows"]
        def rng(pred):
            v = [r["rho_med"] for k, r in rows.items() if pred(k, r) and r["rho_med"] is not None]
            return (min(v), max(v)) if v else (None, None)
        for tag, pred in (("Local", lambda k, r: r["serving"] == "local" and not k.startswith("own")),
                          ("Own", lambda k, r: k.startswith("own")),
                          ("Api", lambda k, r: r["serving"] == "API/hosted"),
                          ("Human", lambda k, r: k.startswith("kypo"))):
            lo, hi = rng(pred)
            if lo is not None:
                mac("mech" + tag + "Lo", f"{lo:+.2f}"); mac("mech" + tag + "Hi", f"{hi:+.2f}")
        if "cyberlab" in rows and rows["cyberlab"]["rho_med"] is not None:
            mac("mechBot", f"{rows['cyberlab']['rho_med']:+.2f}")
        loc = {k: r["rho_med"] for k, r in rows.items() if r["serving"] == "local" and r["rho_med"] is not None}
        noq = [v for k, v in loc.items() if "qwen3" not in k]
        mac("mechLocalN", str(len(loc))); mac("mechLocalNoQN", str(len(noq)))
        mac("mechLocalNoQLo", f"{min(noq):+.2f}"); mac("mechLocalNoQHi", f"{max(noq):+.2f}")
        for k, v in loc.items():
            if "qwen3:30b" in k:
                mac("mechQwenThirty", f"{v:+.2f}")
            if "qwen3:4b" in k:
                mac("mechQwenFour", f"{v:+.2f}")
        mac("mechApiN", str(sum(1 for r in rows.values() if r["serving"] == "API/hosted")))
        for k, r in rows.items():
            if k.startswith("dtu:deepseek-r1") and r["rho_med"] is not None:
                mac("mechDeepseek" + ("Small" if "1.5b" in k else "Large"), f"{r['rho_med']:+.2f}")
        mac("mechSumSessions", f"{rv['R6_mech']['sum_sessions']:,}".replace(",", "{,}"))
        if "kypo:server" in rows:
            mac("mechServerN", str(rows["kypo:server"]["n_sessions"]))

    # ---- own conditions (R7) ----
    if rv:
        cd = rv["R7_own_conditions"]
        if cd:
            mac("condAlarmMin", pct(min(v["loco_own_above_thr"] for v in cd.values())))
            mac("condIdMin", pct(min(v["in_dist_tpr1"] for v in cd.values())))
        for c, v in cd.items():
            t = c.capitalize()
            mac("cond" + t + "N", str(v["n"])); mac("cond" + t + "Tpr", pct(v["in_dist_tpr1"]))
            mac("cond" + t + "Loco", pct(v["loco_own_argmax_recall"])); mac("cond" + t + "LocoThr", pct(v["loco_own_above_thr"]))
            mac("cond" + t + "Human", pct(v["loco_own_called_human"]))
            mac("cond" + t + "Iv", f"{v['median_iv_s']:.1f}")
            if v.get("duration_vs_base") is not None:
                mac("cond" + t + "Dur", f"{v['duration_vs_base']:.2f}")

    # ---- timestamps (R8) ----
    if rv:
        ts = rv["R8_timestamps"]
        pc = ts["per_corpus"]
        if "kypo" in pc:
            mac("kypoIntSessions", pct(pc["kypo"]["frac_sessions_integer_seconds"]))
            mac("kypoNonintTs", pct(pc["kypo"]["frac_nonint_timestamps"]))
        rb = ts["coarsened_1s"]
        mac("robMacroF", f"{rb['in_dist_macro_f1']:.3f}"); mac("robTpr", pct(rb["in_dist_agent_tpr1"]))
        for h in HELD:
            if f"loco_{h}_argmax_recall" in rb:
                mac("rob" + h, pct(rb[f"loco_{h}_argmax_recall"])); mac("rob" + h + "Tpr", pct(rb[f"loco_{h}_tpr1"]))

    with open(os.path.join(OUT, "numbers_A.tex"), "w", encoding="ascii") as fh:
        for k, v in M.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    if not rv:
        print("a_revision.json missing - figures/tables not regenerated"); return

    # ---- Figure 2: leak-free LOCO (both metrics) + scale sweep (one detector) ----
    L, a = rv["R3_loco"], rv["R5_adaptive"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    idg = rv["R3_in_distribution"]["gbdt"]
    labels = ["in-dist."] + [f"held out\n{HNAME[h]}" for h in HELD if h in L["gbdt"]]
    rec = [idg["agent_argmax_recall"]] + [L["gbdt"][h]["argmax_recall"] for h in HELD if h in L["gbdt"]]
    tpr = [idg["agent_tpr1"]] + [L["gbdt"][h]["tpr1"] for h in HELD if h in L["gbdt"]]
    x = np.arange(len(labels)); w = 0.38
    ax[0].bar(x - w / 2, [100 * v for v in rec], w, color=C["agent"], label="3-class agent recall")
    ax[0].bar(x + w / 2, [100 * v for v in tpr], w, color=C["g2"], label="agent TPR at 1% FPR")
    for i, h in enumerate([None] + [h for h in HELD if h in L["gbdt"]]):
        if h:
            lo, hi = L["gbdt"][h]["argmax_ci"]
            ax[0].errorbar(i - w / 2, 100 * rec[i], yerr=[[100 * (rec[i] - lo)], [100 * (hi - rec[i])]], fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("agent detection (%)"); ax[0].set_ylim(0, 128); ax[0].set_yticks([0, 20, 40, 60, 80, 100])
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
    ax[0].set_title("(a) in-distribution vs leave-one-corpus-out", fontsize=8.5)
    xs = sorted(float(k) for k in a["scale_tpr1"])
    ax[1].plot(xs, [100 * a["scale_tpr1"][str(v)] for v in xs], "o-", color=C["g2"], ms=4, label="agent TPR at 1% FPR")
    ax[1].plot(xs, [100 * a["scale_argmax_recall_3class"][str(v)] for v in xs], "s--", color=C["agent"], ms=3.5, label="3-class agent recall")
    ax[1].set_xscale("log"); ax[1].axvline(1.0, ls=":", color="gray")
    ax[1].set_xlabel("agent time-scale factor $a$ (intervals $\\times a$)"); ax[1].set_ylabel("agent detection (%)"); ax[1].set_ylim(0, 105)
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower center"); ax[1].set_title("(b) timing scale sweep (one detector)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "a_fig_confound.pdf")); plt.close(fig)

    # ---- Figure 3: adaptive attackers (one cross-fitted detector) + early detection ----
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.7, 1]})
    items = [("clean", a["clean_agent_tpr1"], 1.0), ("de-chain", a["structure_dechain"]["residual_tpr1"], 1.0),
             ("random\ndelay", a["add_only_delay"]["residual_tpr1"], a["add_only_delay"]["median_inflation"]),
             ("white-box\n$\\leq$10x cost", a["whitebox_addonly_infl_le_10"]["residual_tpr1"], a["whitebox_addonly_infl_le_10"]["median_inflation"]),
             ("white-box\nany cost", a["whitebox_addonly"]["residual_tpr1"], a["whitebox_addonly"]["median_inflation"]),
             ("human\nreplay", a["human_replay"]["residual_tpr1"], a["human_replay"]["median_inflation"])]
    ax[0].bar(range(len(items)), [100 * t for _, t, _ in items], color=[C["g1"]] + [C["agent"]] * (len(items) - 1))
    for i, (_, t, f) in enumerate(items):
        ax[0].text(i, 100 * t + 2, f"x{f:.1f}" if i else "", ha="center", fontsize=6.3)
    ax[0].axhline(100 * a["scale_tpr1"]["10.0"], ls="--", color="#555", lw=0.8)
    ax[0].text(len(items) - 1, 100 * a["scale_tpr1"]["10.0"] + 2, "uniform\n10x", ha="center", va="bottom", fontsize=6, color="#555")
    ax[0].set_xticks(range(len(items))); ax[0].set_xticklabels([n for n, _, _ in items], fontsize=6.3)
    ax[0].set_ylabel("residual agent TPR at 1% FPR (%)"); ax[0].set_ylim(0, 112)
    ax[0].set_title("(a) adaptive attackers (x = median time cost)", fontsize=8.5)
    e = rv.get("_early") or J("a_eval.json")["E7_early"]
    ks = sorted(int(k) for k in e); ax[1].plot(ks, [100 * e[str(k)]["tpr1_agent"] for k in ks], "s-", color=C["g1"])
    ax[1].set_xlabel("commands observed"); ax[1].set_ylabel("agent TPR at 1% FPR (%)"); ax[1].set_ylim(0, 105)
    ax[1].set_title("(b) early detection (in-distribution)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "a_fig_adaptive.pdf")); plt.close(fig)

    # ---- Table: leak-free LOCO for every model ----
    nf = rv["R1_features"]["n_informative"]
    names = [("logistic", f"Logistic regression ({nf} feat.)"), ("gbdt", f"Gradient-boosted trees ({nf} feat.)"),
             ("gru", "GRU (per-command sequence)"), ("rule1", "Single-feature rule (median interval)"),
             ("tfidf", "TF-IDF content model"), ("reworr", "Reworr--Volkov window (1--2\\,s)")]
    idr = rv["R3_in_distribution"]
    with open(os.path.join(OUT, "tab_loco.tex"), "w", encoding="utf-8") as t:
        t.write("\\begin{tabular}{lrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{Held out (LOCO)} & LOCO \\\\\n\\cmidrule(lr){3-5}\n")
        t.write("Detector & In-dist. & DTU & AIPot & Own & mean \\\\\n\\midrule\n")
        for block, met, idk in (("3-class agent recall (\\%)", "argmax_recall", "agent_argmax_recall"),
                                ("Agent TPR at 1\\% FPR, held-out threshold (\\%)", "tpr1", "agent_tpr1")):
            t.write("\\multicolumn{6}{l}{\\textit{%s}} \\\\\n" % block)
            for k, nm in names:
                d = L.get(k, {})
                if met == "tpr1" and k == "reworr":
                    continue
                cells = [pct(idr[k][idk]) if k in idr and idk in idr[k] else "--"]
                cells += [pct(d[h][met]) if h in d and met in d[h] else "--" for h in HELD]
                cells.append(pct(d[f"mean_{met}"]) if f"mean_{met}" in d else "--")
                t.write("\\quad %s & %s \\\\\n" % (nm, " & ".join(cells)))
        t.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: mechanistic cue on the benchmark sample ----
    mech = rv["R6_mech"]["rows"]
    order = [k for k in mech if k.startswith("dtu:")] + [k for k in mech if k.startswith("own:")] + \
            [k for k in mech if k.startswith("aipot:")] + [k for k in mech if k.startswith("kypo")] + [k for k in mech if k == "cyberlab"]
    with open(os.path.join(OUT, "tab_mech.tex"), "w", encoding="utf-8") as t:
        t.write("\\begin{tabular}{llrrcr}\n\\toprule\nSource & Serving & Sessions & Valid & Median $\\rho$ [IQR] & Median think (s) \\\\\n\\midrule\n")
        for k in order:
            v = mech[k]
            src = "\\texttt{%s}" % k.replace("_", "\\_")
            rho = "--" if v["rho_med"] is None else f"${v['rho_med']:+.2f}$ [${v['rho_q25']:+.2f}$, ${v['rho_q75']:+.2f}$]"
            serv = {"local": "local", "API/hosted": "API", "n/a (no LLM)": "--"}[v["serving"]]
            t.write("%s & %s & %d & %d & %s & %.2f \\\\\n" % (src, serv, v["n_sessions"], v["n_valid_rho"], rho, v["think_med_s"] or 0))
        t.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: own-corpus prompting conditions ----
    cd = rv["R7_own_conditions"]
    label = {"base": "plain authorised-pentest prompt", "deceive": "counter-deception prompt",
             "mimic": "human-mimicking prompt", "jitter": "live timing jitter"}
    with open(os.path.join(OUT, "tab_conditions.tex"), "w", encoding="utf-8") as t:
        t.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & & Median & Duration & In-dist.\\ TPR & \\multicolumn{2}{c}{Own held out (LOCO)} \\\\\n")
        t.write("\\cmidrule(lr){6-7}\nCondition & $n$ & interval (s) & vs.\\ plain & at 1\\% FPR & agent recall & called human \\\\\n\\midrule\n")
        for c in ("base", "deceive", "mimic", "jitter"):
            if c in cd:
                v = cd[c]
                t.write("%s & %d & %.1f & %s & %s & %s & %s \\\\\n" % (label[c], v["n"], v["median_iv_s"],
                        f"{v['duration_vs_base']:.2f}$\\times$" if v.get("duration_vs_base") else "--",
                        pct(v["in_dist_tpr1"]), pct(v["loco_own_argmax_recall"]), pct(v["loco_own_called_human"])))
        t.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table: timestamp resolution per corpus ----
    pc = rv["R8_timestamps"]["per_corpus"]; rb = rv["R8_timestamps"]["coarsened_1s"]
    with open(os.path.join(OUT, "tab_resolution.tex"), "w", encoding="utf-8") as t:
        t.write("\\begin{tabular}{lrrrr}\n\\toprule\nCorpus & Sessions with & Non-integer & Min.\\ positive & Intervals \\\\\n")
        t.write(" & integer-second times & timestamps & interval (s) & $<0.1$\\,s \\\\\n\\midrule\n")
        for c in ("cyberlab", "kypo", "dtu", "aipot", "own"):
            if c in pc:
                v = pc[c]
                t.write("%s & %s\\%% & %s\\%% & %.3f & %s\\%% \\\\\n" % (c, pct(v["frac_sessions_integer_seconds"]), pct(v["frac_nonint_timestamps"]),
                        v["min_pos_interval_s"], pct(v["frac_iv_lt0p1"])))
        t.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Appendix table: every computed feature ----
    with open(os.path.join(OUT, "tab_features.tex"), "w", encoding="utf-8") as t:
        t.write("\\begin{tabular}{rllp{6.6cm}l}\n\\toprule\n\\# & Feature & Group & Definition (first $K$ commands) & Status \\\\\n\\midrule\n")
        for i, (n, g, dsc, st) in enumerate(FEAT_DEF, 1):
            t.write("%d & \\texttt{%s} & %s & %s & %s \\\\\n" % (i, n.replace("_", "\\_"), g, dsc, st))
        t.write("\\bottomrule\n\\end{tabular}\n")
    print("a_report done ->", OUT)


if __name__ == "__main__":
    main()
