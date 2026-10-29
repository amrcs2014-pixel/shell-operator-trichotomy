"""Quick descriptive statistics of the unified session corpora (sanity checks, not a paper table)."""
import collections, sys
import numpy as np
from a_data import read, seq_hash


def q(v, ps=(10, 50, 90)):
    return " / ".join(f"{np.percentile(v, p):.2f}" for p in ps) if len(v) else "-"


for name in sys.argv[1:] or ["dtu", "aipot", "kypo", "cyberlab"]:
    S = list(read(name))
    n = np.array([len(s["cmds"]) for s in S])
    print(f"\n=== {name}: {len(S)} sessions, {n.sum()} commands, groups {len(set(s['group'] for s in S))}")
    print("  commands/session q10/50/90:", q(n), "| >=3:", int((n >= 3).sum()), ">=5:", int((n >= 5).sum()), ">=8:", int((n >= 8).sum()), ">=10:", int((n >= 10).sum()))
    iv = np.concatenate([np.diff([c[0] for c in s["cmds"]]) for s in S if len(s["cmds"]) > 1] or [np.array([])])
    print("  inter-command interval s q10/50/90:", q(iv), "| q1/q99:", q(iv, (1, 99)), "| share<0.05s:", round(float((iv < 0.05).mean()), 3) if len(iv) else "-")
    pre = [s["pre"] for s in S if s["pre"] is not None]
    print("  pre (login->first cmd) s:", q(pre))
    cl = [len(c[1]) for s in S for c in s["cmds"]]
    print("  command length chars:", q(cl))
    subs = collections.Counter()
    for s in S:
        for k in ("env", "model", "game", "host", "day", "proto"):
            if k in s["sub"]:
                subs[(k, s["sub"][k])] += 1
    print("  factors:", sorted(subs.items(), key=lambda x: (x[0][0], -x[1]))[:40])
    if name in ("cyberlab", "cowrie160"):
        h = collections.defaultdict(set); hs = collections.Counter()
        for s in S:
            k = seq_hash(s["cmds"]); h[k].add(s["group"]); hs[k] += 1
        rep = sum(1 for s in S if hs[seq_hash(s["cmds"])] >= 5 and len(h[seq_hash(s["cmds"])]) >= 3)
        print(f"  distinct scripts: {len(hs)} | sessions in replicated scripts (>=5 sessions, >=3 IPs): {rep} ({rep/len(S):.1%})")
        for k in (3, 5, 8, 10):
            sel = [s for s in S if len(s["cmds"]) >= k]
            repk = sum(1 for s in sel if hs[seq_hash(s["cmds"])] >= 5 and len(h[seq_hash(s["cmds"])]) >= 3)
            print(f"    >= {k} commands: {len(sel)} sessions, replicated {repk}, distinct scripts {len(set(seq_hash(s['cmds']) for s in sel))}, IPs {len(set(s['group'] for s in sel))}")
        top = hs.most_common(6)
        for k, c in top:
            ex = next(s for s in S if seq_hash(s["cmds"]) == k)
            print(f"    script x{c} from {len(h[k])} IPs, {len(ex['cmds'])} cmds, dt={[round(b[0]-a[0],2) for a,b in zip(ex['cmds'], ex['cmds'][1:])][:8]}: {[c[1][:50] for c in ex['cmds'][:4]]}")
        print("  clients:", collections.Counter(s["sub"].get("client") for s in S).most_common(8))
