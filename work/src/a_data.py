"""Paper A - unified session corpus.

Every source is converted to the same record (one JSON object per line, gzip):
  sid, corpus, label (bot|human|agent|wild), group (unit for grouped splits), sub (dict of source-specific factors),
  pre (seconds from login/session start to the first command, or null),
  cmds: list of [t, cmd, out_len, t_out, think]
        t      seconds since the first command of the session
        cmd    command line as received by the shell
        out_len / t_out   length of the output and time it was complete (null when the source has no outputs)
        think  LLM inference seconds reported by the harness (agents only, null otherwise)
"""
import collections, datetime as dt, glob, gzip, hashlib, json, os, re, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "sessions")
os.makedirs(OUT, exist_ok=True)


def ts(s):
    """ISO timestamp -> epoch seconds (float)."""
    s = s.strip().replace("Z", "+00:00").replace(" ", "T")
    if "+" not in s[10:] and "-" not in s[10:]:
        s += "+00:00"
    return dt.datetime.fromisoformat(s).timestamp()


def write(name, recs):
    path = os.path.join(OUT, name + ".jsonl.gz")
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"{name}: {n} sessions -> {path}", flush=True)
    return n


def read(name):
    with gzip.open(os.path.join(OUT, name + ".jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ----------------------------------------------------------------------------- DTU "Honey for the Agent"
def dtu():
    z = zipfile.ZipFile(os.path.join(DATA, "dtu", "Logs_final.zip"))
    for n in sorted(z.namelist()):
        if not n.endswith(".json"):
            continue
        try:
            s = json.loads(z.read(n))
        except Exception:
            continue
        _, prompt, maxturn, env, model, fn = n.split("/")
        cmds = []
        for t in s[1:]:
            if len(t) < 2 or not isinstance(t[0], list) or not isinstance(t[1], list):
                continue
            cmd = t[2][1] if len(t) > 2 and isinstance(t[2], list) and len(t[2]) > 1 and isinstance(t[2][1], str) else t[0][0]
            cmds.append([t[0][1] / 1000.0, str(cmd), len(t[1][0] or ""), t[1][1] / 1000.0, None])
        if not cmds:
            continue
        # think time = command time minus the time the previous output was complete (first: since session start)
        prev = 0.0
        for c in cmds:
            c[4] = round(c[0] - prev, 4); prev = c[3]
        t0 = cmds[0][0]
        for c in cmds:
            c[0] = round(c[0] - t0, 4); c[3] = round(c[3] - t0, 4)
        yield {"sid": "dtu/" + n[len("Logs_final/"):-5], "corpus": "dtu", "label": "agent", "group": model,
               "sub": {"model": model, "prompt": prompt, "maxturn": int(maxturn), "env": env}, "pre": round(t0, 4), "cmds": cmds}


# ----------------------------------------------------------------------------- Erdem AIPot (attacker-console agents)
def aipot():
    z = zipfile.ZipFile(os.path.join(DATA, "aipot", "aipot-paper1-zenodo.zip"))
    for n in sorted(z.namelist()):
        if "consistency-" not in n or not n.endswith(".json"):
            continue
        r = json.loads(z.read(n))
        start = ts(r["start_time"]); cmds = []
        for it in r.get("interactions", []):
            cmd = it.get("parsed_command")
            if not cmd:
                continue
            cmds.append([ts(it["timestamp"]), str(cmd), len(it.get("command_output") or ""), None, float(it.get("inference_seconds") or 0)])
        if not cmds:
            continue
        t0 = cmds[0][0]
        for c in cmds:
            c[0] = round(c[0] - t0, 4)
        yield {"sid": "aipot/" + os.path.basename(n)[:-5], "corpus": "aipot", "label": "agent", "group": r["model"],
               "sub": {"model": r["model"], "provider": r.get("provider"), "prompt": r.get("prompt_style"), "env": "console"},
               "pre": round(t0 - start, 4), "cmds": cmds}


# ----------------------------------------------------------------------------- KYPO cyber-range trainees (humans)
def kypo(gap=600.0):
    """One session = consecutive commands of one trainee on one host with no pause longer than `gap` seconds."""
    z = zipfile.ZipFile(os.path.join(DATA, "kypo", "data.zip"))
    for n in sorted(z.namelist()):
        if not n.endswith(".json"):
            continue
        game, run = n.split("/")[0], n.split("/")[1]
        recs = []
        for ln in z.read(n).decode("utf-8", "replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("cmd") is None:
                continue
            recs.append((r.get("hostname") or "?", ts(r["timestamp_str"]), str(r["cmd"]), r.get("cmd_type") or "?"))
        trainee = f"{game}/{run}/{os.path.basename(n).split('-')[1]}"
        by_host = collections.defaultdict(list)
        for h, t, c, ty in recs:
            by_host[h].append((t, c, ty))
        for h, ev in by_host.items():
            ev.sort(key=lambda e: e[0])
            seg, k = [], 0
            for e in ev + [None]:
                if e is None or (seg and e[0] - seg[-1][0] > gap):
                    if seg:
                        t0 = seg[0][0]
                        yield {"sid": f"kypo/{trainee}/{h}/{k}", "corpus": "kypo", "label": "human", "group": trainee,
                               "sub": {"game": game, "host": h, "env": "console" if h == "attacker" else "victim",
                                       "msf": round(sum(x[2] == "msf-command" for x in seg) / len(seg), 3)},
                               "pre": None, "cmds": [[round(x[0] - t0, 4), x[1], None, None, None] for x in seg]}
                        k += 1
                    seg = []
                if e is not None:
                    seg.append(e)


# ----------------------------------------------------------------------------- Cowrie event logs (CyberLab honeynet and our own honeypots)
CMD_RE = re.compile(r"^CMD: ?(.*)$", re.S)


def cowrie_sessions(events_by_session, corpus, label):
    for sid, evs in events_by_session:
        cmds, t_login, t_conn, src, sensor, client, proto = [], None, None, None, None, None, None
        for e in evs:
            eid = e.get("eventid")
            if eid == "cowrie.session.connect":
                t_conn = ts(e["timestamp"]); src = e.get("src_ip_identifier") or e.get("src_ip"); sensor = e.get("sensor"); proto = e.get("protocol")
            elif eid == "cowrie.login.success" and t_login is None:
                t_login = ts(e["timestamp"])
            elif eid == "cowrie.client.version":
                client = e.get("ssh_client_version") or e.get("version")
            elif eid == "cowrie.command.input":
                c = e.get("input")
                if c is None:
                    m = CMD_RE.match(e.get("message") or "")
                    c = m.group(1) if m else (e.get("message") or "")
                cmds.append([ts(e["timestamp"]), str(c), None, None, None])
        if not cmds:
            continue
        cmds.sort(key=lambda c: c[0])
        t0 = cmds[0][0]
        ref = t_login if t_login is not None else t_conn
        for c in cmds:
            c[0] = round(c[0] - t0, 4)
        yield {"sid": f"{corpus}/{sid}", "corpus": corpus, "label": label, "group": str(src),
               "sub": {"sensor": sensor, "client": str(client)[:60] if client else None, "proto": proto, "env": "cowrie"},
               "pre": None if ref is None else round(t0 - ref, 4), "cmds": cmds}


def cyberlab():
    for fn in sorted(glob.glob(os.path.join(DATA, "cyberlab", "cyberlab_*.json.gz"))):
        day = os.path.basename(fn)[9:19]
        with gzip.open(fn, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        def gen():
            for obj in data:
                for sid, evs in obj.items():
                    yield f"{day}/{sid}", evs
        n = 0
        for s in cowrie_sessions(gen(), "cyberlab", "wild"):
            s["sub"]["day"] = day; n += 1
            yield s
        print("  cyberlab", day, len(data), "sessions,", n, "with commands", flush=True)
        del data


def cowrie_jsonl(paths, corpus, label, sub=None):
    """Plain Cowrie cowrie.json(.gz) event logs (one event per line): the 160-instance deployment and our own honeypots."""
    by = collections.OrderedDict()
    for fn in paths:
        op = gzip.open if fn.endswith(".gz") else open
        with op(fn, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                s = e.get("session")
                if s:
                    by.setdefault(s, []).append(e)
    for s in cowrie_sessions(by.items(), corpus, label):
        if sub:
            s["sub"].update(sub)
        yield s


def seq_hash(cmds):
    """Identity of a command script: the ordered command lines with digits and long hex/base64 blobs masked."""
    norm = [re.sub(r"[A-Za-z0-9+/=]{24,}", "B", re.sub(r"\d+", "N", c[1].strip())) for c in cmds]
    return hashlib.md5("\n".join(norm).encode("utf-8", "replace")).hexdigest()[:16]


if __name__ == "__main__":
    which = sys.argv[1:] or ["dtu", "aipot", "kypo", "cyberlab"]
    for w in which:
        write(w, globals()[w]())
