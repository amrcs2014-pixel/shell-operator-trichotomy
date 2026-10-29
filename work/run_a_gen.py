"""Resumable launcher for the AgentPrint own-corpus generation (scheduled task p1516_agentgen).

For each model: ensure the WSL sshd target is up, start llama-server on the GPU, generate all missing
sessions (a_gen.py), stop the server, move on. Idempotent: a_gen skips sessions already in own.jsonl.gz,
and a single-instance lock lets the 10-minute task trigger relaunch it after a session restart.
"""
import gzip, json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "pylib"))
PY = sys.executable
EXE = os.path.join(ROOT, "bin", "llama", "llama-server.exe")
OWN = os.path.join(ROOT, "data", "sessions", "own.jsonl.gz")
LOGS = os.path.join(ROOT, "logs"); os.makedirs(LOGS, exist_ok=True)
LOCK = os.path.join(LOGS, "agentgen.lock")
KEEPALIVE = "/opt/agentprint/ap_keepalive.sh"      # self-contained, no space-path (staged once)
MODELS = [("qwen2.5-3b", "qwen2.5-3b-instruct-q4_k_m.gguf"),
          ("qwen2.5-coder-3b", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
          ("llama-3.2-3b", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
          ("phi-3.5-mini", "Phi-3.5-mini-instruct-Q4_K_M.gguf")]
SEEDS = 30
CONDS = 4
PORT = 8080


def log(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    open(os.path.join(LOGS, "agentgen.log"), "a", encoding="utf-8").write(line + "\n")


def have():
    s = {}
    if os.path.exists(OWN):
        with gzip.open(OWN, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    s[json.loads(line)["sid"]] = 1
                except Exception:
                    pass
    return s


def count(model):
    h = have()
    return sum(1 for c in range(CONDS) for s in range(SEEDS)
               for cond in ["base", "deceive", "mimic", "jitter"][:1]) and \
        sum(k.startswith(f"own/{model}/") for k in h)


def start_keepalive():
    """Launch the persistent WSL keepalive (holds the VM + target up). Return the Popen."""
    ka = subprocess.Popen(["wsl", "-d", "Ubuntu-22.04", "-u", "root", "--", "bash", KEEPALIVE],
                          stdout=open(os.path.join(LOGS, "keepalive.out"), "w"),
                          stderr=open(os.path.join(LOGS, "keepalive.err"), "w"))
    for _ in range(40):
        time.sleep(3)
        try:
            import socket
            s = socket.create_connection(("127.0.0.1", 2222), timeout=2); s.close()
            log("target reachable on 2222"); return ka
        except Exception:
            pass
    log("WARNING: target not reachable after keepalive start")
    return ka


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def main():
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 900:
        return
    open(LOCK, "w").write(str(os.getpid()))
    ka = None
    try:
        ka = start_keepalive()
        for model, gguf in MODELS:
            path = os.path.join(ROOT, "models", gguf)
            if not os.path.exists(path):
                log(f"skip {model}: model file missing"); continue
            done = sum(k.startswith(f"own/{model}/") for k in have())
            if done >= SEEDS * CONDS:
                log(f"{model}: complete ({done})"); continue
            log(f"=== {model}: {done}/{SEEDS*CONDS} done; starting server ===")
            srv = subprocess.Popen([EXE, "-m", path, "-ngl", "99", "-c", "8192", "-np", "4",
                                    "--port", str(PORT), "--host", "127.0.0.1"],
                                   stdout=open(os.path.join(LOGS, f"srv_{model}.out"), "w"),
                                   stderr=open(os.path.join(LOGS, f"srv_{model}.err"), "w"))
            t0 = time.time()
            while time.time() - t0 < 300 and not healthy():
                time.sleep(3)
                open(LOCK, "w").write(str(os.getpid()))
            if not healthy():
                log(f"{model}: server failed to start"); srv.terminate(); continue
            env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "pylib"))
            r = subprocess.run([PY, os.path.join(ROOT, "src", "a_gen.py"), "--model", model,
                                "--url", f"http://127.0.0.1:{PORT}", "--seeds", str(SEEDS), "--workers", "4"],
                               env=env, cwd=os.path.join(ROOT, "src"))
            srv.terminate()
            try:
                srv.wait(20)
            except Exception:
                srv.kill()
            time.sleep(3)
            log(f"{model}: now {sum(k.startswith(f'own/{model}/') for k in have())}/{SEEDS*CONDS}")
        total = sum(k.startswith("own/") for k in have())
        log(f"ALL DONE: {total} own sessions" if total >= len(MODELS) * SEEDS * CONDS else f"PARTIAL: {total} own sessions")
    finally:
        if ka is not None:
            ka.terminate()
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()
