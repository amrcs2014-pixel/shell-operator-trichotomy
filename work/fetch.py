"""Resumable downloader for models, binaries and datasets (stdlib only, safe to re-run; used by the p1516_fetch scheduled task)."""
import os, sys, time, urllib.request, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (research downloader)"}
LLAMA = "b11370"
ZEN = "https://zenodo.org/api/records/{}/files/{}/content"
HF = "https://huggingface.co/{}/resolve/main/{}"
CYBERLAB_DAYS = ["2019-06-15", "2019-08-15", "2019-10-15", "2019-12-27", "2020-01-20", "2020-02-25"]

JOBS = [
    (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/llama-{LLAMA}-bin-win-cuda-12.4-x64.zip", "bin/llama.zip"),
    (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/cudart-llama-bin-win-cuda-12.4-x64.zip", "bin/cudart.zip"),
    (HF.format("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"), "models/qwen2.5-3b-instruct-q4_k_m.gguf"),
] + [(ZEN.format("3687527", f"cyberlab_{d}.json.gz"), f"data/cyberlab/cyberlab_{d}.json.gz") for d in CYBERLAB_DAYS] + [
    (HF.format("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"), "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    (HF.format("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "qwen2.5-coder-3b-instruct-q4_k_m.gguf"), "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
    (HF.format("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"), "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"),
]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(ROOT, "logs", "fetch.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(url, dest):
    dest = os.path.join(ROOT, dest)
    if os.path.exists(dest):
        return True
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(8):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        hdr = dict(UA)
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                if have and r.status != 206:      # server ignored the range request: start again
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                t0, got = time.time(), 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b); got += len(b)
                if total and os.path.getsize(part) < total:
                    raise IOError("short read")
            os.replace(part, dest)
            log(f"done {os.path.basename(dest)} {os.path.getsize(dest)/1e6:.1f} MB ({got/1e6/max(time.time()-t0,1e-9):.2f} MB/s)")
            return True
        except Exception as e:
            if "416" in str(e) and os.path.exists(part):      # already complete
                os.replace(part, dest); return True
            log(f"retry {attempt} {os.path.basename(dest)}: {e!r}")
            time.sleep(10)
    return False


def main():
    lock = os.path.join(ROOT, "logs", "fetch.lock")
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 300:
        return                                   # another instance is alive (it touches the lock per file)
    ok = True
    for url, dest in JOBS:
        open(lock, "w").write(str(os.getpid()))
        ok &= fetch(url, dest)
    bindir = os.path.join(ROOT, "bin", "llama")
    if ok and not os.path.exists(os.path.join(bindir, "llama-server.exe")):
        for z in ("llama.zip", "cudart.zip"):
            with zipfile.ZipFile(os.path.join(ROOT, "bin", z)) as zf:
                zf.extractall(bindir)
        log("llama.cpp unpacked: " + ", ".join(sorted(f for f in os.listdir(bindir) if f.endswith(".exe"))[:12]))
    if os.path.exists(lock):
        os.remove(lock)
    log("ALL DONE" if ok else "INCOMPLETE")


if __name__ == "__main__":
    main()
