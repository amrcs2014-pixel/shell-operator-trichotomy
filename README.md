# shell-operator-trichotomy

Analysis code for a cross-corpus evaluation of shell sessions operated by scripted bots, human operators and
LLM-driven agents, using only vocabulary-agnostic behavioural features (inter-command timing and the structure of
the command stream; no command vocabulary).

## Contents
- `work/fetch.py` — downloads the public corpora; `work/src/a_data.py` — converts them to one session format
- `work/src/a_feat.py` — vocabulary-agnostic session features
- `work/src/a_clf.py`, `a_confound.py`, `a_mech.py`, `a_eval.py`, `a_revision.py`, `a_stats.py` — evaluations
- `work/src/a_gen.py`, `work/run_a_gen.py` — ReAct agent-generation harness (local models via llama.cpp,
  isolated container)
- `work/src/a_report.py` — renders figures and tables from the result files
- `work/src/requirements_A.txt` — pinned environment (Python 3.11)

## Data
Public corpora under CC BY 4.0, not redistributed here: CyberLab honeynet (Zenodo 3687527), KYPO shell commands
(Zenodo 8136017), "Honey for the Agent" (Zenodo 20818246), AIPot (Zenodo 20421592).
