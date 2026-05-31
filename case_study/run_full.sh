#!/usr/bin/env bash
# run_full.sh — run the WHOLE case study in FULL mode, one log file per section.
#
# Usage:   bash case_study/run_full.sh            # run everything
#          bash case_study/run_full.sh 03 06      # run only sections 03 and 06
#
# Logs land in case_study/logs/NN_*.log. Each section is self-sufficient + idempotent, so re-running
# is safe; training sections pass --force so FULL retrains over any cached TRIAL artifacts.
# HF sections use .venv-rl; Unsloth uses .venv (§3 unsloth, §8 shells out itself).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

export CASE_STUDY_MODE=full
HF="../.venv-rl/bin/python"
US="../.venv/bin/python"
ts() { date +%H:%M:%S; }

# run <id> <logname> <python> <script> [args...]
run() {
  local id="$1" log="logs/$2"; shift 2
  if [ "$SECTIONS_SET" = "1" ] && ! grep -qw "$id" <<<"$SECTIONS"; then return; fi
  echo "[$(ts)] ===== §$id START -> $log ====="
  "$@" >"$log" 2>&1
  echo "[$(ts)] ===== §$id DONE (exit $?) ; tail: ====="
  tail -n 3 "$log" | sed 's/^/    /'
}

SECTIONS="$*"; SECTIONS_SET=$([ -n "$SECTIONS" ] && echo 1 || echo 0)

echo "[$(ts)] FULL run starting. CASE_STUDY_MODE=$CASE_STUDY_MODE"
run 01 01_corpus.log        $HF 01_build_corpus.py
run 02 02_data.log          $HF 02_data_availability.py
run 03 03_cpt_hf.log        $HF 03_cpt.py --backend hf --force
run 03 03_cpt_unsloth.log   $US 03_cpt.py --backend unsloth --force
run 04 04_forgetting.log    $HF 04_cpt_base_vs_instruct.py --force
run 05 05_sft.log           $HF 05_sft.py --force
run 06 06_sweep.log         $HF 06_base_vs_instruct_sweep.py --force
run 07 07_eval.log          $HF 07_eval.py --force
run 08 08_bench.log         $HF 08_unsloth_vs_hf.py --force
run 09 09_merge.log         $HF 09_merge_and_gguf.py
run 10 10_ollama.log        $HF 10_ollama_deploy.py
run 11 11_edge.log          $HF 11_edge_benchmark.py
run 12 12_harness.log       $HF 12_harness.py
run 13 13_smollm3.log       $HF 13_smollm3_toolcall.py
echo "[$(ts)] ALL DONE. Read logs in case_study/logs/  (then tell Claude: 'full run done')"
