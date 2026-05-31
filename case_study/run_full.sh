#!/usr/bin/env bash
# run_full.sh — run the WHOLE case study in FULL mode, one log file per section.
#
# Usage:
#   bash case_study/run_full.sh                 # 135M pipeline (bf16; HF env, §3-unsloth + §8 use .venv)
#   bash case_study/run_full.sh smollm3         # SmolLM3-3B pipeline (QLoRA 4-bit, all in .venv)
#   bash case_study/run_full.sh smollm2-135m 06 07   # only sections 06 and 07 of the 135M run
#
# Logs: case_study/logs/<model>/NN_*.log. Self-sufficient + idempotent; --force retrains over TRIAL.
set -uo pipefail
cd "$(dirname "$0")"

export CASE_STUDY_MODE=full
HF="../.venv-rl/bin/python"     # trl 1.x, no bitsandbytes  (135M bf16)
US="../.venv/bin/python"        # unsloth + trl 0.24 + bitsandbytes  (SmolLM3 QLoRA)
ts() { date +%H:%M:%S; }

MODEL="${1:-smollm2-135m}"
case "$MODEL" in smollm2-135m|smollm3) shift || true ;; *) MODEL="smollm2-135m" ;; esac
export CASE_STUDY_MODEL="$MODEL"
SECTIONS="$*"; SECTIONS_SET=$([ -n "$SECTIONS" ] && echo 1 || echo 0)
LOGDIR="logs/$MODEL"; mkdir -p "$LOGDIR"

# run <id> <logname> <python> <script> [args...]
run() {
  local id="$1" log="$LOGDIR/$2"; shift 2
  if [ "$SECTIONS_SET" = "1" ] && ! grep -qw "$id" <<<"$SECTIONS"; then return; fi
  echo "[$(ts)] ===== §$id START ($MODEL) -> $log ====="
  "$@" >"$log" 2>&1
  echo "[$(ts)] ===== §$id DONE (exit $?) ; tail: ====="
  tail -n 3 "$log" | sed 's/^/    /'
}

echo "[$(ts)] FULL run: MODEL=$MODEL  MODE=$CASE_STUDY_MODE"

if [ "$MODEL" = "smollm3" ]; then
  # SmolLM3-3B: QLoRA 4-bit — every training/eval section runs in .venv (bitsandbytes + unsloth).
  run 01 01_corpus.log     $US 01_build_corpus.py
  run 02 02_data.log       $US 02_data_availability.py
  run 03 03_cpt.log        $US 03_cpt.py --backend unsloth --force
  run 04 04_forgetting.log $US 04_cpt_base_vs_instruct.py --force
  run 05 05_sft.log        $US 05_sft.py --force
  run 06 06_sweep.log      $US 06_base_vs_instruct_sweep.py --force
  run 07 07_eval.log       $US 07_eval.py --force
  # §8 (HF-vs-Unsloth backend bench) is a 135M comparison — skipped for SmolLM3.
  run 09 09_merge.log      $US 09_merge_and_gguf.py
  run 10 10_ollama.log     $US 10_ollama_deploy.py
  run 11 11_edge.log       $US 11_edge_benchmark.py
  run 12 12_harness.log    $US 12_harness.py
  run 13 13_smollm3.log    $US 13_smollm3_toolcall.py   # fine-tuned SmolLM3 vs stock vs 135M
else
  # 135M (bf16): HF env, with the Unsloth bits where they belong.
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
fi
echo "[$(ts)] ALL DONE ($MODEL). Logs in case_study/$LOGDIR/  (then tell Claude: 'full run done')"
