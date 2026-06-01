#!/usr/bin/env bash
# overnight.sh — leave running all night. Runs the heavy comparisons unattended, each logged:
#   1) full pipeline for MiniCPM5-1B (small, tool-capable, Llama-arch → Ollama imports it directly)
#   2) CPT-vs-SFT side-by-side + equation-reproduction probe for MiniCPM5
#   3) LoRA-rank sweep (rsLoRA: r ∈ 8/16/32/64/128) for MiniCPM5 and SmolLM3
#
# Usage:  bash case_study/overnight.sh            # FULL (the real overnight run)
#         MODE=trial bash case_study/overnight.sh # quick smoke test of the whole batch (~minutes)
#
# Logs:    case_study/logs/overnight/<step>.log   |   results: outputs/<model>/{lora_sweep,compare,equation_probe}_*.json
# Models are cached after step 1, so steps 2–3 reuse them (no re-download). Safe to leave overnight.
set -uo pipefail
cd "$(dirname "$0")"

export CASE_STUDY_MODE="${MODE:-full}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
US="../.venv/bin/python"
L="logs/overnight"; mkdir -p "$L"
ts() { date '+%F %T'; }

step() {  # step <name> <cmd...>
  local name="$1"; shift
  echo; echo "════════════════════════════════════════════════════════════════"
  echo "[$(ts)] ▶ $name"
  echo "════════════════════════════════════════════════════════════════"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$L/$name.log"
  echo "[$(ts)] ✓ $name finished (exit ${PIPESTATUS[0]}, $((SECONDS - t0))s)"
}

echo "[$(ts)] OVERNIGHT START — mode=$CASE_STUDY_MODE"
[ -x "$US" ] || { echo "missing .venv python ($US)"; exit 1; }

# 1) Full pipeline for the new small, tool-capable model (downloads MiniCPM5-1B on first use)
step minicpm5_pipeline   bash run.sh minicpm5

# 2) CPT-vs-SFT side-by-side + equation reproduction for MiniCPM5
step minicpm5_compare    env HF_HUB_OFFLINE=1 CASE_STUDY_MODEL=minicpm5 "$US" scripts/14_compare_cpt_vs_sft.py
step minicpm5_equations  env HF_HUB_OFFLINE=1 CASE_STUDY_MODEL=minicpm5 "$US" scripts/15_equation_probe.py

# 3) LoRA-rank sweep (rsLoRA) — quality vs cost across ranks — for each QLoRA model
for M in minicpm5 smollm3; do
  step ${M}_lora_sweep   env HF_HUB_OFFLINE=1 CASE_STUDY_MODEL="$M" "$US" scripts/16_lora_rank_sweep.py
done

echo; echo "[$(ts)] OVERNIGHT DONE."
echo "Logs:     case_study/$L/"
echo "Results:  case_study/outputs/<model>/{lora_sweep,compare,equation_probe}_${CASE_STUDY_MODE}.json"
echo "Tell Claude 'overnight done' and it will read + summarize everything."
