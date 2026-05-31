#!/usr/bin/env bash
# run.sh — ONE command to run the whole case study end-to-end, with clear logs + a results digest.
#
#   bash case_study/run.sh                      # 135M pipeline (bf16)
#   bash case_study/run.sh smollm3              # SmolLM3-3B pipeline (QLoRA 4-bit)
#   bash case_study/run.sh smollm3 06 07        # only sections 06 and 07
#   MODE=trial bash case_study/run.sh smollm3   # fast smoke test instead of the full run
#
# Per-section logs:  case_study/logs/<model>/NN_*.log
# Results digest:    printed at the end + written to case_study/logs/<model>/SUMMARY.md
#
# Envs are handled for you: 135M=bf16 in .venv-rl; SmolLM3=QLoRA 4-bit in .venv (Unsloth+bitsandbytes).
set -uo pipefail
cd "$(dirname "$0")"

# ── config ────────────────────────────────────────────────────────────────────
export CASE_STUDY_MODE="${MODE:-full}"          # MODE=trial for a fast smoke test
HF="../.venv-rl/bin/python"                     # trl 1.x, bf16            (135M)
US="../.venv/bin/python"                        # unsloth+trl0.24+bnb     (SmolLM3 QLoRA)
ts() { date +%H:%M:%S; }
say() { printf '\033[1;36m%s\033[0m\n' "$*"; }

MODEL="${1:-smollm2-135m}"
case "$MODEL" in smollm2-135m|smollm3) shift || true ;; *) MODEL="smollm2-135m" ;; esac
export CASE_STUDY_MODEL="$MODEL"
SECTIONS="$*"; SECSET=$([ -n "$SECTIONS" ] && echo 1 || echo 0)
LOGDIR="logs/$MODEL"; mkdir -p "$LOGDIR"
PY=$([ "$MODEL" = "smollm3" ] && echo "$US" || echo "$HF")

# ── preflight ──────────────────────────────────────────────────────────────────
say "════════════════════════════════════════════════════════════════"
say " CASE STUDY  |  model=$MODEL  mode=$CASE_STUDY_MODE  started $(ts)"
say "════════════════════════════════════════════════════════════════"
[ -x "$HF" ] || { echo "missing .venv-rl python ($HF)"; exit 1; }
[ "$MODEL" = "smollm3" ] && { [ -x "$US" ] || { echo "missing .venv python ($US) for QLoRA"; exit 1; }; }
"$PY" - <<'PY'
import torch
print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()}"
      f" {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
PY
command -v ollama >/dev/null && echo "  ollama: $(command -v ollama)" || echo "  ollama: not found (Part B deploy/edge/tool sections will skip)"
echo

# ── run one section ──────────────────────────────────────────────────────────
run() {  # run <id> <logname> <python> <script> [args...]
  local id="$1" log="$LOGDIR/$2"; shift 2
  if [ "$SECSET" = "1" ] && ! grep -qw "$id" <<<"$SECTIONS"; then return; fi
  printf '\033[1;33m[%s] ▶ §%s  %s\033[0m\n' "$(ts)" "$id" "$(basename "$2" .log 2>/dev/null || echo "$2")"
  local t0=$SECONDS
  "$@" >"$log" 2>&1; local rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then printf '\033[1;32m[%s] ✓ §%s done in %ds\033[0m\n' "$(ts)" "$id" "$dt"
  else printf '\033[1;31m[%s] ✗ §%s FAILED (exit %d) in %ds — see %s\033[0m\n' "$(ts)" "$id" "$rc" "$dt" "$log"; fi
  tail -n 2 "$log" | sed 's/^/      /'
}

# ── the pipeline ────────────────────────────────────────────────────────────────
if [ "$MODEL" = "smollm3" ]; then
  run 01 01_corpus.log     $US 01_build_corpus.py
  run 02 02_data.log       $US 02_data_availability.py
  run 03 03_cpt.log        $US 03_cpt.py --backend unsloth --force
  run 04 04_forgetting.log $US 04_cpt_base_vs_instruct.py --force
  run 05 05_sft.log        $US 05_sft.py --force
  run 06 06_sweep.log      $US 06_base_vs_instruct_sweep.py --force
  run 07 07_eval.log       $US 07_eval.py --force
  run 09 09_merge.log      $US 09_merge_and_gguf.py
  run 10 10_ollama.log     $US 10_ollama_deploy.py
  run 11 11_edge.log       $US 11_edge_benchmark.py
  run 12 12_harness.log    $US 12_harness.py
  run 13 13_toolcall.log   $US 13_smollm3_toolcall.py
else
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
  run 13 13_toolcall.log      $HF 13_smollm3_toolcall.py
fi

# ── results digest ───────────────────────────────────────────────────────────
S="$LOGDIR/SUMMARY.md"
{
  echo "# Case study results — $MODEL ($CASE_STUDY_MODE)   $(date)"
  dig() { grep -rhE "$1" "$LOGDIR"/*.log 2>/dev/null | sed 's/^ *//'; }
  echo; echo "## §1 Corpus";            dig "Corpus: .* unique pages"
  echo; echo "## §2 Data availability"; dig "Asymmetry:|pairs \| ~"
  echo; echo "## §3 CPT (perplexity)";  dig "RESULT.*perplexity|perplexity (BEFORE|AFTER)"
  echo; echo "## §4 Forgetting";        dig "after CPT:|domain ppl Δ"
  echo; echo "## §5 SFT (masking)";     dig "Unmasked .*%|unmasked-token fraction"
  echo; echo "## §6 Base-vs-instruct sweep"; dig "init=.*N=|winner|completion perplexity \(lower"
  echo; echo "## §7 Eval scorecard";    dig "\[instruct|domain_ppl|completion_ppl|keyword_recall|metric "
  echo; echo "## §8 Unsloth-vs-HF";     dig "wall time|peak VRAM|Unsloth speedup"
  echo; echo "## §11 Edge";             dig "FOOTPRINT|SPEED:"
  echo; echo "## §13 Tool-calling";     dig "tool_call|NO tool_call|TAKEAWAY|tools API error"
} > "$S"

say "════════════════════════════════════════════════════════════════"
say " DONE ($MODEL, $CASE_STUDY_MODE) at $(ts).  Digest:"
say "════════════════════════════════════════════════════════════════"
cat "$S"
echo
say "Full per-section logs: case_study/$LOGDIR/   |   digest: case_study/$S"
say "Tell Claude: 'full run done' and it will read the logs + report."
