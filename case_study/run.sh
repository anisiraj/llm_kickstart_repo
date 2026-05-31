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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (helps 3B on 12GB)
HF="../.venv-rl/bin/python"                     # trl 1.x, bf16            (135M)
US="../.venv/bin/python"                        # unsloth+trl0.24+bnb     (SmolLM3 QLoRA)
ts() { date +%H:%M:%S; }
say() { printf '\033[1;36m%s\033[0m\n' "$*"; }

MODEL="${1:-smollm2-135m}"
case "$MODEL" in smollm2-135m|smollm3|minicpm5) shift || true ;; *) MODEL="smollm2-135m" ;; esac
export CASE_STUDY_MODEL="$MODEL"
# 4-bit QLoRA models run in .venv (Unsloth+bitsandbytes); the 135M runs bf16 in .venv-rl.
case "$MODEL" in smollm3|minicpm5) QLORA=1 ;; *) QLORA=0 ;; esac
SECTIONS="$*"; SECSET=$([ -n "$SECTIONS" ] && echo 1 || echo 0)
LOGDIR="logs/$MODEL"; mkdir -p "$LOGDIR"
PY=$([ "$QLORA" = 1 ] && echo "$US" || echo "$HF")

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
# Go OFFLINE for HF if the model family is already cached. transformers 5.5's tokenizer loader
# phones home (model_info() via _patch_mistral_regex) even for cached models — a flaky network
# then crashes a section on a mere tokenizer load. Offline uses the cache and skips that call.
# (Wikipedia §1 and Ollama pulls use their own HTTP, unaffected by HF_HUB_OFFLINE.)
case "$MODEL" in
  smollm3)  _cache="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B" ;;
  minicpm5) _cache="$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM5-1B-sft" ;;
  *)        _cache="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M" ;;
esac
if [ -d "$_cache" ]; then export HF_HUB_OFFLINE=1; echo "  HF cache present -> HF_HUB_OFFLINE=1 (no phone-home)"; else echo "  (first run: HF online for downloads)"; fi
echo

# ── run one section (streams output LIVE to console + log file) ───────────────
run() {  # run <id> <logname> <python> <script> [args...]
  local id="$1" logname="$2" log="$LOGDIR/$2"; shift 2
  if [ "$SECSET" = "1" ] && ! grep -qw "$id" <<<"$SECTIONS"; then return; fi
  echo
  printf '\033[1;33m┌─[%s] §%s START  %s\033[0m\n' "$(ts)" "$id" "$logname"
  printf '\033[0;33m│  cmd: %s   (live log: case_study/%s)\033[0m\n' "$*" "$log"
  printf '\033[0;33m│  gpu: %s\033[0m\n' "$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"
  printf '\033[1;33m└────────────────────────────────────────────────────────────\033[0m\n'
  local t0=$SECONDS
  # Stream output LIVE to console + log. stdbuf line-buffers; PYTHONUNBUFFERED flushes prints/bars.
  export PYTHONUNBUFFERED=1
  stdbuf -oL -eL "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))
  if [ "$rc" -eq 0 ]; then printf '\033[1;32m[%s] ✓ §%s DONE in %ds\033[0m\n' "$(ts)" "$id" "$dt"
  else printf '\033[1;31m[%s] ✗ §%s FAILED (exit %s) in %ds — full log: case_study/%s\033[0m\n' "$(ts)" "$id" "$rc" "$dt" "$log"; fi
}

# ── the pipeline ────────────────────────────────────────────────────────────────
if [ "$QLORA" = 1 ]; then   # SmolLM3 / MiniCPM5 — QLoRA 4-bit, all in .venv
  run 01 01_corpus.log     $US scripts/01_build_corpus.py
  run 02 02_data.log       $US scripts/02_data_availability.py
  run 03 03_cpt.log        $US scripts/03_cpt.py --backend unsloth --force
  run 04 04_forgetting.log $US scripts/04_cpt_base_vs_instruct.py --force
  run 05 05_sft.log        $US scripts/05_sft.py --force
  run 06 06_sweep.log      $US scripts/06_base_vs_instruct_sweep.py --force
  run 07 07_eval.log       $US scripts/07_eval.py --force
  run 09 09_merge.log      $US scripts/09_merge_and_gguf.py
  run 10 10_ollama.log     $US scripts/10_ollama_deploy.py
  run 11 11_edge.log       $US scripts/11_edge_benchmark.py
  run 12 12_harness.log    $US scripts/12_harness.py
  run 13 13_toolcall.log   $US scripts/13_smollm3_toolcall.py
else
  run 01 01_corpus.log        $HF scripts/01_build_corpus.py
  run 02 02_data.log          $HF scripts/02_data_availability.py
  run 03 03_cpt_hf.log        $HF scripts/03_cpt.py --backend hf --force
  run 03 03_cpt_unsloth.log   $US scripts/03_cpt.py --backend unsloth --force
  run 04 04_forgetting.log    $HF scripts/04_cpt_base_vs_instruct.py --force
  run 05 05_sft.log           $HF scripts/05_sft.py --force
  run 06 06_sweep.log         $HF scripts/06_base_vs_instruct_sweep.py --force
  run 07 07_eval.log          $HF scripts/07_eval.py --force
  run 08 08_bench.log         $HF scripts/08_unsloth_vs_hf.py --force
  run 09 09_merge.log         $HF scripts/09_merge_and_gguf.py
  run 10 10_ollama.log        $HF scripts/10_ollama_deploy.py
  run 11 11_edge.log          $HF scripts/11_edge_benchmark.py
  run 12 12_harness.log       $HF scripts/12_harness.py
  run 13 13_toolcall.log      $HF scripts/13_smollm3_toolcall.py
fi

# ── results digest ───────────────────────────────────────────────────────────
S="$LOGDIR/SUMMARY.md"
{
  echo "# Case study results — $MODEL ($CASE_STUDY_MODE)   $(date)"
  dig() { grep -rhaE "$1" "$LOGDIR"/*.log 2>/dev/null | sed 's/^ *//'; }
  echo; echo "## §1 Corpus";            dig "Corpus: .* unique pages"
  echo; echo "## §2 Data availability"; dig "Asymmetry:|pairs \| ~"
  echo; echo "## §3 CPT (perplexity)";  dig "RESULT.*perplexity|held-out perplexity (BEFORE|AFTER)"
  echo; echo "## §4 Forgetting";        dig "before CPT:|after CPT:|^(base|instruct) +\|"
  echo; echo "## §5 SFT (masking)";     dig "[Uu]nmasked.*%"
  echo; echo "## §6 Base-vs-instruct sweep"; dig "init=.*N=|winner|held-out completion perplexity"
  echo; echo "## §7 Eval scorecard";    dig "\[instruct|domain_ppl|completion_ppl|keyword_recall|^[a-z_]+ +\| +[0-9]"
  echo; echo "## §8 Unsloth-vs-HF";     dig "wall time|peak VRAM|speedup"
  echo; echo "## §11 Edge";             dig "FOOTPRINT|SPEED"
  echo; echo "## §13 Tool-calling";     dig "tool_call|NO tool_call|TAKEAWAY|tools API error|tool-use"
} > "$S"

say "════════════════════════════════════════════════════════════════"
say " DONE ($MODEL, $CASE_STUDY_MODE) at $(ts).  Digest:"
say "════════════════════════════════════════════════════════════════"
cat "$S"
echo
say "Full per-section logs: case_study/$LOGDIR/   |   digest: case_study/$S"
say "Tell Claude: 'full run done' and it will read the logs + report."
