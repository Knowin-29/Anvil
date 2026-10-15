#!/usr/bin/env bash
# _find_sft_adapter.sh — print the newest usable E1 LoRA adapter directory.
#
# Kept as one helper because three call sites (GRPO, smoke, eval) previously each
# carried their own glob, and the GRPO copy omitted the `_l20` variant: run_all_dlc.sh
# selects run_sft_9b_lora_l20_dlc.sh on L20/L40/A10/4090 nodes, which writes to
# output/sft_9b_lora_l20/v*/checkpoint-*. E1 completed there while E3 reported
# "no SFT adapter found" and exited in 2 seconds.
#
# Ordering note: checkpoint dirs must be ranked by their NUMERIC step. A plain
# `sort -t- -k2 -n` splits on the first '-' and compares the version-dir field
# ("20260816" from v0-20260816-044214), so it returns an arbitrary — in practice the
# earliest — checkpoint. The step is extracted explicitly before sorting.
set -uo pipefail

W="${W:-${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

ADAPTER=$(ls -d "$W"/output/sft_9b_lora/checkpoint-*         \
                "$W"/output/sft_9b_lora/v*/checkpoint-*      \
                "$W"/output/sft_9b_lora_l20/checkpoint-*     \
                "$W"/output/sft_9b_lora_l20/v*/checkpoint-*  \
                "$W"/output/sft_9b_gsabench/checkpoint-*     \
                "$W"/output/sft_9b_gsabench/v*/checkpoint-*  \
             2>/dev/null \
         | sed 's|.*/checkpoint-\([0-9]\{1,\}\)$|\1 &|' \
         | sort -n -k1 \
         | awk 'NF==2' | tail -1 | cut -d' ' -f2)

if [[ -z "$ADAPTER" ]]; then
  echo "FATAL: no SFT adapter under $W/output/{sft_9b_lora,sft_9b_lora_l20,sft_9b_gsabench}" >&2
  echo "       run E1 first: bash $W/scripts/run_sft_9b_lora_l20_dlc.sh" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "FATAL: $ADAPTER has no adapter_model.safetensors" >&2
  exit 1
fi
echo "$ADAPTER"
