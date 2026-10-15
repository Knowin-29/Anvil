#!/bin/bash
# Wait for GSA_volc download to finish, then auto-run convert_gsa_volc.py.
# Stops before rebuild/restart (those need explicit confirmation).
#
# Usage:
#   nohup bash scripts/after_download_convert.sh > logs/after_download_convert.log 2>&1 &
set -u

VOLC_ROOT="/mnt/data/huggingface_downloads/datasets/GSA_volc"
FORGE="/path/to/storage/anvil/GSA_Forge"
EXPECTED_IMAGES=180000   # ~186083; allow small shortfall from missing-on-source
POLL=180

echo "[$(date '+%F %T')] after-download watcher started"

while true; do
    running=0
    pgrep -f "hf download" >/dev/null 2>&1 && running=1
    nimg=$(find "${VOLC_ROOT}/images" -type f 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] download_running=${running} images=${nimg}"
    # Done when downloader exited AND image count looks complete-ish.
    if [ "${running}" -eq 0 ]; then
        echo "[$(date '+%F %T')] downloader exited (images=${nimg}); proceeding to convert"
        break
    fi
    if [ "${nimg}" -ge "${EXPECTED_IMAGES}" ]; then
        echo "[$(date '+%F %T')] image count reached threshold (${nimg}); proceeding to convert"
        break
    fi
    sleep "${POLL}"
done

cd "${FORGE}"
echo "[$(date '+%F %T')] running convert_gsa_volc.py ..."
/usr/local/bin/python3 trains/convert_gsa_volc.py \
    --volc_root "${VOLC_ROOT}" \
    --out_sft trains/gsa_volc_sft.jsonl \
    --out_grpo trains/gsa_volc_grpo.jsonl \
    --report trains/gsa_volc_report.json 2>&1

echo "[$(date '+%F %T')] convert done. Report:"
cat trains/gsa_volc_report.json 2>/dev/null
echo ""
echo "[$(date '+%F %T')] NEXT STEPS (manual): run_data_prepare.sh rebuild + resume training from checkpoint-5000"
echo "[$(date '+%F %T')] after-download watcher exiting"
