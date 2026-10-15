#!/usr/bin/env bash
# Download additional OPEN-LICENSE embodied datasets to expand scene coverage
# (ego / exploration / household / manipulation) so Visual Genome can be down-weighted.
#
# Disk is ~99% full (~1.8T free), so each dataset is size-capped and we prefer
# image/annotation subsets over full raw releases. Everything lands under
# /path/to/datasets/embodied_new/<name>/.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env
export ALL_PROXY="${ALL_PROXY:-http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DEST_ROOT=/path/to/datasets/embodied_new
mkdir -p "$DEST_ROOT"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

dl() {  # dl <repo_id> <repo_type> <local_name> [allow_pattern]
  local repo="$1" rtype="$2" name="$3" pat="${4:-}"
  local dest="$DEST_ROOT/$name"
  if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
    log "SKIP $name (already present: $(du -sh "$dest" 2>/dev/null | cut -f1))"; return 0
  fi
  mkdir -p "$dest"
  log "DOWNLOAD $repo -> $dest ${pat:+(pattern: $pat)}"
  if [[ -n "$pat" ]]; then
    hf download "$repo" --repo-type "$rtype" --local-dir "$dest" \
      --include "$pat" >/dev/null 2>>"logs/download_${name}.err" \
      && log "OK $name ($(du -sh "$dest" 2>/dev/null | cut -f1))" \
      || log "FAIL $name (see logs/download_${name}.err)"
  else
    hf download "$repo" --repo-type "$rtype" --local-dir "$dest" \
      >/dev/null 2>>"logs/download_${name}.err" \
      && log "OK $name ($(du -sh "$dest" 2>/dev/null | cut -f1))" \
      || log "FAIL $name (see logs/download_${name}.err)"
  fi
  df -h /path/to/workspace/foundation | tail -1
}

log "=== embodied dataset download start ==="

# --- Spatial / embodied QA + grounding (annotation-light, high value) ---
dl "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1" dataset groot_teleop "*.json*"
dl "jxu124/OpenX-Embodiment"                    dataset openx_meta    "*.json*"

# --- Ego-centric (ego scene bucket) ---
dl "lmms-lab/EgoThink"                          dataset egothink
dl "Salesforce/EgoExoLearn-subset"              dataset egoexo         "*.json*"

# --- Exploration / navigation ---
dl "ai-habitat/ReplicaCAD_dataset"              dataset replicacad     "*.json*"

# --- Manipulation / affordance ---
dl "HuggingFaceM4/PACO-LVIS"                    dataset paco_lvis      "*.json*"
dl "allenai/pixmo-points"                       dataset pixmo_pts_meta "*.json*"

log "=== download complete ==="
du -sh "$DEST_ROOT"/* 2>/dev/null
