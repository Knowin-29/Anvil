#!/usr/bin/env bash
# Launch routed MOPD only from source-clean, semantically audited, qualified adapters.
# This wrapper never searches the historical E1 outputs or calls _find_sft_adapter.sh.
set -euo pipefail

W="${GSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$W"
# shellcheck disable=SC1091
source "$W/.env" 2>/dev/null || true

AUDIT="${SEMANTIC_AUDIT:-$W/artifacts/audits/pilot_semantic_audit_composite.json}"
ELIGIBILITY="${TEACHER_ELIGIBILITY_JSON:-$W/artifacts/audits/teacher_eligibility.json}"
BENCH_DIR="${BENCH_DIR:-$W/trains/gsa_bench_v4_composite_pilot_clean}"
V3_PACK="${V3_AUDIT_PACK:-$W/data/semantic_audit_v3}"
V3_ADJ="${V3_ADJUDICATION:-$W/data/semantic_audit_v3/adjudication.jsonl}"
V4_PACK="${V4_AUDIT_PACK:-$W/data/semantic_audit_v4_depth}"
V4_ADJ="${V4_ADJUDICATION:-$W/data/semantic_audit_v4_depth/adjudication.jsonl}"
TRAIN="$BENCH_DIR/train.jsonl"
DEFAULT_OUT="$W/output/mopd_9b_lora"
[[ "${ANVIL_MTD:-0}" == "1" ]] && DEFAULT_OUT="${ANVIL_ROOT:-$(dirname "$W")/Anvil_ICLR27}/engineering/data/anvil_mtd_9b"
OUT="${OUT:-$DEFAULT_OUT}"

# Machine-verifiable eligibility manifest schema:
# {
#   "verdict": "PASS",
#   "joint_adapter": "/absolute/.../pilot_clean_joint/.../checkpoint-N",
#   "teachers": {
#     "G": {"eligible": true, "adapter": "/absolute/.../teacher_lora_G/.../checkpoint-N"},
#     "S": {"eligible": true, "adapter": "/absolute/.../teacher_lora_S/.../checkpoint-N"},
#     "A": {"eligible": true, "adapter": "/absolute/.../teacher_lora_A/.../checkpoint-N"}
#   }
# }

# Recompute both components and require the exact composite protocol. A historical
# global-v3 PASS cannot substitute for the independently calibrated depth component.
python "$W/scripts/verify_composite_qualification.py" \
  --audit "$AUDIT" --manifest "$BENCH_DIR/manifest.json" \
  --v3-pack "$V3_PACK" --v3-adjudication "$V3_ADJ" \
  --v4-pack "$V4_PACK" --v4-adjudication "$V4_ADJ"

[[ -s "$TRAIN" ]] || { echo "FATAL: clean pilot train split missing: $TRAIN" >&2; exit 2; }

# Select only checkpoint-N directories. A root-level adapter and historical E1 are never
# candidates; the highest numeric N wins even when Swift nests checkpoints under v0-*.
latest_checkpoint_adapter() {
  local root="$1" label="$2" dir base step
  local rows=()
  while IFS= read -r dir; do
    base="${dir##*/}"
    [[ "$base" =~ ^checkpoint-([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"
    [[ -f "$dir/adapter_model.safetensors" ]] || {
      echo "WARN: ignoring $label checkpoint without adapter_model.safetensors: $dir" >&2
      continue
    }
    rows+=("$step $dir")
  done < <(find "$root" -type f -name adapter_config.json -printf '%h\n' 2>/dev/null)
  [[ "${#rows[@]}" -gt 0 ]] || {
    echo "FATAL: no valid numeric $label adapter checkpoint under $root" >&2
    return 2
  }
  printf '%s\n' "${rows[@]}" | sort -k1,1n -k2,2 | tail -1 | cut -d' ' -f2-
}

STUDENT_ADAPTER=$(latest_checkpoint_adapter "$W/output/pilot_clean_joint" "clean-joint")
export GSA_TEACHER_G_ADAPTER
export GSA_TEACHER_S_ADAPTER
export GSA_TEACHER_A_ADAPTER
GSA_TEACHER_G_ADAPTER=$(latest_checkpoint_adapter "$W/output/teacher_lora_G" "G-teacher")
GSA_TEACHER_S_ADAPTER=$(latest_checkpoint_adapter "$W/output/teacher_lora_S" "S-teacher")
GSA_TEACHER_A_ADAPTER=$(latest_checkpoint_adapter "$W/output/teacher_lora_A" "A-teacher")

eligibility_gate() {
  python - "$ELIGIBILITY" "$STUDENT_ADAPTER" \
    "$GSA_TEACHER_G_ADAPTER" "$GSA_TEACHER_S_ADAPTER" "$GSA_TEACHER_A_ADAPTER" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(
        f"teacher eligibility manifest missing: {path}\n"
        "Expected a reviewed manifest with verdict PASS and path-bound entries for G/S/A."
    )
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"teacher eligibility manifest is unreadable: {exc}") from exc
verdict = str(report.get("verdict", "")).upper()
if verdict not in {"PASS", "APPROVED", "ELIGIBLE"}:
    raise SystemExit(f"teacher eligibility verdict is not passing: {verdict!r}")

selected_joint = pathlib.Path(sys.argv[2]).resolve()
declared_joint = pathlib.Path(str(report.get("joint_adapter", ""))).resolve()
if declared_joint != selected_joint:
    raise SystemExit(f"joint adapter mismatch: manifest={declared_joint}, selected={selected_joint}")

teachers = report.get("teachers") or {}
for family, selected_raw in zip(("G", "S", "A"), sys.argv[3:]):
    row = teachers.get(family) or {}
    if row.get("eligible") is not True:
        raise SystemExit(f"teacher {family} is not explicitly eligible")
    declared = pathlib.Path(str(row.get("adapter", ""))).resolve()
    selected = pathlib.Path(selected_raw).resolve()
    if declared != selected:
        raise SystemExit(
            f"teacher {family} adapter mismatch: manifest={declared}, selected={selected}")
print(f"teacher eligibility gate PASS: {path}")
PY
}

if ! eligibility_gate; then
  if [[ "${ALLOW_UNQUALIFIED_TEACHERS:-0}" != "1" ]]; then
    cat >&2 <<EOF
FATAL: no machine-verifiable teacher-eligibility decision covers the selected adapters.
Do not treat completed teacher training as qualification. Produce $ELIGIBILITY after
the matched joint-vs-specialist held-out evaluation, or set
ALLOW_UNQUALIFIED_TEACHERS=1 only for an explicitly non-publication debug run.
EOF
    exit 3
  fi
  cat >&2 <<'EOF'
WARNING: ALLOW_UNQUALIFIED_TEACHERS=1 bypassed the teacher-eligibility gate.
This run is DEBUG ONLY and is not eligible for a paper table or method claim.
EOF
fi

echo "qualified MOPD preflight:"
echo "  student=$STUDENT_ADAPTER"
echo "  teacher_G=$GSA_TEACHER_G_ADAPTER"
echo "  teacher_S=$GSA_TEACHER_S_ADAPTER"
echo "  teacher_A=$GSA_TEACHER_A_ADAPTER"
echo "  train=$TRAIN"
echo "  out=$OUT"

if [[ "${ANVIL_MTD:-0}" == "1" ]]; then
  python "$W/scripts/validate_grpo_dataset.py" "$TRAIN" --anvil --allow-human-audit
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: not launching training"
  exit 0
fi

export STUDENT_ADAPTER TRAIN OUT ALLOW_HUMAN_AUDIT=1
export TEACHER_ELIGIBILITY_JSON="$ELIGIBILITY"
exec bash "$W/scripts/run_mopd_9b_lora_l20.sh"
