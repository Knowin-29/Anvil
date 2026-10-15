#!/usr/bin/env bash
# Short, pinned entrypoint for every currently legal claim-bearing 4xL20 stage.
set -Eeuo pipefail

W="${GSA_ROOT:-/path/to/Anvil}"
W=$(cd "$W" && pwd -P)
cd "$W"

check_sha256() {
  local expected="$1" path="$2" actual
  [[ -f "$path" ]] || { echo "FATAL: missing pinned file: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "FATAL: stale or modified DLC code: $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual  : $actual" >&2
    exit 2
  }
}

check_sha256 b33331c3ee376135b1e718f804c4a4be2b248218280c4a23b09bf067cd3e542a \
  eval/general_retention_formal_v3.py
check_sha256 f543fd85d3fb558a4a4767b751909562391bba5879919404c0a65c0ef8588223 \
  configs/eval/general_retention_formal_v3.yaml
check_sha256 fe2fce610d24d444a4f8218109ae45b51005d95c06995a21a3609907cdc12171 \
  scripts/run_general_retention_dp.sh
check_sha256 7dde5b8b26fe728495d9ea4147db8dc8e5f83c80618ea21cead912d134a667ec \
  tools/check_scannet_rgb_depth_v52_unique_referent_formal.py
check_sha256 c6cd8153ee79fbaa4fe64181293cfafa2eedaf35d005ba28c6c986dd2e097989 \
  eval/scannet_rgb_depth_v52_eval.py
check_sha256 c602474eb6e8d2c850cd290c9d7003da4d14b78a8462327a051378f364918b25 \
  eval/scannet_rgb_depth_v52_dlc.py
check_sha256 107c050fffb98bb6911234000dde8c91bb10aed0f45749bd009539e8f9267048 \
  eval/scannet_rgb_depth_v51_dlc.py
check_sha256 0dcb84ac6873a6368f9bab58a5ed0cc882500536fadcaecf04f206134becc5c4 \
  eval/scannet_rgb_depth_v52_interface.py
check_sha256 0b25b84eeda1e7519aedb5a87263df84ffb4525fae7718e97e410835ba2186ac \
  eval/gsa_eval/runner.py
check_sha256 27c819c538d4cc5032baaefb4b95710a2bcea1646c56a831b0875d99efc14f3c \
  experiments/scannet_rgb_depth_v52_gs_intervention_contract.json
check_sha256 93d5d6f1d512cd6484542f7cf85cfb923e23778184649072e4532190393c3c82 \
  scripts/run_scannet_rgb_depth_v52_eval_l20_4gpu.sh
check_sha256 776dce30d7d461469dfb6f950271afb93d331d765095236347e4d760da0438f9 \
  scripts/run_gsa_4xl20_dlc.sh
COMPILER_SHA256=f95fdfc03e9af808b2485fc0f50e9505e2b6f5b97ee1f7019b360a9f0585bfdc
COMPILER_REL=eval/compile_iclr_claim_evidence.py
check_sha256 "$COMPILER_SHA256" "$COMPILER_REL"
RENDERER_SHA256=e7ea8548c7fc845c891e456641d4c78982f59705a754b6887518d06d42ff136f
RENDERER_REL=eval/render_iclr_claim_artifacts.py
check_sha256 "$RENDERER_SHA256" "$RENDERER_REL"
VISUAL_CONTRACT_SHA256=88ce07538c5c0f9df7b86910b1c120680d865350edb22adf2af0e5d66066a4b7
VISUAL_CONTRACT_REL=experiments/claim_evidence_visual_contract.yaml
check_sha256 "$VISUAL_CONTRACT_SHA256" "$VISUAL_CONTRACT_REL"

RUN_ID="${RUN_ID:-iclr_claims_4xl20_$(date -u +%Y%m%d_%H%M%S)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]] || {
  echo "FATAL: unsafe RUN_ID: $RUN_ID" >&2
  exit 2
}
PREFLIGHT_VALUE="${PREFLIGHT_ONLY:-0}"
RESUME_VALUE="${RESUME_RUN:-0}"
CANONICAL_RUN_ROOT="$W/artifacts/dlc_runs/$RUN_ID"
if [[ -n "${RUN_ROOT:-}" && "$RUN_ROOT" != "$CANONICAL_RUN_ROOT" ]]; then
  echo "FATAL: RUN_ROOT must be canonical for post-DLC evidence compilation" >&2
  echo "  expected: $CANONICAL_RUN_ROOT" >&2
  echo "  actual  : $RUN_ROOT" >&2
  exit 2
fi
echo "GENERAL_RETENTION_V3_CODE_READY run_id=$RUN_ID"
echo "workers: 4 concurrent single-GPU processes; AI2D=16 ScienceQA=16 TextVQA=1"
echo "visual budget: IMAGE_MAX_TOKEN_NUM=4096; singleton OOM fails without skipping"
if [[ "${PIN_CHECK_ONLY:-0}" == 1 ]]; then
  echo "GENERAL_RETENTION_V3_PIN_CHECK_COMPLETE: no GPU query or experiment was started"
  exit 0
fi

mkdir -p "$W/artifacts/dlc_runs"
POST_LOCK="$W/artifacts/dlc_runs/.${RUN_ID}.post_dlc_evidence.lock"
exec 8>"$POST_LOCK"
flock -n 8 || {
  echo "FATAL: another wrapper invocation owns RUN_ID=$RUN_ID" >&2
  exit 3
}

PYTHON_BIN="${PYTHON_BIN:-/path/to/environment/env/gsa_train/bin/python}"
ACTIVE_PID=""
cleanup_active() {
  local rc=$? killer=""
  trap - EXIT INT TERM
  if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
    "$PYTHON_BIN" -c \
      'import os,signal,sys,time; time.sleep(40); os.killpg(int(sys.argv[1]), signal.SIGKILL)' \
      "$ACTIVE_PID" </dev/null >/dev/null 2>&1 &
    killer=$!
    wait "$ACTIVE_PID" 2>/dev/null || true
    kill -TERM "$killer" 2>/dev/null || true
    wait "$killer" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup_active EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PYTHON_BIN" "$RENDERER_REL" --help >/dev/null 2>&1 &
ACTIVE_PID=$!
set +e
wait "$ACTIVE_PID"
RENDERER_PREFLIGHT_RC=$?
set -e
ACTIVE_PID=""
if [[ "$RENDERER_PREFLIGHT_RC" != 0 ]]; then
  echo "FATAL: renderer dependency/CLI preflight failed with rc=$RENDERER_PREFLIGHT_RC" >&2
  echo "  python: $PYTHON_BIN" >&2
  echo "  renderer: $W/$RENDERER_REL" >&2
  exit 2
fi
echo "CLAIM_RENDERER_PREFLIGHT_READY: target Python imported renderer dependencies; no evidence or GPU work started"

setsid env \
  RUN_ID="$RUN_ID" \
  RUN_ROOT="$CANONICAL_RUN_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  PREFLIGHT_ONLY="$PREFLIGHT_VALUE" \
  RESUME_RUN="$RESUME_VALUE" \
  RUN_GENERAL_RETENTION=1 \
  RUN_V52_PAIRED=auto \
  bash "$W/scripts/run_gsa_4xl20_dlc.sh" &
ACTIVE_PID=$!
set +e
wait "$ACTIVE_PID"
ORCHESTRATOR_RC=$?
set -e
ACTIVE_PID=""
if [[ "$ORCHESTRATOR_RC" != 0 ]]; then
  echo "DLC_EXPERIMENT_FAILED: rc=$ORCHESTRATOR_RC evidence_compiler=not_started" >&2
  echo "run root: $CANONICAL_RUN_ROOT" >&2
  exit "$ORCHESTRATOR_RC"
fi
if [[ "$PREFLIGHT_VALUE" == 1 ]]; then
  echo "CLAIM_EVIDENCE_SKIPPED: PREFLIGHT_ONLY=1; no compiler, renderer, or evidence output"
  exit 0
fi

echo "DLC_EXPERIMENT_SUCCEEDED: run_root=$CANONICAL_RUN_ROOT"
COMPILER_ACTUAL=""
if [[ ! -f "$COMPILER_REL" ]] || ! COMPILER_ACTUAL=$(sha256sum "$COMPILER_REL" | awk '{print $1}'); then
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: compiler disappeared after the run" >&2
  echo "  compiler: $W/$COMPILER_REL" >&2
  echo "  run root: $CANONICAL_RUN_ROOT" >&2
  exit 7
fi
if [[ "$COMPILER_ACTUAL" != "$COMPILER_SHA256" ]]; then
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: compiler SHA changed during the run" >&2
  echo "  expected: $COMPILER_SHA256" >&2
  echo "  actual  : $COMPILER_ACTUAL" >&2
  echo "  run root: $CANONICAL_RUN_ROOT" >&2
  exit 7
fi

MANIFEST="$CANONICAL_RUN_ROOT/run_manifest.json"
set +e
RESUME_COUNT=$("$PYTHON_BIN" - "$MANIFEST" "$RUN_ID" <<'PY'
import json, pathlib, sys
path, expected_run_id = pathlib.Path(sys.argv[1]), sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
count = value.get("resume_count")
if value.get("run_id") != expected_run_id or not isinstance(count, int) or count < 0:
    raise SystemExit("invalid run manifest identity/resume_count")
print(count)
PY
)
MANIFEST_RC=$?
set -e
if [[ "$MANIFEST_RC" != 0 ]]; then
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: cannot bind evidence to run manifest" >&2
  echo "  manifest: $MANIFEST" >&2
  echo "  run root: $CANONICAL_RUN_ROOT" >&2
  exit 7
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EVIDENCE_PARENT="$W/artifacts/claim_evidence/$RUN_ID/resume_$RESUME_COUNT"
EVIDENCE_DIR="$EVIDENCE_PARENT/${STAMP}_pid_$$"
mkdir -p "$EVIDENCE_PARENT"
if ! mkdir "$EVIDENCE_DIR"; then
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: fresh evidence directory collision" >&2
  echo "  path: $EVIDENCE_DIR" >&2
  exit 7
fi
EVIDENCE_JSON="$EVIDENCE_DIR/claim_evidence.json"
EVIDENCE_MD="$EVIDENCE_DIR/claim_evidence.md"
EVIDENCE_LOG="$EVIDENCE_DIR/compile.log"
echo "CLAIM_EVIDENCE_COMPILATION_START: log=$EVIDENCE_LOG"
setsid "$PYTHON_BIN" "$COMPILER_REL" \
  --run-root "$CANONICAL_RUN_ROOT" \
  --out "$EVIDENCE_JSON" \
  --markdown "$EVIDENCE_MD" \
  >"$EVIDENCE_LOG" 2>&1 &
ACTIVE_PID=$!
set +e
wait "$ACTIVE_PID"
COMPILER_RC=$?
set -e
ACTIVE_PID=""
if [[ "$COMPILER_RC" != 0 ]]; then
  tail -n 100 "$EVIDENCE_LOG" >&2 || true
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: compiler_rc=$COMPILER_RC" >&2
  echo "  run root: $CANONICAL_RUN_ROOT" >&2
  echo "  log: $EVIDENCE_LOG" >&2
  exit "$COMPILER_RC"
fi
if [[ ! -s "$EVIDENCE_JSON" || ! -s "$EVIDENCE_MD" ]]; then
  echo "EXPERIMENT_SUCCEEDED_EVIDENCE_COMPILATION_FAILED: compiler omitted JSON or Markdown" >&2
  echo "  log: $EVIDENCE_LOG" >&2
  exit 7
fi

for PIN_SPEC in \
  "$COMPILER_SHA256:$COMPILER_REL:compiler" \
  "$RENDERER_SHA256:$RENDERER_REL:renderer" \
  "$VISUAL_CONTRACT_SHA256:$VISUAL_CONTRACT_REL:visual contract"; do
  IFS=: read -r EXPECTED_SHA PINNED_REL PINNED_LABEL <<<"$PIN_SPEC"
  PINNED_ACTUAL=""
  if [[ ! -f "$PINNED_REL" ]] || ! PINNED_ACTUAL=$(sha256sum "$PINNED_REL" | awk '{print $1}'); then
    echo "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED: $PINNED_LABEL disappeared before render" >&2
    echo "  file: $W/$PINNED_REL" >&2
    echo "  evidence: $EVIDENCE_JSON" >&2
    exit 7
  fi
  if [[ "$PINNED_ACTUAL" != "$EXPECTED_SHA" ]]; then
    echo "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED: $PINNED_LABEL SHA changed before render" >&2
    echo "  expected: $EXPECTED_SHA" >&2
    echo "  actual  : $PINNED_ACTUAL" >&2
    echo "  evidence: $EVIDENCE_JSON" >&2
    exit 7
  fi
done

RENDER_DIR="$EVIDENCE_DIR/claim_artifacts"
RENDER_LOG="$EVIDENCE_DIR/render.log"
RENDER_MANIFEST="$RENDER_DIR/render_manifest.json"
echo "CLAIM_ARTIFACT_RENDER_START: log=$RENDER_LOG"
setsid "$PYTHON_BIN" "$RENDERER_REL" \
  --evidence "$EVIDENCE_JSON" \
  --output-dir "$RENDER_DIR" \
  >"$RENDER_LOG" 2>&1 &
ACTIVE_PID=$!
set +e
wait "$ACTIVE_PID"
RENDER_RC=$?
set -e
ACTIVE_PID=""
if [[ "$RENDER_RC" != 0 ]]; then
  tail -n 100 "$RENDER_LOG" >&2 || true
  echo "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED: renderer_rc=$RENDER_RC" >&2
  echo "  evidence: $EVIDENCE_JSON" >&2
  echo "  log: $RENDER_LOG" >&2
  exit "$RENDER_RC"
fi
if [[ ! -s "$RENDER_MANIFEST" ]]; then
  echo "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED: renderer omitted render_manifest.json" >&2
  echo "  render directory: $RENDER_DIR" >&2
  echo "  log: $RENDER_LOG" >&2
  exit 7
fi

EVIDENCE_READY="$EVIDENCE_DIR/READY.json"
set +e
"$PYTHON_BIN" - \
  "$EVIDENCE_READY" "$EVIDENCE_JSON" "$EVIDENCE_MD" "$RENDER_MANIFEST" "$RENDER_DIR" \
  "$RUN_ID" "$RESUME_COUNT" "$CANONICAL_RUN_ROOT" \
  "$COMPILER_SHA256" "$W/$COMPILER_REL" \
  "$RENDERER_SHA256" "$W/$RENDERER_REL" \
  "$VISUAL_CONTRACT_SHA256" "$W/$VISUAL_CONTRACT_REL" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile

(
    ready,
    json_path,
    markdown_path,
    render_manifest_path,
    render_dir,
) = map(pathlib.Path, sys.argv[1:6])
(
    run_id,
    resume_count,
    run_root,
    compiler_sha256,
    compiler_path,
    renderer_sha256,
    renderer_path,
    contract_sha256,
    contract_path,
) = sys.argv[6:]

def binding(path):
    if path.is_symlink():
        raise ValueError(f"symlink is not admissible for handoff: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing or non-regular handoff file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def canonical_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

json_binding = binding(json_path)
markdown_binding = binding(markdown_path)
compiler_binding = binding(pathlib.Path(compiler_path))
renderer_binding = binding(pathlib.Path(renderer_path))
contract_binding = binding(pathlib.Path(contract_path))
if compiler_binding["sha256"] != compiler_sha256:
    raise ValueError("live compiler no longer matches its frozen SHA")
if renderer_binding["sha256"] != renderer_sha256:
    raise ValueError("live renderer no longer matches its frozen SHA")
if contract_binding["sha256"] != contract_sha256:
    raise ValueError("live visual contract no longer matches its frozen SHA")

evidence_payload = json.loads(json_path.read_text(encoding="utf-8"))
evidence_identity = evidence_payload.get("identity_sha256")
unsigned_evidence = dict(evidence_payload)
unsigned_evidence.pop("identity_sha256", None)
if (
    not isinstance(evidence_identity, str)
    or canonical_hash(unsigned_evidence) != evidence_identity
):
    raise ValueError("claim evidence identity mismatch")

render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
render_identity = render_manifest.get("identity_sha256")
unsigned_manifest = dict(render_manifest)
unsigned_manifest.pop("identity_sha256", None)
if not isinstance(render_identity, str) or canonical_hash(unsigned_manifest) != render_identity:
    raise ValueError("render manifest identity mismatch")
if (
    render_manifest.get("schema_version") != 1
    or render_manifest.get("protocol") != "iclr-claim-artifacts-v3"
    or render_manifest.get("status") != "CLAIM_ARTIFACTS_RENDERED"
):
    raise ValueError("render manifest protocol/status mismatch")
expected_evidence_binding = {**json_binding, "identity_sha256": evidence_identity}
if render_manifest.get("evidence") != expected_evidence_binding:
    raise ValueError("render manifest evidence binding mismatch")
if render_manifest.get("compiler") != compiler_binding:
    raise ValueError("render manifest compiler binding mismatch")
if render_manifest.get("renderer") != renderer_binding:
    raise ValueError("render manifest renderer binding mismatch")
if render_manifest.get("visual_contract") != contract_binding:
    raise ValueError("render manifest visual-contract binding mismatch")
v52_availability = evidence_payload.get("scannet_v52_paired_gs", {}).get("availability")
if v52_availability not in {"available", "unavailable"}:
    raise ValueError("claim evidence has invalid v5.2 availability")
if render_manifest.get("v52_availability") != v52_availability:
    raise ValueError("render manifest v5.2 availability mismatch")
if render_manifest.get("publication_policy") != {
    "paper_was_not_modified": True,
    "missing_values_filled": False,
    "unavailable_v52_table_created": False,
    "editable_vector_output_created": True,
}:
    raise ValueError("render manifest publication policy mismatch")

required_outputs = {
    "general_retention_table": "tab_general_retention.tex",
    "claim_figure_svg": "fig_claim_evidence.svg",
    "claim_figure_pdf": "fig_claim_evidence.pdf",
    "claim_figure_png": "fig_claim_evidence.png",
}
if v52_availability == "available":
    required_outputs["scannet_v52_paired_gs_table"] = "tab_scannet_v52_paired_gs.tex"
outputs = render_manifest.get("outputs")
if not isinstance(outputs, dict) or set(outputs) != set(required_outputs):
    raise ValueError("render manifest output set mismatch")
render_root = render_dir.resolve()
verified_render_outputs = {}
for key, relative_name in required_outputs.items():
    observed = outputs[key]
    if not isinstance(observed, dict) or observed.get("path") != relative_name:
        raise ValueError(f"render manifest path mismatch for {key}")
    output_path = (render_root / relative_name).resolve()
    try:
        output_path.relative_to(render_root)
    except ValueError as exc:
        raise ValueError(f"render output escapes its bundle: {key}") from exc
    absolute_binding = binding(output_path)
    expected_relative_binding = {**absolute_binding, "path": relative_name}
    if observed != expected_relative_binding:
        raise ValueError(f"render output binding mismatch for {key}")
    verified_render_outputs[key] = observed

render_manifest_binding = binding(render_manifest_path)

payload = {
    "schema_version": 1,
    "protocol": "iclr-claim-handoff-v2",
    "status": "ICLR_CLAIM_HANDOFF_READY",
    "run_id": run_id,
    "resume_count": int(resume_count),
    "run_root": str(pathlib.Path(run_root).resolve()),
    "compiler_sha256": compiler_sha256,
    "renderer_sha256": renderer_sha256,
    "visual_contract_sha256": contract_sha256,
    "render_identity_sha256": render_identity,
    "outputs": {
        "evidence_json": json_binding,
        "evidence_markdown": markdown_binding,
        "render_manifest": render_manifest_binding,
    },
    "claim_artifacts_root": str(render_root),
    "claim_artifacts": verified_render_outputs,
}
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{ready.name}.", dir=ready.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, ready)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
PY
READY_RC=$?
set -e
if [[ "$READY_RC" != 0 || ! -s "$EVIDENCE_READY" ]]; then
  echo "EXPERIMENT_AND_EVIDENCE_SUCCEEDED_CLAIM_RENDER_FAILED: render verification or READY commit failed" >&2
  echo "  partial directory: $EVIDENCE_DIR" >&2
  echo "  compile log: $EVIDENCE_LOG" >&2
  echo "  render log: $RENDER_LOG" >&2
  exit 7
fi
cat "$EVIDENCE_LOG"
cat "$RENDER_LOG"
echo "ICLR_CLAIM_HANDOFF_READY: json=$EVIDENCE_JSON"
echo "ICLR_CLAIM_HANDOFF_READY: markdown=$EVIDENCE_MD"
echo "ICLR_CLAIM_HANDOFF_READY: artifacts=$RENDER_DIR"
echo "ICLR_CLAIM_HANDOFF_READY: render_manifest=$RENDER_MANIFEST"
echo "ICLR_CLAIM_HANDOFF_READY: compile_log=$EVIDENCE_LOG"
echo "ICLR_CLAIM_HANDOFF_READY: render_log=$RENDER_LOG"
echo "ICLR_CLAIM_HANDOFF_READY: receipt=$EVIDENCE_READY"
