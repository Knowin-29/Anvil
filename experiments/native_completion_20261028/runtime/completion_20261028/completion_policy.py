"""Final-model acceptance takes precedence over intermediate SFT retention.

Evidence is byte-bound for provenance. Diagnostic scores are never interpreted
as permission to skip teacher validation or final benchmark evaluation.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "completion_policy.json").read_text())
assert POLICY["sft_retention_admission_required"] is False
assert POLICY["teacher_quality_admission_required"] is True


def student_diagnostics(path, adapter):
    adapter = Path(adapter)
    with (adapter / "adapter_model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(role="diagnostic_only", required_for_mopd_admission=False,
                  adapter=str(adapter), adapter_sha256=digest,
                  score_threshold_applied=False)
    if path is not None:
        path = Path(path)
        raw = path.read_bytes()
        evidence = json.loads(raw)  # Reject unreadable provenance, not low scores.
        if not isinstance(evidence, dict):
            raise ValueError("Diagnostic evidence must be a JSON object")
        result.update(report=str(path), report_sha256=hashlib.sha256(raw).hexdigest(),
                      model_match_asserted=False)
    return result
