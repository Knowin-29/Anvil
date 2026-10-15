import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import render_general_retention_paper_values as renderer  # noqa: E402


def _paired(tmp_path: Path, *, status: str = "PAIRED_GENERAL_RETENTION_ADMITTED") -> Path:
    base = tmp_path / "base.json"
    e1 = tmp_path / "e1.json"
    base.write_text("{}\n", encoding="utf-8")
    e1.write_text("{}\n", encoding="utf-8")
    payload = {
        "protocol": renderer.formal.PROTOCOL,
        "status": status,
        "paired_health_admitted": status == "PAIRED_GENERAL_RETENTION_ADMITTED",
        "base_result": {"path": str(base), "bytes": 3, "sha256": "x"},
        "e1_result": {"path": str(e1), "bytes": 3, "sha256": "y"},
        "base_general_avg_percent": 80.0,
        "e1_general_avg_percent": 74.5,
        "raw_general_avg_delta_points": -5.5,
    }
    payload["identity_sha256"] = renderer._canonical_hash(payload)
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_render_admitted_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paired = _paired(tmp_path)
    monkeypatch.setattr(
        renderer.formal,
        "verify_file_binding",
        lambda binding, deep: Path(binding["path"]),
    )

    def fake_result(path: Path) -> dict:
        if path.name == "base.json":
            return {"status": "FORMAL_SCORE_ADMITTED", "model_role": "base", "general_avg_percent": 80.0}
        return {"status": "FORMAL_SCORE_ADMITTED", "model_role": "e1", "general_avg_percent": 74.5}

    monkeypatch.setattr(renderer.formal, "_verify_result", fake_result)
    out = tmp_path / "values.tex"
    result = renderer.render(paired, out)
    text = out.read_text(encoding="utf-8")
    assert result["status"] == "GENERAL_RETENTION_PAPER_VALUES_RENDERED"
    assert r"\newcommand{\BaseGeneralAvg}{80.00}" in text
    assert r"\newcommand{\EOneGeneralAvg}{74.50}" in text
    assert r"\newcommand{\EOneGeneralDelta}{\negdelta{(-5.50)}}" in text


def test_render_rejects_unadmitted_pair(tmp_path: Path) -> None:
    paired = _paired(tmp_path, status="PAIRED_SCORED_HEALTH_NOT_ADMITTED")
    with pytest.raises(renderer.formal.FormalProtocolError, match="not admitted"):
        renderer.render(paired, tmp_path / "values.tex")
    assert not (tmp_path / "values.tex").exists()
