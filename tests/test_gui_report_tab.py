import os

from apps.data_flywheel import captioner
from gui.app import DEFAULT_IMAGE, PROJECT_ROOT, list_reports, load_report_html, refresh_reports
from gui.app import _gradio_allowed_paths
from gui.app import test_apis as render_api_probe


def test_list_reports_sorts_by_mtime(tmp_path):
    older = tmp_path / "z_old_20260411_090000.html"
    newer = tmp_path / "a_new_20260411_100000.html"
    older.write_text("<html>older</html>", encoding="utf-8")
    newer.write_text("<html>newer</html>", encoding="utf-8")

    older_ts = 1_700_000_000
    newer_ts = older_ts + 120
    os.utime(older, (older_ts, older_ts))
    os.utime(newer, (newer_ts, newer_ts))

    update = list_reports(str(tmp_path))

    assert update["choices"] == [newer.name, older.name]
    assert update["value"] == newer.name


def test_refresh_reports_returns_latest_html(tmp_path):
    report_path = tmp_path / "desk_20260411_120000.html"
    report_path.write_text("<html><body>latest report</body></html>", encoding="utf-8")

    selector_update, html = refresh_reports(str(tmp_path))

    assert selector_update["value"] == report_path.name
    assert "latest report" in html
    assert "<iframe" in html


def test_load_report_html_wraps_document_in_iframe_srcdoc(tmp_path):
    report_path = tmp_path / "interactive.html"
    report_path.write_text(
        "<html><body><script>window.demo = 1;</script><div>interactive report</div></body></html>",
        encoding="utf-8",
    )

    html = load_report_html(report_path.name, str(tmp_path))

    assert "<iframe" in html
    assert "srcdoc=" in html
    assert "sandbox=" in html
    assert "&lt;script&gt;window.demo = 1;&lt;/script&gt;" in html
    assert "interactive report" in html


def test_load_report_html_handles_missing_file(tmp_path):
    html = load_report_html("missing.html", str(tmp_path))

    assert "报告不存在" in html


def test_gradio_allowed_paths_include_repo_and_debug_dir():
    allowed = _gradio_allowed_paths()

    assert PROJECT_ROOT in allowed
    assert os.path.join(PROJECT_ROOT, "data", "data", "input", "debug") in allowed


def test_default_image_dir_prefers_debug_assets():
    assert DEFAULT_IMAGE == os.path.join(PROJECT_ROOT, "data", "data", "input", "debug")


def test_probe_apis_uses_debug_dir_when_default_images_dir_missing(monkeypatch):
    def fake_probe(image_path):
        return f"ok:{os.path.basename(image_path)}"

    monkeypatch.setattr(captioner, "_API_BACKENDS", [("fake", fake_probe, "Fake API")])

    results = captioner.probe_apis()

    assert results["fake"]["available"] is True
    assert results["fake"]["sample"].startswith("ok:")


def test_test_apis_shows_probe_reason(monkeypatch):
    monkeypatch.setattr(
        captioner,
        "probe_apis",
        lambda test_image=None: {
            "dashscope": {"available": False, "reason": "no test image"},
            "openai": {"available": False, "reason": "401 unauthorized"},
            "anthropic": {"available": True, "sample": "caption ok"},
        },
    )
    monkeypatch.setattr(captioner, "select_best_backend", lambda results=None: "anthropic")

    output = render_api_probe()

    assert "dashscope" in output
    assert "no test image" in output
    assert "401 unauthorized" in output
    assert "anthropic" in output
    assert "caption ok" in output
