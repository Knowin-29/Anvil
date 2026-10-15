from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tools import quick_scannet_rgb_depth_v52_adjudication as module
from tools import check_scannet_rgb_depth_v52_unique_referent_formal as checker


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _binding(pack: Path, path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(pack)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _make_pack(tmp_path: Path) -> tuple[Path, list[dict]]:
    pack = tmp_path / "pack"
    image = tmp_path / "review.jpg"
    image.write_bytes(b"public-rgb")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    labels: list[dict] = []
    selected: list[dict] = []
    for index in range(33):
        sample_id = f"v52-new-{index:02d}"
        nearer = "A" if index % 2 == 0 else "B"
        label = {
            "sample_id": sample_id,
            "image_path": str(image),
            "lineage": "new_v52",
            "review_required_in_v52": True,
            "public_interface": {
                "grounding_targets": [
                    {
                        "slot": "A",
                        "category": "chair",
                        "referring_expression": "the only visible chair",
                    },
                    {
                        "slot": "B",
                        "category": "table",
                        "referring_expression": "the only visible table",
                    },
                ]
            },
            "audit_targets": {
                "grounding": [
                    {
                        "slot": "A",
                        "bbox": [10, 20, 300, 400],
                        "source_instance_id": 91,
                        "source_label": "private-chair",
                    },
                    {
                        "slot": "B",
                        "bbox": [500, 100, 900, 700],
                        "source_instance_id": 37,
                        "source_label": "private-table",
                    },
                ],
                "spatial": {
                    "nearer_slot": nearer,
                    "farther_slot": "B" if nearer == "A" else "A",
                },
            },
            "source_payload": {"private": True},
            "sensor_measurements": [{"private": True}],
            "scene_id": f"private-scene-{index}",
        }
        labels.append(label)
        selected.append(
            {
                "sample_id": sample_id,
                "lineage": "new_v52",
                "image_path": str(image),
                "image_sha256": image_sha,
                "label_sha256": module.v52.canonical_hash(label),
            }
        )
    reused = [
        {"sample_id": f"v52-reused-{index}", "lineage": "reused_v51"}
        for index in range(7)
    ]
    selected.extend(reused)

    provenance: dict[str, dict] = {}
    for side in ("a", "b"):
        label_path = pack / f"annotator_{side}/labels/GS_dependency/unique_pair.jsonl"
        _write_jsonl(label_path, labels)
        provenance[f"reviewer_{side}_labels"] = _binding(pack, label_path)
        (pack / f"annotator_{side}/annotations").mkdir(parents=True)
    all_labels_path = pack / "labels/GS_dependency/unique_pair.jsonl"
    _write_jsonl(
        all_labels_path,
        labels + [{"sample_id": row["sample_id"]} for row in reused],
    )
    provenance["all_labels"] = _binding(pack, all_labels_path)
    manifest = {
        "protocol_version": module.v52.PACK_PROTOCOL,
        "audit_contract": module.v52.AUDIT_CONTRACT,
        "stage": "formal_pending_review",
        "records": 40,
        "selected": selected,
        "provenance": provenance,
    }
    manifest["integrity"] = {
        "manifest_payload_sha256": module.v52.canonical_hash(manifest)
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pack, labels


def _vote(label: dict, key: str, *, correct: bool = True) -> dict:
    return {
        "annotation_id": f"{key}-{label['sample_id']}",
        "case_id": label["sample_id"],
        "annotator_key": key,
        "timestamp": "2026-08-27T00:00:00+00:00",
        "correct": correct,
        "error_code": None if correct else "ambiguous_depth",
        "comment": "" if correct else "The visible depth evidence is ambiguous.",
        "label_snapshot": label,
    }


def _complete_ab(
    pack: Path, labels: list[dict], *, disagreement_index: int | None
) -> None:
    votes_a = [_vote(label, "quick-audit-v52-a") for label in labels]
    votes_b = [_vote(label, "quick-audit-v52-b") for label in labels]
    if disagreement_index is not None:
        votes_b[disagreement_index] = _vote(
            labels[disagreement_index], "quick-audit-v52-b", correct=False
        )
    _write_jsonl(pack / "annotator_a/annotations/annotations.jsonl", votes_a)
    _write_jsonl(pack / "annotator_b/annotations/annotations.jsonl", votes_b)


def test_c_waits_for_exact_complete_a_and_b(tmp_path: Path) -> None:
    pack, labels = _make_pack(tmp_path)
    status = module.adjudication_status(pack)
    assert status == {
        "state": module.WAITING,
        "disagreement_count": None,
        "a_complete": 0,
        "b_complete": 0,
        "expected_per_side": 33,
        "service_ready": False,
        "c_complete": 0,
    }
    _write_jsonl(
        pack / "annotator_a/annotations/annotations.jsonl",
        [_vote(label, "quick-audit-v52-a") for label in labels],
    )
    assert module.adjudication_status(pack)["state"] == module.WAITING
    with pytest.raises(ValueError, match="unavailable"):
        module.V52AdjudicationStore(pack)


def test_zero_disagreements_requires_no_c_service(tmp_path: Path) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=None)
    status = module.adjudication_status(pack)
    assert status["state"] == module.NOT_REQUIRED
    assert status["disagreement_count"] == 0
    assert status["service_ready"] is False
    with pytest.raises(ValueError, match=module.NOT_REQUIRED):
        module.V52AdjudicationStore(pack)


def test_one_disagreement_shows_a_b_evidence_and_saves_exact_snapshot(
    tmp_path: Path,
) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=2)
    store = module.V52AdjudicationStore(pack)
    assert store.image_count == 1
    case = store.case(0)
    form = case["forms"][0]
    assert form["object_class"] == {
        "reviewer_A": {"decision": "Supported", "error_type": None, "comment": ""},
        "reviewer_B": {
            "decision": "Unsupported",
            "error_type": "ambiguous_depth",
            "comment": "The visible depth evidence is ambiguous.",
        },
    }
    assert len(form["gsa_context"]["groundings"]) == 2
    assert [row["rank"] for row in form["payload"]] == [1, 2]
    with pytest.raises(ValueError, match="concrete comment"):
        store.submit(
            0,
            [{"form": "depth", "correct": False, "error_code": "ambiguous_depth"}],
        )
    result = store.submit(0, [{"form": "depth", "correct": True}])
    assert result["stats"]["decisions"] == 1
    rows = module.core._read_jsonl(store.annotations_path)
    assert len(rows) == 1
    assert rows[0]["annotator_key"] == module.ANNOTATOR_KEY
    assert rows[0]["label_snapshot"] == labels[2]
    assert rows[0]["case_id"] == labels[2]["sample_id"]


def test_c_output_drives_checker_formal_pass_and_failed(tmp_path: Path) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=0)
    store = module.V52AdjudicationStore(pack)
    store.submit(0, [{"form": "depth", "correct": True}])
    base = {"status": checker.PACK_VALID, "source_artifacts_rehashed": True}
    passed = checker.compile_reviews(pack, base, store.annotations_path)
    assert passed["status"] == checker.FORMAL_PASS
    assert passed["review"]["required_reviewer_c_decisions"] == 1
    store.submit(
        0,
        [
            {
                "form": "depth",
                "correct": False,
                "error_code": "wrong_relation_or_order",
                "comment": "The displayed near/far order is reversed.",
            }
        ],
    )
    failed = checker.compile_reviews(pack, base, store.annotations_path)
    assert failed["status"] == checker.FORMAL_FAILED


def test_running_c_fails_closed_if_an_a_b_stream_changes(tmp_path: Path) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=0)
    store = module.V52AdjudicationStore(pack)
    with (pack / "annotator_b/annotations/annotations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(_vote(labels[0], "quick-audit-v52-b")) + "\n")
    with pytest.raises(ValueError, match="changed after reviewer C started"):
        store.case(0)
    with pytest.raises(ValueError, match="changed after reviewer C started"):
        store.submit(0, [{"form": "depth", "correct": True}])


def test_c_http_case_and_image_do_not_leak_private_identity(tmp_path: Path) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=0)
    store = module.V52AdjudicationStore(pack)
    handler = module.core.make_handler(store, None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        case_body = urllib.request.urlopen(f"{origin}/api/case/0", timeout=5).read()
        case = json.loads(case_body)
        assert case["forms"][0]["object_class"]["reviewer_B"]["comment"]
        serialized = case_body.decode("utf-8")
        for forbidden in (
            '"sample_id"',
            '"scene_id"',
            '"image_path"',
            '"source_payload"',
            '"source_instance_id"',
            '"source_label"',
            '"sensor_measurements"',
            "private-scene",
        ):
            assert forbidden not in serialized
        image_body = urllib.request.urlopen(f"{origin}/api/image/0", timeout=5).read()
        assert image_body == b"public-rgb"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_launcher_declares_conditional_c_lifecycle_and_is_valid_bash() -> None:
    launcher = ROOT / "scripts/run_scannet_rgb_depth_v52_quick_audit.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    for contract in (
        'PORT_C="${PORT_C:-10325}"',
        "quick_scannet_rgb_depth_v52_adjudication.py",
        "WAITING_FOR_A_B|NO_ADJUDICATION_REQUIRED",
        "READY_FOR_ADJUDICATION",
        'stop_side c "$PORT_C"',
        'status_c || failed=1',
        "reviewer_c_%s.log",
        "reviewer_c_watcher_%s.log",
        "watcher_loop",
        "start_watcher",
        "stop_watcher",
        "ERROR_A_B_SERVICE_UNHEALTHY",
        'HOST="${HOST:-127.0.0.1}"',
        "A non-loopback HOST requires TOKEN",
        'tmux set-environment -g GSA_QUICK_AUDIT_TOKEN "$TOKEN"',
    ):
        assert contract in text


def _free_ports(count: int) -> list[int]:
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _launcher_env(tmp_path: Path, pack: Path, ports: list[int]) -> dict[str, str]:
    port_a, port_b, port_c = ports
    return {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "PACK": str(pack),
        "LOG_DIR": str(tmp_path / "logs"),
        "PORT_A": str(port_a),
        "PORT_B": str(port_b),
        "PORT_C": str(port_c),
        "WATCH_INTERVAL_SECONDS": "1",
    }


def _run_launcher(env: dict[str, str], action: str) -> subprocess.CompletedProcess[str]:
    launcher = ROOT / "scripts/run_scannet_rgb_depth_v52_quick_audit.sh"
    return subprocess.run(
        ["bash", str(launcher), action],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )


def _wait_until(predicate, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("condition did not become true before timeout")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    any(shutil.which(command) is None for command in ("tmux", "curl", "ss")),
    reason="launcher lifecycle requires tmux, curl, and ss",
)
@pytest.mark.parametrize(
    ("review_state", "expected_state", "c_should_run"),
    [
        ("pending", module.WAITING, False),
        ("agreement", module.NOT_REQUIRED, False),
        ("disagreement", module.READY, True),
    ],
)
def test_launcher_real_lifecycle_is_conditional_on_c_state(
    tmp_path: Path,
    review_state: str,
    expected_state: str,
    c_should_run: bool,
) -> None:
    pack, labels = _make_pack(tmp_path)
    if review_state == "agreement":
        _complete_ab(pack, labels, disagreement_index=None)
    elif review_state == "disagreement":
        _complete_ab(pack, labels, disagreement_index=0)
    port_a, port_b, port_c = _free_ports(3)
    launcher = ROOT / "scripts/run_scannet_rgb_depth_v52_quick_audit.sh"
    env = _launcher_env(tmp_path, pack, [port_a, port_b, port_c])
    try:
        started = subprocess.run(
            ["bash", str(launcher), "start"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
        assert started.returncode == 0, started.stdout + started.stderr
        assert expected_state in started.stdout
        status = subprocess.run(
            ["bash", str(launcher), "status"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert status.returncode == 0, status.stdout + status.stderr
        assert expected_state in status.stdout or c_should_run
        if c_should_run:
            stats = json.loads(
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port_c}/api/stats", timeout=5
                ).read()
            )
            assert stats["images"] == stats["expected_decisions"] == 1
        else:
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", port_c), timeout=0.5)
    finally:
        subprocess.run(
            ["bash", str(launcher), "stop"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )


@pytest.mark.skipif(
    any(shutil.which(command) is None for command in ("tmux", "curl", "ss")),
    reason="watcher lifecycle requires tmux, curl, and ss",
)
@pytest.mark.parametrize(
    ("disagreement_index", "expected_state", "c_should_run"),
    [
        (None, module.NOT_REQUIRED, False),
        (0, "READY_C_RUNNING", True),
    ],
)
def test_watcher_observes_completed_a_b_without_second_launcher_command(
    tmp_path: Path,
    disagreement_index: int | None,
    expected_state: str,
    c_should_run: bool,
) -> None:
    pack, labels = _make_pack(tmp_path)
    ports = _free_ports(3)
    env = _launcher_env(tmp_path, pack, ports)
    state_path = tmp_path / "logs" / f"reviewer_c_watcher_{ports[2]}.state.tsv"
    try:
        started = _run_launcher(env, "start")
        assert started.returncode == 0, started.stdout + started.stderr
        assert module.WAITING in started.stdout
        _complete_ab(pack, labels, disagreement_index=disagreement_index)

        def transitioned() -> bool:
            return (
                state_path.is_file()
                and expected_state in state_path.read_text(encoding="utf-8")
                and _port_open(ports[2]) is c_should_run
            )

        _wait_until(transitioned)
        status = _run_launcher(env, "status")
        assert status.returncode == 0, status.stdout + status.stderr
        assert "reviewer_c_watcher: RUNNING" in status.stdout
        assert expected_state in status.stdout
    finally:
        _run_launcher(env, "stop")


@pytest.mark.skipif(
    any(shutil.which(command) is None for command in ("tmux", "curl", "ss")),
    reason="watcher lifecycle requires tmux, curl, and ss",
)
def test_watcher_is_idempotent_reports_crash_recovers_c_and_cleans_exactly(
    tmp_path: Path,
) -> None:
    pack, labels = _make_pack(tmp_path)
    _complete_ab(pack, labels, disagreement_index=0)
    ports = _free_ports(3)
    env = _launcher_env(tmp_path, pack, ports)
    watcher_session = f"gsa_v52_watcher_{ports[0]}_{ports[1]}_{ports[2]}"
    c_session = f"gsa_v52_c_{ports[2]}"
    try:
        first = _run_launcher(env, "start")
        assert first.returncode == 0, first.stdout + first.stderr
        _wait_until(lambda: _port_open(ports[2]))
        watcher_pid = subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", watcher_session, "#{pane_pid}"],
            text=True,
        ).strip()
        second = _run_launcher(env, "start")
        assert second.returncode == 0, second.stdout + second.stderr
        assert "already running" in second.stdout
        assert subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", watcher_session, "#{pane_pid}"],
            text=True,
        ).strip() == watcher_pid

        subprocess.run(["tmux", "kill-session", "-t", watcher_session], check=True)
        crashed = _run_launcher(env, "status")
        assert crashed.returncode != 0
        assert "reviewer_c_watcher: STOPPED" in crashed.stdout
        recovered = _run_launcher(env, "start")
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        assert "reviewer_c_watcher started" in recovered.stdout

        state_path = tmp_path / "logs" / f"reviewer_c_watcher_{ports[2]}.state.tsv"
        before_c_crash_state_mtime = state_path.stat().st_mtime_ns
        subprocess.run(["tmux", "kill-session", "-t", c_session], check=True)
        _wait_until(lambda: not _port_open(ports[2]))
        _wait_until(
            lambda: subprocess.run(
                ["tmux", "has-session", "-t", c_session], capture_output=True
            ).returncode
            == 0
            and _port_open(ports[2])
        )
        _wait_until(
            lambda: state_path.is_file()
            and state_path.stat().st_mtime_ns > before_c_crash_state_mtime
            and "READY_C_RUNNING" in state_path.read_text(encoding="utf-8")
        )

        with (pack / "annotator_b/annotations/annotations.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(_vote(labels[0], "quick-audit-v52-b")) + "\n")
        _wait_until(
            lambda: state_path.is_file()
            and module.NOT_REQUIRED in state_path.read_text(encoding="utf-8")
            and not _port_open(ports[2])
        )
    finally:
        stopped = _run_launcher(env, "stop")
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    for session in (
        watcher_session,
        c_session,
        f"gsa_v52_a_{ports[0]}",
        f"gsa_v52_b_{ports[1]}",
    ):
        assert subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True
        ).returncode != 0
    assert all(not _port_open(port) for port in ports)
