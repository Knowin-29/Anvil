"""Versioned Anvil-Gen orchestration on the inherited validators and VLM client.

No endpoint is called without --execute. A SQLite ledger freezes partitions,
persists completed requests, and exports train/dev/benchmark separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time

from trains.flywheel.alignment_schema import (
    VERSION, FORMS, SPATIAL_TASKS, NATIVE_CONTRACTS, Candidate, Scene, Verdict,
    answer_schema, student_answer, student_prompt, validate_candidate,
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                except ValueError as exc:
                    raise ValueError(f"invalid JSON at line {number}") from exc


def pixel_hash(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return hashlib.sha256(encode(rgb.size).encode()+rgb.tobytes()).hexdigest()


class Ledger:
    def __init__(self, path, *, seed, benchmark_fraction, fingerprint="offline"):
        if not 0 <= benchmark_fraction <= 1 or not seed:
            raise ValueError("invalid split configuration")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS groups (group_id TEXT PRIMARY KEY, split TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS images (hash TEXT PRIMARY KEY, split TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS scenes
            (key TEXT PRIMARY KEY, group_id TEXT, split TEXT, input_hash TEXT,
             payload TEXT, status TEXT, result TEXT);
          CREATE INDEX IF NOT EXISTS scenes_status ON scenes(status, split);
          CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS call_receipts
            (id INTEGER PRIMARY KEY, request_key TEXT, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS scene_images
            (scene_key TEXT, hash TEXT, PRIMARY KEY(scene_key, hash));
        """)
        config = encode({"version": VERSION, "seed": seed, "benchmark_fraction": benchmark_fraction,
                         "pipeline": fingerprint})
        old = self.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
        if old and old[0] != config:
            raise ValueError("ledger configuration changed; use a new ledger")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('config', ?)", (config,))
        self.seed, self.fraction = seed, benchmark_fraction

    def reserve(self, scene):
        if scene.source_split == "unknown":
            raise ValueError("source split must be resolved before generation")
        if scene.source_split == "train" and not scene.training_approved:
            raise ValueError("training source requires explicit approval")
        fraction = int(digest([self.seed, scene.group_id])[:16], 16) / 2**64
        split = ("benchmark" if scene.source_split == "test" else "dev"
                 if scene.source_split == "validation" else "benchmark"
                 if fraction < self.fraction else "train")
        image_hashes = [pixel_hash(path) for path in scene.images]
        payload = scene.model_dump()
        # Mount-independent identity. Hash full metadata separately to detect edits.
        key = digest([scene.source, scene.scene_id, scene.group_id, image_hashes])
        canonical = {**payload, "images": image_hashes}
        input_hash = digest(canonical)
        with self.lock, self.db:
            old = self.db.execute("SELECT split FROM groups WHERE group_id=?", (scene.group_id,)).fetchone()
            if old and old[0] != split:
                raise ValueError("scene group crosses split boundary")
            for image_hash in image_hashes:
                old = self.db.execute("SELECT split FROM images WHERE hash=?", (image_hash,)).fetchone()
                if old and old[0] != split:
                    raise ValueError("image crosses split boundary")
            old = self.db.execute("SELECT input_hash FROM scenes WHERE key=?", (key,)).fetchone()
            if old and old[0] != input_hash:
                raise ValueError("reserved scene metadata changed")
            self.db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (scene.group_id, split))
            self.db.executemany("INSERT OR IGNORE INTO images VALUES (?,?)", [(h, split) for h in image_hashes])
            self.db.execute("INSERT OR IGNORE INTO scenes VALUES (?,?,?,?,?,'pending',NULL)",
                            (key, scene.group_id, split, input_hash, encode(payload)))
            # A content-identical image may move mounts between resumptions.
            self.db.execute("UPDATE scenes SET payload=? WHERE key=?", (encode(payload), key))
            self.db.executemany("INSERT OR IGNORE INTO scene_images VALUES (?,?)",
                                [(key, h) for h in image_hashes])
        return key, split

    def completed(self, key):
        row = self.db.execute("SELECT status FROM scenes WHERE key=?", (key,)).fetchone()
        return row is not None and row[0] in {"accepted", "rejected"}

    def finish(self, key, result):
        if result["status"] not in {"accepted", "rejected", "error"}:
            raise ValueError("unknown result state")
        with self.lock, self.db:
            self.db.execute("UPDATE scenes SET status=?,result=? WHERE key=?",
                            (result["status"], encode(result), key))

    def cached(self, key):
        with self.lock:
            row = self.db.execute("SELECT payload FROM requests WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO requests VALUES (?,?)", (key, encode(value)))

    def record_call(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT INTO call_receipts(request_key,payload) VALUES (?,?)",
                            (key, encode(value)))

    def bind_models(self, deployments):
        value = encode(deployments)
        with self.lock, self.db:
            old = self.db.execute("SELECT value FROM meta WHERE key='deployments'").fetchone()
            if old and old[0] != value:
                raise ValueError("model deployments changed; use a new ledger")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('deployments', ?)", (value,))

    def summary(self):
        return [{"split": a, "status": b, "scenes": n} for a, b, n in self.db.execute(
            "SELECT split,status,count(*) FROM scenes GROUP BY split,status")]


class ProductionGoal:
    """Count accepted train records and content-distinct images, never attempts.

    SQLite-backed image sets keep memory bounded. Full record counting occurs
    once on resume; completed batches update counters incrementally.
    """
    def __init__(self, ledger, records, additional_images, baseline=None):
        self.db, self.records, self.additional_images = ledger.db, records, additional_images
        self.db.executescript("""
          DROP TABLE IF EXISTS temp.goal_baseline;
          DROP TABLE IF EXISTS temp.goal_seen;
          CREATE TEMP TABLE goal_baseline (hash TEXT PRIMARY KEY);
          CREATE TEMP TABLE goal_seen (hash TEXT PRIMARY KEY);
        """)
        baseline_digest = hashlib.sha256()
        if baseline:
            from functools import lru_cache
            cached_hash = lru_cache(maxsize=16384)(pixel_hash)
            base = Path(baseline).resolve().parent
            with self.db:
                for row in jsonl(baseline):
                    paths = row.get("images") or ([row["image"]] if "image" in row else None)
                    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
                        raise ValueError("baseline rows require images: [local paths] or image: local path")
                    self.db.executemany("INSERT OR IGNORE INTO goal_baseline VALUES (?)",
                        [(cached_hash(str((base / p).resolve())),) for p in paths])
            for (value,) in self.db.execute("SELECT hash FROM goal_baseline ORDER BY hash"):
                baseline_digest.update(value.encode())
        signature = baseline_digest.hexdigest() if baseline else "unspecified"
        old = self.db.execute("SELECT value FROM meta WHERE key='production_baseline'").fetchone()
        if old and old[0] != signature:
            raise ValueError("production baseline changed; use the original image baseline")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('production_baseline',?)", (signature,))
            # Migrate previously accepted ledgers that predate scene-image indexing.
            missing = self.db.execute("SELECT key,payload FROM scenes s WHERE status='accepted' "
                "AND NOT EXISTS (SELECT 1 FROM scene_images i WHERE i.scene_key=s.key)")
            for key, raw in missing:
                self.db.executemany("INSERT OR IGNORE INTO scene_images VALUES (?,?)",
                    [(key, pixel_hash(p)) for p in json.loads(raw)["images"]])
            overlap = self.db.execute("SELECT 1 FROM scene_images i JOIN scenes s ON s.key=i.scene_key "
                "JOIN goal_baseline b ON b.hash=i.hash WHERE s.split!='train' LIMIT 1").fetchone()
            if overlap:
                raise ValueError("original training images overlap a held-out partition")
            self.db.execute("INSERT OR IGNORE INTO goal_seen SELECT i.hash FROM scene_images i "
                "JOIN scenes s ON s.key=i.scene_key WHERE s.split='train' AND s.status='accepted'")
        self.count = self.db.execute("SELECT COALESCE(SUM(json_array_length(result,'$.candidates')),0) "
            "FROM scenes WHERE split='train' AND status='accepted'").fetchone()[0]
        self.images = self.db.execute("SELECT count(*) FROM goal_seen").fetchone()[0]
        self.new_images = self.db.execute("SELECT count(*) FROM goal_seen s WHERE NOT EXISTS "
            "(SELECT 1 FROM goal_baseline b WHERE b.hash=s.hash)").fetchone()[0]
        self.baseline_images = self.db.execute("SELECT count(*) FROM goal_baseline").fetchone()[0]

    def observe(self, key, split, result):
        if split != "train" or result["status"] != "accepted":
            return
        self.count += len(result["candidates"])
        rows = self.db.execute("SELECT i.hash, NOT EXISTS (SELECT 1 FROM goal_baseline b WHERE b.hash=i.hash) "
            "FROM scene_images i WHERE i.scene_key=? AND NOT EXISTS "
            "(SELECT 1 FROM goal_seen s WHERE s.hash=i.hash)", (key,)).fetchall()
        with self.db:
            self.db.executemany("INSERT INTO goal_seen VALUES (?)", [(h,) for h, _ in rows])
        self.images += len(rows)
        self.new_images += sum(new for _, new in rows)

    def reached(self):
        return self.count >= self.records and self.new_images >= self.additional_images

    def summary(self):
        return {"accepted_train_records": self.count, "accepted_train_images": self.images,
                "additional_train_images": self.new_images, "baseline_images": self.baseline_images,
                "target_records": self.records, "target_additional_images": self.additional_images,
                "reached": self.reached()}


class Models:
    """Explicit role endpoints; retain actual returned model identities and usage."""
    def __init__(self, config, ledger):
        from apps.data_flywheel.workers.vlm_client import VLMClient
        self.roles, self.ledger = {}, ledger
        self.max_tokens = config.get("max_tokens", 2048)
        self.generation_attempts = config.get("generation_attempts", 2)
        for role, spec in config["roles"].items():
            if not spec.get("enabled", True):
                continue
            url, model = os.environ.get(spec["url_env"], ""), os.environ.get(spec["model_env"], "")
            if not url or not model:
                raise ValueError(f"configure {spec['url_env']} and {spec['model_env']} for {role}")
            self.roles[role] = VLMClient(base_url=url, model=model,
                api_key=os.environ.get(spec.get("key_env", ""), ""),
                timeout=config.get("timeout_seconds", 120), max_retries=config.get("transport_retries", 1))
        if not {"teacher_G", "teacher_S", "teacher_A", "strong_verifier"} <= self.roles.keys():
            raise ValueError("G/S/A teachers and strong_verifier must all be configured")
        student = self.roles.get("student_verifier")
        strong = self.roles["strong_verifier"]
        if student and student.model == strong.model:
            raise ValueError("recirculated student cannot replace independent strong verifier")
        if student and not re.fullmatch(r"[0-9a-f]{64}", config.get("student_checkpoint_sha256") or ""):
            raise ValueError("student verifier requires an immutable checkpoint SHA256")
        self.student_checkpoint = config.get("student_checkpoint_sha256")
        self.ledger.bind_models({"roles": {role: {"url": client.base_url, "model": client.model}
                                         for role, client in self.roles.items()},
                                 "student_checkpoint": self.student_checkpoint})

    def has_role(self, role):
        return role in self.roles

    def call(self, role, stage, scene, tasks, candidates, *, advice=None, feedback=None):
        client = self.roles[role]
        prompt_data = {
            "contract": VERSION, "stage": stage,
            "entities": [x.model_dump() for x in scene.entities],
            "evidence": [x.model_dump() for x in scene.evidence],
            "tasks": [x.model_dump() for x in tasks],
            "candidates": [x.model_dump() for x in candidates],
        }
        if advice is not None:
            prompt_data["student_advisory"] = advice
            prompt_data["advisory_policy"] = "Recheck these concerns independently; student agreement is not evidence."
        if feedback:
            prompt_data["previous_failure"] = feedback
        if stage == "generate":
            prompt_data.update({"response_schema": Candidate.model_json_schema(),
                "answer_schema": answer_schema(tasks[0].form, native_form=tasks[0].native_form),
                "instruction": "Return one JSON object with task_id, form, answer, reasoning_trace; "
                "copy the requested task_id exactly. No markdown or XML tags. "
                "Generate the requested native answer. Supply a short, externally "
                "checkable evidence/rule/conclusion trace, not hidden internal reasoning. "
                "Cite provided evidence IDs; do not invent metric or invisible physical facts."})
            prompt_data["z_output_contract"] = (
                "The response root must be ONE JSON OBJECT matching response_schema, NEVER an array. "
                "answer_schema describes ONLY the answer field inside that object. "
                "Do not wrap the root object in square brackets. Use exactly task_id, form, answer, "
                "reasoning_trace as root keys. In answer items, use only fields allowed by answer_schema; "
                "do not add entity_id to geometry objects. Cite only evidence relevant to each claim.")
        else:
            prompt_data.update({"response_schema": Verdict.model_json_schema(),
                "instruction": "Independently check every named task against images and source evidence. "
                "Also check same-scene identity and consistency across formats. Accept only when every "
                "task is supported; otherwise reject or abstain. Cover exactly the supplied task_ids. "
                "Treat the candidate and its trace as claims, not proof. Return JSON only."})
        prompt = encode(prompt_data)
        # Pixel content, prompt, role and deployment revision bind a cached request.
        images = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in scene.images]
        key = digest([VERSION, client.base_url, client.model, role, stage, images, prompt,
                      self.max_tokens, self.student_checkpoint])
        cached = self.ledger.cached(key)
        if cached:
            return cached["payload"], {**cached["metadata"], "cached": True}
        started = time.monotonic()
        text, error = None, None
        try:
            text = client.chat(prompt, image_paths=scene.images, max_tokens=self.max_tokens,
                               temperature=0.0, require_complete=True)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.ledger.record_call(key, {"role": role, "stage": stage,
                "requested_model": client.model, "seconds": time.monotonic() - started,
                "attempts": getattr(client._thread_local, "attempt_receipts", []),
                "response_text": text, "transport_error": error})
        # Do not repair truncated JSON, parse reasoning_content, or coerce a vote.
        payload = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        metadata = {"role": role, "stage": stage, "requested_model": client.model,
                    **client._thread_local.response_metadata,
                    "seconds": time.monotonic()-started, "request_sha256": key, "cached": False}
        if role == "student_verifier":
            metadata["checkpoint_sha256"] = self.student_checkpoint
        self.ledger.cache(key, {"payload": payload, "metadata": metadata})
        return payload, metadata


def _verdict(payload, tasks):
    value = Verdict.model_validate(payload)
    expected = {t.task_id for t in tasks}
    if set(value.task_ids) != expected or len(value.task_ids) != len(expected):
        raise ValueError("verifier must cover exactly the requested task IDs")
    return value


def align_scene(scene, split, models):
    """Fail closed. Student judgement is advisory and never edits benchmark labels."""
    candidates, audit = [], []
    tasks = sorted(scene.tasks, key=lambda task: list(FORMS).index(task.form))
    result = {"contract": VERSION, "split": split, "status": "rejected",
              "candidates": [], "audit": audit}
    for task in tasks:
        role = "teacher_" + FORMS[task.form]
        feedback = None
        for attempt in range(getattr(models, "generation_attempts", 1)):
            payload, metadata = models.call(role, "generate", scene, [task], candidates, feedback=feedback)
            audit.append(metadata)
            try:
                value = Candidate.model_validate(payload)
                validate_candidate(value, scene, task, candidates)
            except ValueError as exc:
                # ValidationError dumps can contain target values; send only the
                # error types, never a reference answer disguised as feedback.
                feedback = {"attempt": attempt+1, "failure": "schema_or_rule",
                            "reason": [e["type"] for e in exc.errors()] if hasattr(exc, "errors") else str(exc)}
                audit.append({"stage": "rule_failure", **feedback, "task_id": task.task_id})
                continue
            vote, metadata = models.call(role, "verify", scene, [task], [value])
            verdict = _verdict(vote, [task])
            audit.append({**metadata, "verdict": verdict.model_dump()})
            if verdict.decision == "accept":
                break
            feedback = {"attempt": attempt+1, "failure": "teacher_"+verdict.decision,
                        "reason": verdict.reason}
        else:
            result["reason"] = "task_attempts_exhausted:" + task.task_id
            return result
        candidates.append(value)
    advice = None
    if split == "train" and models.has_role("student_verifier"):
        vote, metadata = models.call("student_verifier", "assist", scene, tasks, candidates)
        advice = _verdict(vote, tasks).model_dump()
        audit.append({**metadata, "verdict": advice, "advisory": True})
    vote, metadata = models.call("strong_verifier", "cross_format_verify", scene, tasks, candidates, advice=advice)
    verdict = _verdict(vote, tasks)
    audit.append({**metadata, "verdict": verdict.model_dump()})
    result.update({"status": "accepted" if verdict.decision == "accept" else "rejected",
                   "reason": "strong_verifier_" + verdict.decision,
                   "candidates": [c.model_dump() for c in candidates]})
    return result


def export_records(ledger, split, family=None):
    if split not in {"train", "dev", "benchmark"}:
        raise ValueError("invalid export split")
    for key, raw, result in ledger.db.execute(
            "SELECT key,payload,result FROM scenes WHERE split=? AND status='accepted' ORDER BY key", (split,)):
        scene, result = Scene.model_validate_json(raw), json.loads(result)
        strong = next(a for a in reversed(result["audit"]) if a.get("role") == "strong_verifier")
        tasks = {t.task_id: t for t in scene.tasks}
        for value in result["candidates"]:
            candidate = Candidate.model_validate(value)
            task = tasks[candidate.task_id]
            if family and FORMS[task.form] != family:
                continue
            record = {"record_id": digest([key, task.task_id]), "images": scene.images,
                      "gsa_form": task.form, "gsa_family": FORMS[task.form],
                      "gsa_native_form": task.native_form,
                      "gsa_contract_version": VERSION, "spatial_task": task.spatial_task,
                      "group_id": scene.group_id, "source": scene.source, "split": split,
                      "teacher_role": "teacher_" + FORMS[task.form],
                      "gsa_meta": {"split": split, "verification": "strong_verifier_accept",
                                   "native_form": task.native_form,
                                   "verifier_model": strong["model"], "ledger_scene_key": key},
                      "messages": [{"role": "user", "content": student_prompt(scene, task)}]}
            target = student_answer(candidate)
            if split == "benchmark":
                record["target"] = target
            else:
                record["messages"].append({"role": "assistant", "content": encode(target)})
            yield record


def score_predictions(targets, predictions):
    """Conservative JSON exact match, not a fabricated semantic/planning metric."""
    predicted = {}
    for row in predictions:
        if row["record_id"] in predicted:
            raise ValueError("duplicate prediction ID")
        predicted[row["record_id"]] = row.get("answer")
    counts, seen = defaultdict(Counter), set()
    for row in targets:
        if row["split"] != "benchmark" or row["record_id"] in seen:
            raise ValueError("invalid or duplicate benchmark target")
        seen.add(row["record_id"])
        name = row.get("spatial_task") or row["gsa_form"]
        counts[name]["total"] += 1
        answer = predicted.get(row["record_id"])
        try:
            payload = answer
            if row["gsa_form"] == "spatial":
                if any("evidence_ids" in item for item in payload):
                    raise ValueError("unexpected private evidence IDs in prediction")
                payload = [{**item, "evidence_ids": ["prediction"]} for item in payload]
            parsed = Candidate.model_validate({"task_id": "prediction", "form": row["gsa_form"],
                "answer": payload, "reasoning_trace": [{"evidence_ids": ["prediction"],
                "rule": "prediction_schema_only", "conclusion": "prediction_schema_only"}]})
            answer = student_answer(parsed)
        except (ValueError, TypeError):
            answer = None
        # Parsed numeric values compare semantically (100 and 100.0 are equal).
        counts[name]["correct"] += int(answer is not None and answer == row["target"])
    if set(predicted) - seen:
        raise ValueError("prediction IDs not in benchmark")
    total = sum(c["total"] for c in counts.values())
    correct = sum(c["correct"] for c in counts.values())
    return {"metric": "native_json_exact_match", "total": total, "correct": correct,
            "exact_match": correct/total if total else None,
            "by_task": {k: dict(v, exact_match=v["correct"]/v["total"]) for k, v in counts.items()}}


def catalog(path):
    """Preserve user-supplied inventory claims; no downloads or licensing inference."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("数据集名称", "").strip():
                continue
            yield {"name": row["数据集名称"], "dimensions": row["类别"].split("、"),
                   "modality": row["模态"], "native_labels": row["原始标签"],
                   "reported_scale": row["数据规模"], "homepage": row["官方主页"],
                   "download": row["下载入口"], "access": row["获取方式"],
                   "status": "inventory_only", "training_approved": False}


def write_jsonl(path, rows):
    # Exclusive creation prevents silently overwriting a frozen release.
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(encode(row)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema")
    schema_selection = schema.add_mutually_exclusive_group()
    schema_selection.add_argument("--form", choices=FORMS)
    schema_selection.add_argument("--native-form", choices=NATIVE_CONTRACTS)
    sources = sub.add_parser("catalog")
    sources.add_argument("--csv", required=True)
    sources.add_argument("--out", required=True)
    adapter = sub.add_parser("import-spatialsense")
    adapter.add_argument("--annotations", required=True)
    adapter.add_argument("--image-index", required=True, help="JSONL: url, image, shared group_id")
    adapter.add_argument("--out", required=True)
    adapter.add_argument("--training-approved", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True, action="append")
    run.add_argument("--config", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--target-records", type=int, default=0)
    run.add_argument("--min-source-images", type=int, default=0,
                     help="minimum total content-distinct source images before model calls")
    run.add_argument("--target-new-images", type=int, default=0)
    run.add_argument("--baseline-images", help="JSONL of original image paths, for content-level novelty counting")
    for name in ("export", "status"):
        command = sub.add_parser(name)
        command.add_argument("--ledger", required=True)
        if name == "export":
            command.add_argument("--split", choices=("train", "dev", "benchmark"), required=True)
            command.add_argument("--out", required=True)
            command.add_argument("--family", choices=("G", "S", "A"))
    score = sub.add_parser("score")
    score.add_argument("--targets", required=True)
    score.add_argument("--predictions", required=True)
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(encode(answer_schema(NATIVE_CONTRACTS[args.native_form]["form"], native_form=args.native_form)
                     if args.native_form else answer_schema(args.form) if args.form else
                     {"version": VERSION, "forms": FORMS, "native_contracts": NATIVE_CONTRACTS,
                      "spatial_tasks": {k: sorted(v) for k,v in SPATIAL_TASKS.items()},
                      "scene": Scene.model_json_schema(), "candidate": Candidate.model_json_schema()}))
        return
    if args.command == "catalog":
        write_jsonl(args.out, catalog(args.csv))
        return
    if args.command == "import-spatialsense":
        from trains.flywheel.alignment_sources import spatialsense_scenes
        rows = json.loads(Path(args.annotations).read_text())
        index, base = {}, Path(args.image_index).resolve().parent
        for item in jsonl(args.image_index):
            if item["url"] in index:
                raise ValueError("duplicate source image URL")
            index[item["url"]] = {"group_id": item["group_id"], "image": str((base / item["image"]).resolve())}
        write_jsonl(args.out, (scene.model_dump() for scene in
                    spatialsense_scenes(rows, index, training_approved=args.training_approved)))
        return
    if args.command == "score":
        print(encode(score_predictions(jsonl(args.targets), jsonl(args.predictions))))
        return
    if args.command in {"export", "status"}:
        # Read-only SQL connection: inspection/export never initializes a ledger.
        ledger = object.__new__(Ledger)
        ledger.db = sqlite3.connect(Path(args.ledger).resolve().as_uri()+"?mode=ro", uri=True)
        if args.command == "status":
            print(encode(ledger.summary()))
        else:
            write_jsonl(args.out, export_records(ledger, args.split, args.family))
        ledger.db.close()
        return
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("contract") != VERSION:
        parser.error("config contract must be " + VERSION)
    for name in ("max_tokens", "timeout_seconds"):
        if type(config.get(name)) is not int or config[name] <= 0:
            parser.error(name + " must be a positive integer")
    if type(config.get("transport_retries", 1)) is not int or not 0 <= config.get("transport_retries", 1) <= 3:
        parser.error("transport_retries must be an integer in [0,3]")
    if type(config.get("generation_attempts", 2)) is not int or not 1 <= config.get("generation_attempts", 2) <= 3:
        parser.error("generation_attempts must be an integer in [1,3]")
    if args.workers < 1 or min(args.limit, args.target_records, args.target_new_images, args.min_source_images) < 0:
        parser.error("workers must be positive; limits and targets must be nonnegative")
    if args.target_new_images and not args.baseline_images:
        parser.error("--target-new-images requires --baseline-images")
    ledger = Ledger(args.ledger, seed=config["split_seed"],
                    benchmark_fraction=config["benchmark_fraction"], fingerprint=digest(config))
    # One orchestrator per ledger; workers share its request cache and transaction lock.
    with Path(str(args.ledger)+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = 0
        for manifest in args.manifest:
            base = Path(manifest).resolve().parent
            for raw in jsonl(manifest):
                raw["images"] = [str((base / p).resolve()) for p in raw["images"]]
                ledger.reserve(Scene.model_validate(raw))
                count += 1
                if count % 1000 == 0:
                    print(encode({"preflight_scenes": count, "model_calls": 0}), flush=True)
        source_images = ledger.db.execute("SELECT count(*) FROM images").fetchone()[0]
        print(encode({"distinct_source_images": source_images,
                      "target_source_images": args.min_source_images}), flush=True)
        if source_images < args.min_source_images:
            print(encode({"status": "source_target_unmet", "model_calls": 0}), flush=True)
            return 3
        goal = ProductionGoal(ledger, args.target_records, args.target_new_images, args.baseline_images) \
            if args.target_records or args.target_new_images else None
        print(encode({"preflight_scenes": count, "partitions": ledger.summary(), "execute": args.execute}), flush=True)
        if goal:
            print(encode({"production_goal": goal.summary()}), flush=True)
        if not args.execute:
            return
        if goal and goal.reached():
            return 0  # Do not require endpoints or make calls after reaching the goal.
        models = Models(config, ledger)

        def work(row):
            key, split, payload = row
            try:
                result = align_scene(Scene.model_validate_json(payload), split, models)
            except Exception as exc:
                result = {"status": "error", "error_type": type(exc).__name__}
            ledger.finish(key, result)
            return key, split, result

        # Bounded batches: executor.map otherwise eagerly loads millions of scenes.
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            cursor = ledger.db.execute("SELECT key,split,payload FROM scenes WHERE status IN ('pending','error') ORDER BY key")
            while True:
                if goal and goal.reached():
                    break
                remaining = min(args.workers, args.limit-done) if args.limit else args.workers
                if remaining <= 0:
                    break
                batch = cursor.fetchmany(remaining)
                if not batch:
                    break
                results = list(pool.map(work, batch))
                states = [result["status"] for _, _, result in results]
                if goal:
                    for key, split, result in results:
                        goal.observe(key, split, result)
                done += len(states)
                print(encode({"processed": done, "batch": dict(Counter(states)),
                              **({"production_goal": goal.summary()} if goal else {})}), flush=True)
        summary = ledger.summary()
        print(encode({"summary": summary}))
        if goal:
            print(encode({"production_goal": goal.summary()}), flush=True)
            return 0 if goal.reached() else 3  # Never claim completion on input exhaustion.
        return 2 if any(item["status"] == "error" for item in summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
