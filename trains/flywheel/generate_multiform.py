#!/usr/bin/env python3
"""Multi-form GSA flywheel: generates all 8 task forms per image with cascaded
context, per-form hard validators, retry-on-invalid, and L1 cross-check.

Emits BOTH:
  raw/<form>.jsonl    -- verified annotation records (audit + dashboard)
  swift/<form>.jsonl  -- ms-swift {messages, images} training records

Balance: each form has a target share; a form stops once it hits its quota so the
corpus does not skew toward the cheap forms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trains.flywheel import prompts, prompts_ext, verifier  # noqa: E402
from trains.flywheel.contracts import (TRAINING_INSTRUCTION_VERSION, context_for,
                                       grounding, image_sha256, task_columns,
                                       training_instruction)  # noqa: E402
from trains.flywheel.validators import validate, validate_structural  # noqa: E402
from trains.flywheel.vlm_client import chat_json  # noqa: E402

# Target share per form (sums to 1.0). Cheap/abundant forms are capped lower.
# Balanced shares. Mask is anchored to the cascade object (same object as bbox/dof/
# affordance) and validated for a genuinely non-rectangular outline, so it can carry a
# normal share; invalid masks are retried rather than silently down-weighted.
FORM_SHARE = {
    "bbox": 0.12, "detection": 0.14, "mask": 0.12, "point": 0.10,
    "depth": 0.10, "affordance": 0.14, "dof": 0.14, "action_cot": 0.14,
}

MIN_SEMANTIC_CONFIDENCE = 0.80

_lock = threading.Lock()
counts = Counter()
rejects = Counter()


def _is_permanent_image_failure(exc: Exception) -> bool:
    """Return whether retrying the same local image cannot repair the failure."""
    return isinstance(exc, (OSError, ValueError))


def _is_terminal_image_failure(
        exc: Exception, attempt: int, max_transient_failures: int) -> bool:
    return (
        _is_permanent_image_failure(exc)
        or attempt >= max_transient_failures
    )


def _src(img: str) -> str:
    if "bridge" in img:
        return "bridge_data_v2"
    if "changeit" in img.lower():
        return "changeit_frames"
    p = Path(img).parts
    return p[-3] if len(p) >= 3 else "unknown"


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, dict) else []


def _needed(form: str, quota: dict) -> bool:
    with _lock:
        return counts[form] < quota.get(form, 0)


def _correction_prompt(base: str, reason: str) -> str:
    """Turn a validator failure into an explicit second-pass instruction."""
    return (
        f"{base}\n\nYour previous candidate failed the deterministic check "
        f"({reason}). Re-read the image and return a corrected answer that "
        "satisfies every geometric and semantic rule. Do not explain the correction."
    )


def _audit_candidates_enabled() -> bool:
    return os.environ.get("AUDIT_CANDIDATE_MODE", "0") == "1"


def _accept_form(form: str, payload, context):
    """Return (accepted, reason, audit_only) for a candidate payload."""
    ok, reason = validate(form, payload, context)
    if ok:
        return True, "", False
    if _audit_candidates_enabled():
        structural_ok, structural_reason = validate_structural(
            form, payload, context)
        if structural_ok:
            return True, reason or structural_reason, True
    return False, reason, False


def gen_one(img: str, quota: dict, scene: str = "embodied_home") -> list[tuple]:
    """Generate one contract-locked G -> S -> A cascade."""
    out = []
    source = _src(img)
    try:
        image_id = image_sha256(img)
    except OSError:
        with _lock:
            rejects["image:unreadable"] += 1
        return out

    scene_check = None
    scene_feedback = ""
    for attempt in range(2):
        scene_prompt = (
            prompts_ext.SCENE_ELIGIBILITY if not scene_feedback
            else _correction_prompt(prompts_ext.SCENE_ELIGIBILITY, scene_feedback)
        )
        candidate = chat_json(
            img, scene_prompt, max_tokens=220, temperature=0.0,
            use_cache=(attempt == 0))
        if candidate is None:
            raise RuntimeError("scene eligibility request failed")
        if not isinstance(candidate, dict):
            scene_feedback = "invalid_response"
            with _lock:
                rejects["scene:invalid_response"] += 1
            continue
        scene_reason = candidate.get("reason_code")
        if (
            not isinstance(candidate.get("eligible"), bool)
            or not isinstance(scene_reason, str)
            or (candidate["eligible"] and scene_reason != "single_scene")
            or (not candidate["eligible"] and scene_reason == "single_scene")
        ):
            scene_feedback = "inconsistent_response"
            with _lock:
                rejects["scene:inconsistent_response"] += 1
            continue
        scene_confidence = candidate.get("confidence")
        if candidate["eligible"] is True and (
            not isinstance(scene_confidence, (int, float))
            or scene_confidence < MIN_SEMANTIC_CONFIDENCE
        ):
            scene_feedback = "eligible_scene_requires_confidence_at_least_0.80"
            with _lock:
                rejects["scene:low_confidence"] += 1
            continue
        scene_check = candidate
        break
    if scene_check is None:
        return out
    if scene_check.get("eligible") is not True:
        reason = str(scene_check.get("reason_code") or "ineligible")
        with _lock:
            rejects[f"scene:{reason}"] += 1
        return out

    detection_payload = None
    groundings = []
    cross_checks = []
    detection_feedback = ""
    for attempt in range(2):
        detection_prompt = (
            prompts_ext.DETECTION if not detection_feedback
            else _correction_prompt(prompts_ext.DETECTION, detection_feedback)
        )
        detection_response = chat_json(
            img, detection_prompt, max_tokens=700,
            temperature=0.1 + 0.2 * attempt, use_cache=(attempt == 0))
        if detection_response is None:
            raise RuntimeError("detection request failed")
        raw_detection = _as_list(detection_response)
        ok, why = validate("detection", raw_detection, None)
        if not ok:
            detection_feedback = why
            with _lock:
                rejects[f"detection:{why}"] += 1
            continue

        proposed = [
            grounding(item.get("label", ""), item.get("bbox_2d") or item.get("bbox"))
            for item in raw_detection
        ]
        checked_response = chat_json(
            img, prompts.cross_check_set(proposed), temperature=0.0,
            max_tokens=420, use_cache=(attempt == 0))
        if checked_response is None:
            raise RuntimeError("detection verification request failed")
        checked = _as_list(checked_response)
        by_index = {
            item.get("index"): item for item in checked
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }
        keep = [
            i for i in range(len(proposed))
            if (i in by_index and by_index[i].get("match") is True
                and isinstance(by_index[i].get("confidence"), (int, float))
                and by_index[i]["confidence"] >= MIN_SEMANTIC_CONFIDENCE)
        ]
        if len(keep) < 2:
            detection_feedback = "cross_check_fewer_than_two_confident_entities"
            with _lock:
                rejects["detection:l1_fewer_than_2"] += 1
            continue
        detection_payload = [raw_detection[i] for i in keep]
        groundings = [proposed[i] for i in keep]
        cross_checks = [by_index[i] for i in keep]
        ok, why = validate("detection", detection_payload, None)
        if ok:
            break
        with _lock:
            rejects[f"detection:{why}"] += 1
        detection_feedback = why
        detection_payload = None

    if detection_payload is None:
        return out

    primary = groundings[0]

    def emit(form: str, payload, spatial: dict | None = None,
             semantic_gate: dict | None = None, audit_only: bool = False,
             audit_reason: str = ""):
        context = context_for(form, groundings, spatial)
        audit = verifier.score_form(form, payload, context, cross_checks)
        if audit["verdict"] != "verified" and not audit_only:
            with _lock:
                rejects[f"{form}:audit_{audit['contract_reason'] or 'context'}"] += 1
            return
        raw = {
            "form": form,
            "image_path": img,
            "image_sha256": image_id,
            "source": source,
            "scene": scene,
            "label_version": prompts_ext.LABEL_VERSION,
            "prompt_version": prompts_ext.PROMPT_VERSION,
            "payload": payload,
            "gsa_context": context,
            "_cross_check": cross_checks,
            "_verification": audit,
            "_semantic_gate": {
                "scene": scene_check,
                **(semantic_gate or {}),
            },
        }
        if audit_only:
            raw["_audit_only"] = True
            raw["_audit_reason"] = audit_reason or audit["contract_reason"] or "semantic_review"
            # The local form can pass while a consumed predecessor is withheld.
            # Keep that local measurement without calling the cascade verified.
            audit["local_verdict"] = audit["verdict"]
            audit["verdict"] = "audit_only"
            # Keep uncertain candidates visible to the independent audit, but never
            # serialize them into the training stream.
            out.append((form, raw, None))
            return
        sw = {
            "messages": [
                {"role": "user", "content": "<image>" + training_instruction(form, context)},
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "images": [img],
        }
        sw.update(task_columns(
            form, context, source=source, scene=scene, image_id=image_id))
        sw["gsa_meta"].update({
            "label_version": prompts_ext.LABEL_VERSION,
            "prompt_version": prompts_ext.PROMPT_VERSION,
            "training_instruction_version": TRAINING_INSTRUCTION_VERSION,
        })
        out.append((form, raw, sw))

    if _needed("detection", quota):
        emit("detection", detection_payload)
    if _needed("bbox", quota):
        emit("bbox", [{
            "label": primary["object_class"],
            "bbox_2d": primary["bbox"],
        }])
    if _needed("point", quota):
        point_feedback = ""
        for attempt in range(2):
            point_prompt = prompts_ext.visible_point(primary)
            if point_feedback:
                point_prompt = _correction_prompt(point_prompt, point_feedback)
            response = chat_json(img, point_prompt, max_tokens=220,
                                 temperature=0.1 + 0.2 * attempt, use_cache=(attempt == 0))
            if response is None:
                raise RuntimeError("visible point request failed")
            items = _as_list(response)
            ok, why = validate("point", items, context_for("point", groundings))
            if ok and (len(items) != 1 or items[0].get("label") != primary["object_class"]):
                ok, why = False, "label_or_cardinality_mismatch"
            if ok:
                check = chat_json(img, prompts_ext.verify_visible_point(primary, items),
                                  max_tokens=220, temperature=0.0, use_cache=(attempt == 0))
                if check is None:
                    raise RuntimeError("visible point verification request failed")
                confidence = check.get("confidence") if isinstance(check, dict) else None
                ok = (isinstance(check, dict) and check.get("valid") is True
                      and check.get("reason_code") == "supported"
                      and type(confidence) in (int, float)
                      and MIN_SEMANTIC_CONFIDENCE <= confidence <= 1)
                if ok:
                    emit("point", items, semantic_gate={"point": check})
                    break
                why = "visible_surface_unconfirmed"
            with _lock:
                rejects[f"point:{why}"] += 1
            point_feedback = why

    if _needed("mask", quota):
        mask_feedback = ""
        for attempt in range(2):
            mask_prompt = prompts_ext.mask_polygon(primary)
            if mask_feedback:
                mask_prompt = _correction_prompt(mask_prompt, mask_feedback)
            mask_response = chat_json(
                img, mask_prompt, max_tokens=900,
                temperature=0.2 + 0.3 * attempt, use_cache=(attempt == 0))
            if mask_response is None:
                raise RuntimeError("mask request failed")
            items = _as_list(mask_response)
            context = context_for("mask", groundings)
            ok, why, mask_audit_only = _accept_form("mask", items, context)
            if ok:
                emit("mask", items, audit_only=mask_audit_only,
                     audit_reason=why)
                break
            with _lock:
                rejects[f"mask:{why}"] += 1
            mask_feedback = why

    need_spatial = _needed("depth", quota) or any(
        _needed(form, quota) for form in prompts_ext.ACTION_PROMPT_FN)
    if not need_spatial:
        return out

    depth_payload = None
    depth_gate = None
    depth_audit_only = False
    depth_audit_reason = ""
    depth_context = context_for("depth", groundings)
    depth_feedback = ""
    for attempt in range(2):
        depth_prompt = prompts_ext.depth_order(groundings)
        if depth_feedback:
            depth_prompt = _correction_prompt(depth_prompt, depth_feedback)
        depth_response = chat_json(
            img, depth_prompt, max_tokens=700,
            temperature=0.2 + 0.3 * attempt, use_cache=(attempt == 0))
        if depth_response is None:
            raise RuntimeError("depth request failed")
        items = _as_list(depth_response)
        ok, why, candidate_audit_only = _accept_form(
            "depth", items, depth_context)
        if not ok:
            depth_feedback = why
            with _lock:
                rejects[f"depth:{why}"] += 1
            continue
        verification = chat_json(
            img, prompts_ext.verify_depth_order(groundings, items),
            max_tokens=260, temperature=0.0, use_cache=(attempt == 0))
        if verification is None:
            raise RuntimeError("depth verification request failed")
        confidence = verification.get("confidence") if isinstance(verification, dict) else None
        valid_reason = (
            isinstance(verification, dict)
            and isinstance(verification.get("valid"), bool)
            and isinstance(verification.get("reason_code"), str)
            and (
                (verification["valid"] and verification["reason_code"] == "supported")
                or (not verification["valid"] and verification["reason_code"] != "supported")
            )
        )
        if (valid_reason and verification["valid"] is True
                and isinstance(confidence, (int, float))
                and confidence >= MIN_SEMANTIC_CONFIDENCE):
            depth_payload = items
            depth_gate = verification
            depth_audit_only = candidate_audit_only
            if candidate_audit_only:
                depth_audit_reason = (
                    f"depth_validator_{why or 'semantic_rejection'}"
                )
            break
        if candidate_audit_only and _audit_candidates_enabled():
            depth_payload = items
            depth_gate = (
                verification if isinstance(verification, dict)
                else {"valid": False, "confidence": 0.0,
                      "reason_code": "invalid_response"}
            )
            depth_audit_only = True
            depth_audit_reason = (
                f"depth_verify_{verification.get('reason_code', 'invalid_response')}"
                if isinstance(verification, dict)
                else "depth_verify_invalid_response"
            )
            break
        reason = (
            str(verification.get("reason_code") or "low_confidence")
            if isinstance(verification, dict) else "invalid_response"
        )
        depth_feedback = f"depth_verification_{reason}"
        with _lock:
            rejects[f"depth_verify:{reason}"] += 1
    if depth_payload is None:
        return out

    spatial = {"depth_order": depth_payload}
    if _needed("depth", quota):
        emit("depth", depth_payload, spatial, {"depth": depth_gate},
             audit_only=depth_audit_only, audit_reason=depth_audit_reason)

    # Questionable spatial labels stay visible to the audit but cannot silently
    # become conditioning context for affordance, DoF, or action-plan labels.
    if depth_audit_only:
        return out

    need_action = any(
        _needed(form, quota) for form in ("affordance", "dof", "action_cot"))
    if not need_action:
        return out

    affordance_payload = None
    affordance_audit_only = False
    affordance_audit_reason = ""
    affordance_feedback = ""
    for attempt in range(2):
        affordance_prompt = prompts_ext.affordance_region(primary, spatial)
        if affordance_feedback:
            affordance_prompt = _correction_prompt(
                affordance_prompt, affordance_feedback)
        affordance_response = chat_json(
            img, affordance_prompt, max_tokens=700,
            temperature=0.1 + 0.2 * attempt, use_cache=(attempt == 0))
        if affordance_response is None:
            raise RuntimeError("affordance request failed")
        items = _as_list(affordance_response)
        context = context_for("affordance", groundings, spatial)
        ok, why, candidate_audit_only = _accept_form(
            "affordance", items, context)
        if ok:
            affordance_payload = items
            affordance_audit_only = candidate_audit_only
            affordance_audit_reason = why
            break
        with _lock:
            rejects[f"affordance:{why}"] += 1
        affordance_feedback = why
    if affordance_payload is None:
        return out

    dof_payload = None
    dof_audit_only = False
    dof_audit_reason = ""
    dof_spatial = {**spatial, "affordance": affordance_payload}
    dof_feedback = ""
    for attempt in range(2):
        dof_prompt = prompts_ext.dof_interface(
            primary, spatial, affordance_payload)
        if dof_feedback:
            dof_prompt = _correction_prompt(dof_prompt, dof_feedback)
        dof_response = chat_json(
            img, dof_prompt, max_tokens=700,
            temperature=0.1 + 0.2 * attempt, use_cache=(attempt == 0))
        if dof_response is None:
            raise RuntimeError("dof request failed")
        items = _as_list(dof_response)
        context = context_for("dof", groundings, dof_spatial)
        ok, why, candidate_audit_only = _accept_form(
            "dof", items, context)
        if ok:
            dof_payload = items
            dof_audit_only = candidate_audit_only
            dof_audit_reason = why
            break
        with _lock:
            rejects[f"dof:{why}"] += 1
        dof_feedback = why
    if dof_payload is None:
        return out

    action_payload = None
    action_audit_only = False
    action_audit_reason = ""
    action_spatial = {**dof_spatial, "dof": dof_payload}
    action_feedback = ""
    for attempt in range(2):
        action_prompt = prompts_ext.action_cot(
            primary, spatial, affordance_payload, dof_payload)
        if action_feedback:
            action_prompt = _correction_prompt(action_prompt, action_feedback)
        action_response = chat_json(
            img, action_prompt,
            max_tokens=900, temperature=0.1 + 0.2 * attempt,
            use_cache=(attempt == 0))
        if action_response is None:
            raise RuntimeError("action request failed")
        items = _as_list(action_response)
        context = context_for("action_cot", groundings, action_spatial)
        ok, why, candidate_audit_only = _accept_form(
            "action_cot", items, context)
        if ok:
            action_payload = items
            action_audit_only = candidate_audit_only
            action_audit_reason = why
            break
        with _lock:
            rejects[f"action_cot:{why}"] += 1
        action_feedback = why
    if action_payload is None:
        return out

    manipulation_gate = chat_json(
        img, prompts_ext.verify_manipulation(
            primary, affordance_payload, dof_payload, action_payload),
        max_tokens=300, temperature=0.0, use_cache=True)
    if manipulation_gate is None:
        raise RuntimeError("manipulation verification request failed")
    manipulation_confidence = (
        manipulation_gate.get("confidence")
        if isinstance(manipulation_gate, dict) else None
    )
    manipulation_reason = (
        manipulation_gate.get("reason_code")
        if isinstance(manipulation_gate, dict) else None
    )
    manipulation_schema_ok = (
        isinstance(manipulation_gate, dict)
        and isinstance(manipulation_gate.get("valid"), bool)
        and isinstance(manipulation_reason, str)
        and (
            (manipulation_gate["valid"] and manipulation_reason == "supported")
            or (not manipulation_gate["valid"] and manipulation_reason != "supported")
        )
    )
    manipulation_valid = (
        manipulation_schema_ok
        and manipulation_gate.get("valid") is True
        and isinstance(manipulation_confidence, (int, float))
        and manipulation_confidence >= MIN_SEMANTIC_CONFIDENCE
    )
    if not manipulation_valid and not (
        _audit_candidates_enabled()
        and (affordance_audit_only or dof_audit_only or action_audit_only)
    ):
        reason = (
            str(manipulation_gate.get("reason_code") or "low_confidence")
            if isinstance(manipulation_gate, dict) else "invalid_response"
        )
        with _lock:
            rejects[f"manipulation_verify:{reason}"] += 1
        return out

    semantic_gate = {
        "depth": depth_gate,
        "manipulation": manipulation_gate,
    }
    if _needed("affordance", quota):
        emit("affordance", affordance_payload, spatial, semantic_gate,
             audit_only=affordance_audit_only or not manipulation_valid,
             audit_reason=affordance_audit_reason or "manipulation_review")
    if _needed("dof", quota):
        emit("dof", dof_payload, dof_spatial, semantic_gate,
             audit_only=affordance_audit_only or dof_audit_only or not manipulation_valid,
             audit_reason=(f"upstream_affordance:{affordance_audit_reason}" if affordance_audit_only
                           else dof_audit_reason or "manipulation_review"))
    if _needed("action_cot", quota):
        emit("action_cot", action_payload, action_spatial, semantic_gate,
             audit_only=affordance_audit_only or dof_audit_only or action_audit_only or not manipulation_valid,
             audit_reason=(f"upstream_affordance:{affordance_audit_reason}" if affordance_audit_only
                           else f"upstream_dof:{dof_audit_reason}" if dof_audit_only
                           else action_audit_reason or "manipulation_review"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="data/flywheel_mf")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--target", type=int, default=60000, help="total records target")
    ap.add_argument("--resume", action="store_true",
                    help="skip images recorded in <out>/done_images.txt")
    ap.add_argument("--batch-limit", type=int, default=0,
                    help="after resume filtering, process at most this many images")
    ap.add_argument("--max-transient-failures", type=int, default=3,
                    help="skip an image after this many transient failures")
    args = ap.parse_args()
    if args.max_transient_failures < 1:
        ap.error("--max-transient-failures must be at least 1")

    # Index lines are either "<path>" or "<path>\t<scene>".
    raw_lines = [l.strip() for l in Path(args.index).read_text().splitlines() if l.strip()]
    imgs, scene_of = [], {}
    for l in raw_lines:
        if "\t" in l:
            p, sc = l.split("\t", 1)
            scene_of[p] = sc.strip()
            imgs.append(p)
        else:
            imgs.append(l)
    imgs = imgs[args.offset:]
    if args.limit:
        imgs = imgs[: args.limit]

    # ---- resume: skip images already generated in this output dir ----
    done_path = Path(args.out) / "done_images.txt"
    if args.resume and done_path.is_file():
        done = set(done_path.read_text(errors="ignore").split("\n"))
        before = len(imgs)
        imgs = [i for i in imgs if i not in done]
        print(f"resume: skipping {before - len(imgs):,} already-done images", flush=True)
    if args.batch_limit:
        imgs = imgs[:args.batch_limit]

    quota = {f: max(1, int(args.target * s)) for f, s in FORM_SHARE.items()}

    outp = Path(args.out)
    (outp / "raw").mkdir(parents=True, exist_ok=True)
    (outp / "swift").mkdir(parents=True, exist_ok=True)
    if args.resume:
        for form in FORM_SHARE:
            existing = outp / "swift" / f"{form}.jsonl"
            if existing.is_file():
                with existing.open() as f:
                    counts[form] = sum(1 for line in f if line.strip())
    fh_raw, fh_sw = {}, {}

    def wr(form, raw, sw):
        if form not in fh_raw:
            fh_raw[form] = (outp / "raw" / f"{form}.jsonl").open("a")
            fh_sw[form] = (outp / "swift" / f"{form}.jsonl").open("a")
        fh_raw[form].write(json.dumps(raw, ensure_ascii=False) + "\n")
        if sw is not None:
            fh_sw[form].write(json.dumps(sw, ensure_ascii=False) + "\n")

    failed_path = outp / "failed_images.jsonl"
    failure_attempts = Counter()
    if args.resume and failed_path.is_file():
        with failed_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    failure = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                path = failure.get("image_path")
                if isinstance(path, str) and failure.get("retryable") is True:
                    failure_attempts[path] += 1

    t0 = time.time(); n_img = 0
    permanent_failures = retryable_failures = terminal_retryable_failures = 0
    fh_done = done_path.open("a")
    fh_failed = failed_path.open("a", encoding="utf-8")
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(gen_one, im, quota,
                          scene_of.get(im, "embodied_home")): im for im in imgs}
        for fu in as_completed(futs):
            n_img += 1
            try:
                res = fu.result() or []
            except Exception as exc:
                permanent = _is_permanent_image_failure(exc)
                path = futs[fu]
                failure_attempts[path] += 1
                attempt = failure_attempts[path]
                terminal = _is_terminal_image_failure(
                    exc, attempt, args.max_transient_failures)
                with _lock:
                    rejects[f"runtime:{type(exc).__name__}"] += 1
                fh_failed.write(json.dumps({
                    "timestamp_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "image_path": path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": not permanent,
                    "failure_attempt": attempt,
                    "terminal": terminal,
                }, ensure_ascii=False) + "\n")
                if permanent:
                    permanent_failures += 1
                else:
                    retryable_failures += 1
                    if terminal:
                        terminal_retryable_failures += 1
                if terminal:
                    fh_done.write(path + "\n")
                if n_img % 10 == 0:
                    fh_done.flush()
                    fh_failed.flush()
                continue
            fh_done.write(futs[fu] + "\n")
            with _lock:
                for form, raw, sw in res:
                    wr(form, raw, sw)
                    if sw is not None:
                        counts[form] += 1
            # Flush often: a long run must be inspectable (and resumable) from the
            # first minutes, not only after a 100-image block completes.
            if n_img % 10 == 0:
                for f in list(fh_raw.values()) + list(fh_sw.values()):
                    f.flush()
                fh_done.flush()
                fh_failed.flush()
            if n_img % 100 == 0:
                tot = sum(counts.values()); dt = time.time() - t0
                print(f"[{n_img}/{len(imgs)}] recs={tot} ({n_img/dt:.2f} img/s) "
                      f"forms={dict(counts)}", flush=True)
            if sum(counts.values()) >= args.target:
                for pending in futs:
                    pending.cancel()
                break
    for f in list(fh_raw.values()) + list(fh_sw.values()):
        f.close()
    fh_done.close()
    fh_failed.close()

    stats = {"images_processed": n_img, "records": sum(counts.values()),
             "by_form": dict(counts.most_common()),
             "rejects_by_reason": dict(rejects.most_common(40)),
             "permanent_image_failures": permanent_failures,
             "retryable_image_failures": retryable_failures,
             "terminal_retryable_failures": terminal_retryable_failures,
             "wall_s": round(time.time() - t0, 1)}
    (outp / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
