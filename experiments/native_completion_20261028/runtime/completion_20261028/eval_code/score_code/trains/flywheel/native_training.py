"""Source-bound atomic training records without changing frozen v2/v3 contracts.

This adapter preserves native targets and explicitly distinguishes source-backed
supervision from strong-model adjudication. A caller must first verify the source
file and immutable selection manifest. Row checksums identify that evidence;
they do not, on their own, prove semantic correctness or grant redistribution.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

VERSION = "anvil-source-native-train-v1"
QA_VERSION = "anvil-source-qa-replay-train-v1"
FAMILIES = {**dict.fromkeys(("bbox", "detection", "mask", "point"), "G"),
            **dict.fromkeys(("depth_order", "metric_comparison", "direction", "relation_graph"), "S"),
            **dict.fromkeys(("affordance", "dof", "action_cot"), "A")}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DENSE = {"coco_train2017", "pixmo_points"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON number")
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    # JSON exponents such as 1e999 bypass parse_constant.
    canonical(result)
    return result


def _text(value):
    require(isinstance(value, str) and bool(value.strip()), "empty text field")


def _coords(value, size, *, box=False):
    require(isinstance(value, list) and len(value) == size, "coordinate shape")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1000 for x in value),
            "coordinate range")
    if box:
        require(value[0] < value[2] and value[1] < value[3], "empty or unordered box")


def _edge(value, metric=False):
    keys = {"subject", "object", "predicate", "frame"} | ({"value", "unit"} if metric else set())
    require(isinstance(value, dict) and set(value) == keys, "relation fields")
    _text(value["subject"]); _text(value["object"])
    require(value["subject"] != value["object"], "self relation")
    require(value["frame"] == "image_box_centers", "unsupported relation frame")
    if metric:
        require(value["predicate"] == "distance" and value["unit"] == "normalized_image",
                "image-plane distance is not a physical measurement")
        distance = value["value"]
        require(type(distance) in (int, float) and math.isfinite(distance)
                and 0 <= distance <= math.sqrt(2) + 1e-6, "image distance range")
    else:
        require(value["predicate"] in {"left_of", "right_of", "above", "below"}, "direction predicate")


def validate_target(form, variant, answer, prompt):
    """Validate representation, not independent semantic quality or execution."""
    if form in {"bbox", "detection", "mask", "point"}:
        require(variant == "anvil-source-native-v1", "unsupported grounding variant")
        require(isinstance(answer, list) and bool(answer), "empty grounding reference")
        field = {"bbox": "bbox_2d", "detection": "bbox_2d", "point": "point_2d", "mask": "polygons"}[form]
        for item in answer:
            require(isinstance(item, dict) and set(item) == {"label", field}, "grounding fields")
            _text(item["label"])
            if form == "mask":
                polygons = item[field]
                require(isinstance(polygons, list) and bool(polygons), "empty mask components")
                for polygon in polygons:
                    require(isinstance(polygon, list) and len(polygon) >= 3, "short polygon")
                    for point in polygon:
                        _coords(point, 2)
            else:
                _coords(item[field], 2 if form == "point" else 4, box=form != "point")
    elif form in {"direction", "metric_comparison", "relation_graph"}:
        require(variant == "anvil-source-native-v1", "unsupported image relation variant")
        if form == "relation_graph":
            require(isinstance(answer, list) and len(answer) >= 2, "short relation graph")
            for edge in answer:
                _edge(edge)
            require(len({canonical(e) for e in answer}) == len(answer), "duplicate relation edges")
        else:
            _edge(answer, metric=form == "metric_comparison")
    elif form == "depth_order":
        require(variant == "anvil-prism-depth-order-v1", "unsupported calibrated-depth variant")
        require(isinstance(answer, list) and bool(answer), "empty depth reference")
        for item in answer:
            require(isinstance(item, dict) and set(item) == {
                "task", "subject_id", "object_id", "predicate", "frame", "truth"}, "depth fields")
            require(item["task"] == "relative_depth" and item["frame"] == "camera"
                    and item["truth"] is True, "depth task or frame")
            require(item["predicate"] in {"nearer", "farther", "same_depth"}, "depth predicate")
            _text(item["subject_id"]); _text(item["object_id"])
            require(item["subject_id"] != item["object_id"], "self depth relation")
    elif form == "affordance":
        require(variant == "anvil-affordance-region-v1", "unsupported affordance variant")
        require(isinstance(answer, dict) and set(answer) == {"affordance", "region_bbox_2d"},
                "affordance region is not a contact point")
        _text(answer["affordance"]); _coords(answer["region_bbox_2d"], 4, box=True)
    elif form == "dof":
        require(variant == "anvil-prism-gripper-se3-v1", "unsupported gripper variant")
        from tools.map_prism_native import validate_native_gripper
        question = strict_json(prompt.removeprefix("<image>").strip())
        require(isinstance(question, dict) and question.get("contract") == variant,
                "gripper prompt contract")
        context = question.get("input")
        require(isinstance(context, dict) and {"intrinsics", "image_size_wh", "entity"} <= set(context),
                "gripper input context")
        require(isinstance(context["image_size_wh"], list) and len(context["image_size_wh"]) == 2
                and isinstance(context["entity"], dict) and "entity_id" in context["entity"],
                "gripper image/entity context")
        validate_native_gripper(answer, context["intrinsics"], *context["image_size_wh"])
        require(answer["entity_id"] == context["entity"]["entity_id"], "gripper entity mismatch")
    elif form == "action_cot":
        require(variant == "anvil-alfred-expert-continuation-sft-v1", "unsupported plan variant")
        require(isinstance(answer, dict) and set(answer) == {
            "target_object_type", "target_role", "next_expert_action", "remaining_high_level_actions"},
            "expert continuation fields")
        for name in ("target_object_type", "target_role", "next_expert_action"):
            _text(answer[name])
        steps = answer["remaining_high_level_actions"]
        require(isinstance(steps, list) and bool(steps), "empty expert continuation")
        for step in steps:
            require(isinstance(step, dict) and set(step) == {"action", "args"}, "expert step fields")
            _text(step["action"])
            require(isinstance(step["args"], list), "expert step arguments")
            for argument in step["args"]:
                # Some original ALFRED steps use an empty string for an
                # unspecified argument. Preserve it, never invent an object.
                require(isinstance(argument, str), "expert argument must be a source string")
    else:
        raise ValueError("unknown native form")


def bind_training_record(raw_line: bytes, binding: dict):
    """Wrap an unchanged source SFT row using a verified immutable index entry.

    ``binding`` must come from a hash-checked source/selection manifest; callers
    are responsible for checking complete source files and the image inventory.
    Native held-out assignments cannot be overridden. Dense evaluation_candidate
    is a provisional source designation, not a frozen benchmark assignment.
    """
    require(isinstance(raw_line, bytes) and bool(raw_line.strip()), "source row bytes required")
    record = strict_json(raw_line)
    require(isinstance(record, dict) and isinstance(binding, dict), "record/binding must be objects")
    _text(record.get("id"))
    meta = record.get("anvil_meta", {})
    require(isinstance(meta, dict), "missing source metadata")
    if meta.get("record_type") == "qa_replay":
        return bind_qa_record(raw_line, binding)
    form, source = meta.get("native_form"), meta.get("source")
    require(form in FAMILIES and isinstance(source, str) and source, "unknown native source/form")
    variant = meta.get("native_contract", meta.get("native_variant", meta.get("schema")))
    require(meta.get("split") == "train" or
            (source in DENSE and meta.get("split") == "evaluation_candidate"), "source is frozen nontraining")
    if source not in DENSE:
        require(meta.get("training_approved") is True, "source training approval missing")
    require(binding.get("split") == "train", "selection is not training")
    require(binding.get("source_uid") == record.get("id"), "source record identity mismatch")
    require(binding.get("source_line_sha256") == hashlib.sha256(raw_line).hexdigest(), "source row changed")
    require(binding.get("source_size") == len(raw_line), "source row size mismatch")
    images = record.get("images")
    require(isinstance(images, list) and len(images) == 1, "one source image required")
    require(binding.get("image_path") == images[0], "selected image path mismatch")
    require(binding.get("source") == source and binding.get("native_form") == form, "selected source/form mismatch")
    original_group = meta.get("group_id")
    require(original_group == binding.get("group_id"), "selected scene/image group mismatch")
    provenance = meta.get("source_provenance", {})
    require(isinstance(provenance, dict), "invalid source provenance")
    original_hash = (meta.get("image_sha256") or meta.get("input_image_sha256")
                     or provenance.get("image_sha256"))
    if original_hash is not None:
        require(original_hash == binding.get("image_sha256"), "source image identity mismatch")
    if meta.get("decoded_rgb_sha256") is not None:
        require(meta["decoded_rgb_sha256"] == binding.get("decoded_rgb_sha256"),
                "source decoded image identity mismatch")
    require(bool(meta.get("verification")), "missing source verification description")
    require("messages" in record, "missing source messages")
    result = {"id": record["id"], "canonical_uid": record["id"],
              "messages": deepcopy(record["messages"]), "images": deepcopy(record["images"]),
              "gsa_contract_version": VERSION, "gsa_record_type": "atomic",
              "gsa_form": form, "gsa_native_form": form, "gsa_family": FAMILIES[form],
              "gsa_native_variant": variant,
              "gsa_meta": {"split": "train", "admission": "bound_source_native_annotation",
                           "source_verification": meta["verification"],
                           "source_metadata_sha256": digest(meta), "binding": deepcopy(binding),
                           "human_reviewed": False, "strong_model_verified": False,
                           "public_release_eligible": False}}
    result["gsa_meta"]["training_content_sha256"] = digest({k: v for k, v in result.items() if k != "gsa_meta"})
    validate_training_record(result)
    return result


def validate_training_record(record):
    require(isinstance(record, dict), "training record must be an object")
    if record.get("gsa_contract_version") == QA_VERSION:
        return validate_qa_record(record)
    _text(record.get("id"))
    require(record.get("gsa_contract_version") == VERSION and record.get("gsa_record_type") == "atomic",
            "source-native admission requires its own atomic protocol")
    require(not any(key in record for key in ("gsa_stages", "gsa_edges", "gsa_references")),
            "atomic references cannot impersonate a linked dependency task")
    form = record.get("gsa_form")
    require(form in FAMILIES and record.get("gsa_family") == FAMILIES[form]
            and record.get("gsa_native_form") == form, "native family/form mismatch")
    meta = record.get("gsa_meta", {})
    require(isinstance(meta, dict), "training metadata must be an object")
    require(meta.get("split") == "train" and meta.get("admission") == "bound_source_native_annotation",
            "missing source-bound training admission")
    require(meta.get("strong_model_verified") is False and meta.get("human_reviewed") is False
            and meta.get("public_release_eligible") is False, "unsupported adjudication or publication claim")
    binding = meta.get("binding", {})
    require(isinstance(binding, dict), "training binding must be an object")
    for field in ("selection_sha256", "source_sha256", "source_line_sha256", "image_sha256", "decoded_rgb_sha256"):
        require(isinstance(binding.get(field), str) and SHA256.fullmatch(binding[field]), "missing binding " + field)
    for field in ("source_offset", "source_size"):
        require(type(binding.get(field)) is int and binding[field] >= (1 if field == "source_size" else 0),
                "invalid source offset/size")
    require(binding.get("split") == "train" and binding.get("source_uid") == record.get("id")
            and record.get("canonical_uid") == record.get("id") and binding.get("native_form") == form,
            "record and immutable index entry disagree")
    for field in ("source", "group_id", "image_path"):
        _text(binding.get(field))
    require(record.get("images") == [binding["image_path"]], "image identity changed")
    require(meta.get("training_content_sha256") == digest({k: v for k, v in record.items() if k != "gsa_meta"}),
            "training content changed after source binding")
    require(isinstance(meta.get("source_metadata_sha256"), str)
            and SHA256.fullmatch(meta["source_metadata_sha256"]), "source metadata hash missing")
    _text(meta.get("source_verification"))
    messages = record.get("messages")
    require(isinstance(messages, list) and len(messages) == 2
            and all(isinstance(message, dict) for message in messages)
            and [m.get("role") for m in messages] == ["user", "assistant"], "atomic prompt/reference pair required")
    for message in messages:
        _text(message.get("content"))
    answer = strict_json(messages[-1]["content"])
    validate_target(form, record.get("gsa_native_variant"), answer, messages[0]["content"])


def bind_qa_record(raw_line, binding):
    """Retain independently matched official QA as reference-only replay."""
    record = strict_json(raw_line)
    require(isinstance(record, dict) and isinstance(binding, dict), "QA record/binding must be objects")
    meta = record.get("anvil_meta", {})
    require(isinstance(meta, dict), "QA source metadata must be an object")
    require(meta.get("schema") == "anvil-clevr-official-qa-replay-v1" and meta.get("source") == "CLEVR"
            and meta.get("record_type") == "qa_replay" and meta.get("split") == "train"
            and meta.get("training_approved") is True and meta.get("independent_question") is True,
            "QA requires an independently matched official training question")
    require(binding.get("split") == "train" and binding.get("native_form") == "QA"
            and binding.get("source") == meta["source"] and binding.get("source_uid") == record.get("id")
            and binding.get("group_id") == meta.get("group_id"), "QA selection identity mismatch")
    require(binding.get("source_line_sha256") == hashlib.sha256(raw_line).hexdigest()
            and binding.get("source_size") == len(raw_line), "QA source bytes changed")
    require(record.get("images") == [binding.get("image_path")]
            and binding.get("image_sha256") == meta.get("image_sha256")
            and binding.get("decoded_rgb_sha256") == meta.get("decoded_rgb_sha256"), "QA image binding mismatch")
    for key in ("official_archive_sha256", "official_questions_sha256"):
        require(isinstance(meta.get(key), str) and SHA256.fullmatch(meta[key]), "Missing official QA provenance")
    require(type(meta.get("official_question_id")) is int and meta["official_question_id"] >= 0,
            "Missing official question identity")
    result = {"id": record["id"], "canonical_uid": record["id"], "images": deepcopy(record["images"]),
        "messages": deepcopy(record["messages"]), "gsa_contract_version": QA_VERSION,
        "gsa_record_type": "qa_replay", "gsa_form": "QA", "gsa_native_form": "QA", "gsa_family": "QA",
        "gsa_native_variant": meta["schema"], "gsa_meta": {"split": "train", "admission": "bound_official_qa_replay",
            "binding": deepcopy(binding), "source_verification": meta.get("verification"),
            "source_metadata_sha256": digest(meta), "human_reviewed": False, "strong_model_verified": False,
            "public_release_eligible": False, "reference_only": True,
            "official_question_id": meta["official_question_id"], "official_questions_sha256": meta["official_questions_sha256"]}}
    result["gsa_meta"]["training_content_sha256"] = digest({k: v for k, v in result.items() if k != "gsa_meta"})
    validate_qa_record(result)
    return result


def validate_qa_record(record):
    require(isinstance(record, dict), "QA record must be an object")
    require(record.get("gsa_contract_version") == QA_VERSION and record.get("gsa_record_type") == "qa_replay"
            and record.get("gsa_native_variant") == "anvil-clevr-official-qa-replay-v1"
            and all(record.get(k) == "QA" for k in ("gsa_form", "gsa_native_form", "gsa_family")),
            "QA replay is not a native spatial family")
    require(not any(k in record for k in ("gsa_stages", "gsa_edges", "gsa_references")), "QA cannot impersonate a linked task")
    meta = record.get("gsa_meta", {})
    require(isinstance(meta, dict), "QA metadata must be an object")
    binding = meta.get("binding", {})
    require(isinstance(binding, dict), "QA binding must be an object")
    require(meta.get("split") == "train" and meta.get("admission") == "bound_official_qa_replay"
            and meta.get("reference_only") is True, "QA must remain training-only reference replay")
    require(all(meta.get(k) is False for k in ("human_reviewed", "strong_model_verified", "public_release_eligible")),
            "Unsupported QA adjudication or publication claim")
    for key in ("selection_sha256", "source_sha256", "source_line_sha256", "image_sha256", "decoded_rgb_sha256"):
        require(isinstance(binding.get(key), str) and SHA256.fullmatch(binding[key]), "Missing QA binding " + key)
    for key in ("source_offset", "source_size"):
        require(type(binding.get(key)) is int and binding[key] >= (1 if key == "source_size" else 0), "Invalid QA offset/size")
    require(binding.get("source_uid") == record.get("id") == record.get("canonical_uid")
            and binding.get("native_form") == "QA" and binding.get("split") == "train"
            and binding.get("source") == "CLEVR" and record.get("images") == [binding.get("image_path")],
            "QA record/index mismatch")
    for key in ("image_path", "group_id"): _text(binding.get(key))
    _text(record.get("id")); _text(meta.get("source_verification"))
    for key in ("source_metadata_sha256", "official_questions_sha256"):
        require(isinstance(meta.get(key), str) and SHA256.fullmatch(meta[key]), "Missing QA provenance hash")
    require(type(meta.get("official_question_id")) is int and meta["official_question_id"] >= 0, "Missing QA question ID")
    require(meta.get("training_content_sha256") == digest({k: v for k, v in record.items() if k != "gsa_meta"}),
            "QA content changed after binding")
    messages = record.get("messages")
    require(isinstance(messages, list) and len(messages) == 2 and all(isinstance(m, dict) for m in messages)
            and [m.get("role") for m in messages] == ["user", "assistant"], "QA prompt/reference pair required")
    for message in messages: _text(message.get("content"))
