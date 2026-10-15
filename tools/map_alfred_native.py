#!/usr/bin/env python3
"""Map ALFRED validation interactions to source-native action evaluation records.

This is a CPU-only, deterministic conversion, not synthetic chain-of-thought
generation or model evaluation. Validation splits can never enter training here.
The first frame of an interaction's complete low_idx group is a BEFORE image in
the pinned original ALFRED generator. An object's interior point is deliberately
not exported as a physical contact point. Unsupported native Action fields stay
explicitly absent; this converter does not claim complete action_cot coverage.

Run: python tools/map_alfred_native.py --interactions-root MATERIALIZED --output NEW_DIR
SFT: --sft-train-records TRAIN --sft-train-manifest TRAIN_MANIFEST
     --sft-evaluation-records HELDOUT --sft-evaluation-manifest HELDOUT_MANIFEST
     --sft-contract anvil-alfred-expert-continuation-sft-v1 --output NEW_DIR
"""

import argparse
from collections import Counter
import ctypes
import ctypes.util
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time

from PIL import Image, __version__ as pillow_version


REVISION = "f91f4c0c96c7a29f33d0557f86b0a21035379b3b"
UPSTREAM = f"https://github.com/askforalfred/alfred/blob/{REVISION}/"
GENERATOR = {
    "gen/game_states/game_state_base.py":
        "a4f077a0eee1351eba6afc01d5ab50a106718de82739f54eb89d96f1a554dac3",
    "gen/constants.py":
        "5664c7c163a67fc392ab1f44aa106a8111fb693047a644cc46f76ce627055ff9",
    "gen/utils/image_util.py":
        "c5645bd8a85ecad8c98c7e006ca24c48534e3065d8fdf89e8a2d19f59271a4ff",
}
# BEFORE / MIDDLE / AFTER, from the pinned gen/constants.py. CoolObject is a
# high-level macro whose CloseObject step uses the macro's frame counts.
FRAME_COUNTS = {
    "OpenObject": (2, 0, 2), "CloseObject": (2, 0, 2),
    "PickupObject": (5, 0, 10), "PutObject": (5, 0, 10),
    "ToggleObjectOn": (3, 0, 15), "ToggleObjectOff": (1, 0, 5),
    "SliceObject": (3, 0, 7), "CoolObject": (3, 30, 5),
}
ALLOWED_SPLITS = {"valid_seen", "valid_unseen"}
SFT_CONTRACT = "anvil-alfred-expert-continuation-sft-v1"
SFT_INSTRUCTION = (
    "Given this pre-action observation and task goal, provide a JSON expert-plan continuation "
    "with target_object_type, target_role, next_expert_action, and remaining_high_level_actions "
    "(each with action and args). This is one reference expert continuation, not the only valid "
    "or optimal plan and not a guarantee of executability. Do not supply object IDs, coordinates, "
    "chain-of-thought, preconditions, failure modes, or contact points.\nTask goal: "
)
SFT_PROMPT_POLICIES = ("source", "explicit-schema-v1")
# Fixed ontology, NOT a vocabulary inferred from benchmark references. The
# pinned gen/utils/game_util.py:get_discrete_hl_action defines these names,
# including NoOp with empty args. Low-level interactions use FRAME_COUNTS.
HIGH_LEVEL_ACTIONS = ("GotoLocation", "OpenObject", "CloseObject", "PickupObject",
                      "PutObject", "CleanObject", "HeatObject", "CoolObject",
                      "ToggleObject", "SliceObject", "NoOp")
SFT_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False,
    "required": ["target_object_type", "target_role", "next_expert_action", "remaining_high_level_actions"],
    "properties": {
        "target_object_type": {"type": "string", "minLength": 1},
        "target_role": {"enum": ["interaction_object", "destination_receptacle"]},
        "next_expert_action": {"enum": sorted(set(FRAME_COUNTS) - {"CoolObject"})},
        "remaining_high_level_actions": {
            "type": "array", "minItems": 1,
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["action", "args"], "properties": {
                          "action": {"enum": list(HIGH_LEVEL_ACTIONS)},
                          "args": {"type": "array", "items": {"type": "string"}}}}},
    },
}


def render_sft_prompt(prompt, policy="source"):
    """Opt-in prompt-only clarification; never read an answer or alter a task goal."""
    require(policy in SFT_PROMPT_POLICIES, "Unknown ALFRED prompt policy")
    if policy == "source":
        return prompt
    prefix = "<image>\n" + SFT_INSTRUCTION
    require(isinstance(prompt, str) and prompt.startswith(prefix) and prompt[len(prefix):].strip(),
            "Explicit ALFRED schema requires the unchanged source instruction and goal")
    description = SFT_INSTRUCTION.removesuffix("\nTask goal: ")
    instructions = (
        "\nReturn only one JSON object conforming to this output schema: "
        + json.dumps(SFT_OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":"))
        + "\nUse the ALFRED object type spelling for target_object_type. "
        "next_expert_action is the next low-level interaction, not a high-level macro. "
        "For PutObject, target_object_type is the destination receptacle and target_role is "
        "destination_receptacle; otherwise target_role is interaction_object. "
        "remaining_high_level_actions starts with the current high-level subgoal. "
        "Each args value is an ordered JSON list of lowercase object-type strings, never a dictionary. "
        "PutObject high-level args are [object type, receptacle type]; other named high-level "
        "actions take one object-type argument, except NoOp with an empty list. "
        "An unspecified source argument can be an empty string. "
        "High-level ToggleObject differs from low-level ToggleObjectOn/ToggleObjectOff. "
        "Do not include Markdown fences, explanations or extra keys."
    )
    return "<image>\n" + description + instructions + "\nTask goal: " + prompt[len(prefix):]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def mask_area(runs, width, height):
    """Check ALFRED row-major [start, length] foreground runs, without guessing points."""
    require(isinstance(runs, list) and runs, "missing object mask")
    previous_end, area = 0, 0
    for run in runs:
        require(isinstance(run, list) and len(run) == 2, "invalid RLE pair")
        start, length = run
        require(type(start) is int and type(length) is int, "RLE must be integer")
        require(start >= previous_end and length > 0 and start + length <= width * height,
                "overlapping or out-of-bounds object mask")
        previous_end = start + length
        area += length
    return area


def select_preaction(row, trajectory):
    """Fail closed on incomplete groups, index drift, or nonstandard frame counts."""
    low_idx, high_idx = row["low_idx"], row["high_idx"]
    require(type(low_idx) is int and type(high_idx) is int and min(low_idx, high_idx) >= 0,
            "invalid source step index")
    lows, highs = trajectory["plan"]["low_actions"], trajectory["plan"]["high_pddl"]
    require(low_idx < len(lows) and high_idx < len(highs), "source index out of range")
    require(lows[low_idx] == row["source_action"], "interaction differs from original action")
    require(lows[low_idx]["high_idx"] == high_idx == highs[high_idx]["high_idx"],
            "high/low step association mismatch")
    group = [frame for frame in trajectory["images"] if frame["low_idx"] == low_idx]
    require(group and all(frame["high_idx"] == high_idx for frame in group),
            "inconsistent image high_idx")
    # Keep original trajectory order. A reordered list is an error, not a license
    # to choose a different frame. Repeated metadata/missing frames are errors too.
    indices = [int(Path(frame["image_name"]).stem) for frame in group]
    require(indices == list(range(indices[0], indices[0] + len(indices))),
            "noncontiguous interaction frame group")
    require([frame["metadata"] for frame in row["frames"]] == group,
            "materialized interaction does not contain the complete source group")
    action = lows[low_idx]["api_action"]["action"]
    require(action in FRAME_COUNTS and action != "CoolObject", "unsupported interaction action")
    mode = action
    if action == "CloseObject" and highs[high_idx]["planner_action"]["action"] == "CoolObject":
        # The cooling macro closes the fridge twice: the close after PutObject
        # performs cooling (38 frames); the final close after PickupObject is a
        # normal four-frame close. Distinguish these from the official action
        # sequence, not by choosing whichever frame count happens to fit.
        previous = lows[low_idx - 1] if low_idx else None
        following = lows[low_idx + 1] if low_idx + 1 < len(lows) else None
        if previous and previous["high_idx"] == high_idx and previous["api_action"]["action"] == "PutObject":
            target = lows[low_idx]["api_action"]["objectId"]
            require(previous["api_action"].get("receptacleObjectId") == target
                    and following and following["high_idx"] == high_idx
                    and following["api_action"] == {"action": "OpenObject", "objectId": target},
                    "cooling close lacks expected PutObject/CloseObject/OpenObject sequence")
            mode = "CoolObject"
    counts = FRAME_COUNTS[mode]
    require(len(group) == sum(counts) and counts[0] > 0,
            "frame count differs from pinned BEFORE/MIDDLE/AFTER convention")
    return row["frames"][0], counts, mode


def convert_record(row, trajectory, image_index, root, training_scenes=None):
    is_train = row["split"] == "train" and training_scenes is not None
    require(row["split"] in ALLOWED_SPLITS or is_train, "only held-out validation splits are permitted by default")
    require(row["task_id"] == trajectory["task_id"], "task identity mismatch")
    scene = "FloorPlan" + str(trajectory["scene"]["scene_num"])
    require(row["scene_id"] == scene, "scene identity mismatch")
    if is_train:
        require(scene in training_scenes, "training scene not explicitly authorized after held-out exclusion")
    selected, counts, mode = select_preaction(row, trajectory)
    member = selected["member"]
    parts = PurePosixPath(member).parts
    require(len(parts) == 6 and parts[0] == "full_2.1.0" and parts[1] == row["split"]
            and parts[3] == row["task_id"] and parts[4] == "raw_images"
            and ".." not in parts, "unexpected source image member")
    require(parts[-1] == Path(selected["metadata"]["image_name"]).with_suffix(".jpg").name,
            "frame filename differs from official image metadata")
    require(member in image_index, "selected pre-action image not materialized")
    info = image_index[member]
    image_path = (root / "frames" / member).resolve()
    require(image_path.is_relative_to((root / "frames").resolve()), "image outside frames root")
    require(Path(info["path"]).resolve() == image_path, "image index path mismatch")
    require(sha256(image_path) == info["sha256"], "pre-action image hash mismatch")
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    require([width, height] == info["size"] == [300, 300], "unexpected source image dimensions")
    action = row["source_action"]
    api, discrete = action["api_action"], action["discrete_action"]
    require(api["action"] == discrete["action"], "API/discrete action mismatch")
    args = discrete["args"]
    bbox = args["bbox"]
    require(isinstance(bbox, list) and len(bbox) == 4 and all(type(x) is int for x in bbox),
            "invalid object bbox")
    require(0 <= bbox[0] <= bbox[2] <= width and 0 <= bbox[1] <= bbox[3] <= height,
            "object bbox out of bounds")
    area = mask_area(args["mask"], width, height)
    target_key = "receptacleObjectId" if api["action"] == "PutObject" else "objectId"
    target_id = api.get(target_key)
    require(isinstance(target_id, str) and target_id, "missing interaction target")
    annotation = trajectory["turk_annotations"]["anns"][0]
    goal = annotation["task_desc"]
    require(isinstance(goal, str) and goal.strip(), "missing official goal")
    high_steps = []
    for step in trajectory["plan"]["high_pddl"][row["high_idx"]:]:
        if step["planner_action"]["action"] == "End":
            continue
        idx = step["high_idx"]
        high_steps.append({"high_idx": idx, "discrete_action": step["discrete_action"],
                           "instruction": annotation["high_descs"][idx]
                           if idx < len(annotation["high_descs"]) else None})
    require(high_steps, "empty remaining expert plan")
    return {
        "id": f"alfred:{row['split']}:{row['task_id']}:low{row['low_idx']}",
        "schema_version": "anvil-alfred-source-native-action-v1",
        "contract": "expert_plan_continuation_with_preaction_observation",
        "source": "ALFRED", "source_split": row["split"], "task_type": trajectory["task_type"],
        "scene_group": scene, "trajectory_id": row["task_id"],
        "partition": "training_source_only" if is_train else "evaluation_source_only", "training_approved": is_train,
        "native_form_candidate": "action_cot", "native_schema_complete": False,
        "input": {
            "image": str(image_path), "image_sha256": info["sha256"], "image_size": [width, height],
            "goal": goal,
            "instruction": "Given this observation and task goal, describe the next interaction and a remaining high-level plan.",
        },
        "reference": {
            "next_expert_api_action": api, "remaining_high_level_steps": high_steps,
            "target_object_id": target_id, "target_object_type": target_id.split("|", 1)[0],
            "target_role": "destination_receptacle" if target_key == "receptacleObjectId" else "interaction_object",
            "object_region": {"bbox_xyxy_source_px": bbox,
                              "mask_rle_row_major": args["mask"], "foreground_pixels": area,
                              "semantics": "full_visible_target_object_region_not_contact"},
            "preconditions": None, "failure_modes": None, "target_contact_point_2d": None,
        },
        "missing_source_fields": ["explicit_per_step_precondition_state", "failure_modes", "physical_contact_point"],
        "scope": "Reference expert continuation, not a unique plan from a single image; no generated CoT or physical execution claim.",
        "provenance": {
            "trajectory": row["trajectory"], "low_idx": row["low_idx"], "high_idx": row["high_idx"],
            "source_frame_member": member, "source_image_metadata": selected["metadata"],
            "temporal_phase": "before_action", "frame_position_within_low_step": 0,
            "frame_group_size": len(row["frames"]), "frame_count_mode": mode,
            "before_middle_after_counts": list(counts), "official_revision": REVISION,
            "temporal_evidence": UPSTREAM + "gen/game_states/game_state_base.py#L462-L756",
            "temporal_rule": "store_ll_action records the mask before execution; the first save_act_image BEFORE frame precedes env.step.",
        },
    }


def run(root, output, limit=0):
    require(limit >= 0, "limit must be nonnegative")
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "output directory already exists; use a new run directory")
    manifest_path = root / "manifest.json"
    materialization = json.loads(manifest_path.read_text())
    sources = {name: sha256(root / name) for name in ("interactions.jsonl", "images.jsonl")}
    for name, digest in sources.items():
        require(materialization["files"][name] == digest, f"source hash mismatch: {name}")
    image_rows = list(read_jsonl(root / "images.jsonl"))
    image_index = {image["member"]: image for image in image_rows}
    require(len(image_index) == len(image_rows) == materialization["frames"], "duplicate/missing indexed image")

    @lru_cache(maxsize=None)
    def load_trajectory(path, digest):
        require(sha256(path) == digest, "source trajectory hash mismatch")
        return json.loads(Path(path).read_text())

    counts, action_counts, seen, trajectories, records = Counter(), Counter(), set(), set(), []
    source_rows = list(read_jsonl(root / "interactions.jsonl"))
    require(len(source_rows) == materialization["interactions"], "interaction count mismatch")
    for row in source_rows[:limit or None]:
        trajectory = load_trajectory(row["trajectory"]["path"], row["trajectory"]["sha256"])
        record = convert_record(row, trajectory, image_index, root)
        require(record["id"] not in seen, "duplicate evaluation identity")
        seen.add(record["id"])
        trajectories.add((record["source_split"], record["trajectory_id"]))
        counts[record["source_split"]] += 1
        action_counts[record["reference"]["next_expert_api_action"]["action"]] += 1
        records.append(record)
    require(records, "no records produced")
    # Validate the entire selection before creating any output. Inputs are never changed.
    output.mkdir(parents=True)
    records_path = output / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "state": "source_native_validation_records_converted",
        "records": len(records), "trajectories": len(trajectories), "source_splits": dict(counts),
        "action_counts": dict(action_counts), "source_interaction_count": len(source_rows),
        "limit": limit, "source_hashes": sources, "source_manifest_sha256": sha256(manifest_path),
        "records_sha256": sha256(records_path), "converter_sha256": sha256(__file__),
        "official_revision": REVISION, "official_generator_sha256": GENERATOR,
        "python": platform.python_version(), "pillow": pillow_version, "command": sys.argv,
        "training_records": 0, "new_model_calls": 0, "human_review_performed": False,
        "all_selected_images_hash_checked": True, "all_trajectories_hash_checked": True,
        "temporal_phase": "first_before_action_frame",
        "native_schema_complete": False, "benchmark_scored": False,
        "restrictions": ["Preserve official validation splits exclusively for evaluation.",
                         "No per-step truth of preconditions, failure modes, or physical contact labels is supplied.",
                         "Expert continuation is a reference, not a uniquely inferable plan from one image.",
                         "This artifact alone does not establish complete eleven-form coverage or model accuracy."],
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return receipt


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def exact_train_trajectory(member):
    parts = PurePosixPath(member).parts
    return (len(parts) == 5 and parts[:2] == ("full_2.1.0", "train")
            and parts[-1] == "traj_data.json" and ".." not in parts)


class SevenZipReader:
    """Read an existing archive through system libarchive; never extract arbitrary paths."""

    def __init__(self, path):
        library = ctypes.util.find_library("archive")
        require(library is not None, "system libarchive is required; no download is performed")
        self.lib = ctypes.CDLL(library)
        signatures = {
            "archive_read_new": ([], ctypes.c_void_p),
            "archive_read_support_format_7zip": ([ctypes.c_void_p], ctypes.c_int),
            "archive_read_support_filter_all": ([ctypes.c_void_p], ctypes.c_int),
            "archive_read_free": ([ctypes.c_void_p], ctypes.c_int),
            "archive_read_open_filename": ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
            "archive_read_next_header": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "archive_entry_pathname": ([ctypes.c_void_p], ctypes.c_char_p),
            "archive_entry_size": ([ctypes.c_void_p], ctypes.c_longlong),
            "archive_entry_filetype": ([ctypes.c_void_p], ctypes.c_uint),
            "archive_error_string": ([ctypes.c_void_p], ctypes.c_char_p),
            "archive_read_data": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ssize_t),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.archive_read_new()
        self.lib.archive_read_support_filter_all(self.handle)
        self.lib.archive_read_support_format_7zip(self.handle)
        self.check(self.lib.archive_read_open_filename(self.handle, os.fsencode(path), 10240))

    def check(self, code):
        if code < 0:
            error = self.lib.archive_error_string(self.handle)
            raise RuntimeError(error.decode(errors="replace") if error else "libarchive error")
        return code

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.lib.archive_read_free(self.handle)

    def __iter__(self):
        while True:
            entry = ctypes.c_void_p()
            code = self.check(self.lib.archive_read_next_header(self.handle, ctypes.byref(entry)))
            if code == 1:
                return
            yield (self.lib.archive_entry_pathname(entry).decode(), self.lib.archive_entry_size(entry),
                   self.lib.archive_entry_filetype(entry))

    def copy_current(self, path, expected_size):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        digest, written = hashlib.sha256(), 0
        buffer = ctypes.create_string_buffer(1024 * 1024)
        with temporary.open("wb") as stream:
            while True:
                count = self.check(self.lib.archive_read_data(self.handle, buffer, len(buffer)))
                if not count:
                    break
                data = buffer.raw[:count]
                stream.write(data)
                digest.update(data)
                written += count
            require(written == expected_size, "archive member size mismatch")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return digest.hexdigest()


def selected_header_inventory(archive, select, expected_members=None):
    """List the complete archive without opening any member's compressed data.

    libarchive's 7zip list path avoids decoding file payloads until the first
    archive_read_data call. A separate, complete header pass therefore proves
    completeness/uniqueness without forcing extraction to decode an unused tail.
    """
    inventory = {}
    with SevenZipReader(archive) as reader:
        for member, size, filetype in reader:
            if not select(member):
                continue
            require(member not in inventory, "duplicate selected archive member")
            parts = PurePosixPath(member)
            require(not parts.is_absolute() and ".." not in parts.parts and str(parts) == member,
                    "unsafe archive member path")
            require(filetype == 0o100000 and size > 0, "selected member is not a regular nonempty file")
            inventory[member] = {"bytes": size, "filetype": filetype}
    if expected_members is not None:
        require(set(inventory) == set(expected_members), "expected selected members missing from archive or selector mismatch")
    return inventory


def extract_selected(archive, destination, journal_path, select, status, expected_members=None):
    """Verify a complete header inventory and resume only missing/corrupt targets.

    Journal entries alone never prove completion: every reused file is checked
    against its canonical path, header size/type and recorded SHA-256. Duplicate
    headers are rejected before reading payloads. Once the exact target set is
    available, close the data reader immediately rather than decoding its tail.
    """
    archive = Path(archive)
    before = archive.stat()
    binding = (before.st_size, before.st_mtime_ns)
    inventory = selected_header_inventory(archive, select, expected_members)
    current = archive.stat()
    require((current.st_size, current.st_mtime_ns) == binding, "archive changed during header inventory")
    inventory_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    status(inventory_members=len(inventory), inventory_sha256=inventory_sha, inventory_complete=True)
    done = {}
    if journal_path.exists():
        # Each line is appended only after atomic rename. Ignore a truncated last
        # line from an interrupted write; the corresponding file is re-extracted.
        lines = journal_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                require(index == len(lines) - 1, "corrupt extraction journal")
                break
            member = item["member"]
            require(member not in done, "duplicate extraction journal member")
            require(member in inventory, "journal member is absent from selected header inventory")
            require(isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in item["sha256"]), "invalid journal SHA-256")
            done[member] = item
    selected = {}
    for member, prior in done.items():
        path = destination / member
        require(path.resolve().is_relative_to(destination.resolve()) and not path.is_symlink(),
                "unsafe archive member path")
        require(prior.get("path") == str(path.resolve()), "journal target path mismatch")
        size = inventory[member]["bytes"]
        if (prior.get("bytes") == size and path.is_file() and path.stat().st_size == size
                and sha256(path) == prior["sha256"]):
            selected[member] = prior
    # Retain only verified entries. A corrupt/truncated entry must be genuinely
    # re-extracted and appended once, never silently accepted or duplicated.
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = journal_path.with_name(journal_path.name + ".rewrite")
    with normalized.open("w", encoding="utf-8") as stream:
        for item in selected.values():
            stream.write(json.dumps(item, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    normalized.replace(journal_path)
    pending = set(inventory) - set(selected)
    extracted, reused = 0, len(selected)
    if not pending:
        current = archive.stat()
        require((current.st_size, current.st_mtime_ns) == binding, "archive changed during journal verification")
        status(selected=len(selected), extracted=0, reused=reused, payload_scan_skipped=True)
        return [selected[member] for member in sorted(selected)]
    with SevenZipReader(archive) as reader, journal_path.open("a", encoding="utf-8") as journal:
        for member, size, filetype in reader:
            if member not in pending:
                continue
            require(inventory[member] == {"bytes": size, "filetype": filetype},
                    "selected member changed after header inventory")
            path = destination / member
            require(path.resolve().is_relative_to(destination.resolve()), "unsafe archive member path")
            require(not path.is_symlink(), "selected target is a symlink")
            digest = reader.copy_current(path, size)
            item = {"member": member, "path": str(path.resolve()), "bytes": size, "sha256": digest}
            journal.write(json.dumps(item, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            extracted += 1
            selected[member] = item
            pending.remove(member)
            if len(selected) % 100 == 0:
                status(selected=len(selected), extracted=extracted, reused=reused)
            if not pending:
                break
    current = archive.stat()
    require((current.st_size, current.st_mtime_ns) == binding, "archive changed during selected extraction")
    require(not pending and set(selected) == set(inventory), "selected members missing from data reader")
    status(selected=len(selected), extracted=extracted, reused=reused)
    return [selected[member] for member in sorted(selected)]


def load_heldout_scenes(records_path):
    scenes = set()
    manifest = records_path.parent / "manifest.json"
    require(manifest.is_file(), "held-out conversion manifest is required")
    require(json.loads(manifest.read_text())["records_sha256"] == sha256(records_path), "held-out record hash mismatch")
    for record in read_jsonl(records_path):
        require(record["source"] == "ALFRED" and record["source_split"] in ALLOWED_SPLITS
                and record["training_approved"] is False, "invalid held-out record partition")
        scene = record["scene_group"]
        require(scene.startswith("FloorPlan") and scene[9:].isdigit(), "invalid held-out physical scene")
        scenes.add(scene)
    require(scenes, "empty held-out scene set")
    return scenes


def run_train(archive, heldout_records, output):
    """Two-stage resumable extraction: official metadata, then scene-disjoint BEFORE images."""
    archive, heldout_records, output = map(lambda p: Path(p).resolve(), (archive, heldout_records, output))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        return _run_train_locked(archive, heldout_records, output)


def _run_train_locked(archive, heldout_records, output):
    stat = archive.stat()
    binding = {"archive": {"path": str(archive), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
               "heldout_records": str(heldout_records), "heldout_records_sha256": sha256(heldout_records)}
    config_path = output / "input_binding.json"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == binding, "resume inputs changed")
    else:
        atomic_json(config_path, binding)
    records_path = output.parent / "train_records.jsonl"
    final_manifest = output / "manifest.json"
    if final_manifest.exists():
        receipt = json.loads(final_manifest.read_text())
        require(sha256(records_path) == receipt["records_sha256"], "completed training records changed")
        return receipt
    heldout = load_heldout_scenes(heldout_records)
    start = time.monotonic()
    phase = "metadata_extraction"

    def status(**progress):
        update = {"phase": phase, "pid": os.getpid(), "elapsed_seconds": round(time.monotonic() - start, 2),
                  "converter_sha256": sha256(__file__), **progress}
        atomic_json(output / "status.json", update)
        print(json.dumps(update, sort_keys=True), flush=True)

    try:
        status()
        metadata = extract_selected(archive, output / "source", output / "metadata_journal.jsonl",
                                    exact_train_trajectory, status)
        require(metadata, "archive contains no exact-prefix official training trajectories")
        phase = "physical_scene_selection"
        status(metadata_trajectories=len(metadata))
        scene_counts, exclusions, trajectories, rows, rejects = Counter(), [], {}, [], []
        for item in sorted(metadata, key=lambda x: x["member"]):
            trajectory = json.loads(Path(item["path"]).read_text())
            scene = "FloorPlan" + str(trajectory["scene"]["scene_num"])
            require(scene == trajectory["scene"]["floor_plan"], "scene_num/floor_plan mismatch")
            parts = PurePosixPath(item["member"]).parts
            require(parts[2].rsplit("-", 1)[-1] == str(trajectory["scene"]["scene_num"])
                    and parts[3] == trajectory["task_id"], "archive/trajectory scene or task mismatch")
            scene_counts[scene] += 1
            if scene in heldout:
                exclusions.append({"member": item["member"], "scene": scene,
                                   "trajectory_sha256": item["sha256"], "reason": "heldout_physical_scene"})
                continue
            trajectories[item["path"]] = trajectory
            frame_groups = {}
            for frame in trajectory["images"]:
                frame_groups.setdefault(frame["low_idx"], []).append(frame)
            for low_idx, action in enumerate(trajectory["plan"]["low_actions"]):
                if action["api_action"]["action"] not in FRAME_COUNTS or action["api_action"]["action"] == "CoolObject":
                    continue
                row = {"task_id": trajectory["task_id"], "scene_id": scene, "split": "train",
                       "low_idx": low_idx, "high_idx": action["high_idx"], "source_action": action,
                       "trajectory": {"path": item["path"], "sha256": item["sha256"]},
                       "frames": [{"member": str(PurePosixPath(item["member"]).parent / "raw_images" /
                                                 Path(frame["image_name"]).with_suffix(".jpg").name),
                                   "metadata": frame} for frame in frame_groups.get(low_idx, [])]}
                try:
                    select_preaction(row, trajectory)
                except (ValueError, KeyError, IndexError) as error:
                    rejects.append({"trajectory": item["member"], "low_idx": low_idx,
                                    "stage": "temporal_alignment", "reason": str(error)})
                    continue
                rows.append(row)
        selected_scenes = set(scene_counts) - heldout
        selected_members = {row["frames"][0]["member"] for row in rows}
        selection = {"heldout_scenes": sorted(heldout), "training_scenes": sorted(selected_scenes),
                     "all_training_scene_trajectory_counts": dict(sorted(scene_counts.items())),
                     "metadata_trajectories": len(metadata), "eligible_trajectories": len(trajectories),
                     "excluded_trajectories": len(exclusions), "excluded": exclusions,
                     "candidate_interactions": len(rows), "selected_first_before_images": len(selected_members),
                     "temporal_rejects": rejects, "physical_scene_overlap": sorted(selected_scenes & heldout)}
        require(rows and not selection["physical_scene_overlap"], "no scene-disjoint interactions available")
        atomic_json(output / "scene_selection.json", selection)
        phase = "preaction_image_extraction"
        status(eligible_trajectories=len(trajectories), excluded_trajectories=len(exclusions),
               selected_scenes=len(selected_scenes), selected_images=len(selected_members))
        images = extract_selected(archive, output / "frames", output / "images_journal.jsonl",
                                  selected_members.__contains__, status, expected_members=selected_members)
        image_index = {item["member"]: {**item, "size": [300, 300]} for item in images}
        require(set(image_index) == selected_members, "selected pre-action images missing from archive")
        phase = "conversion_and_verification"
        status(selected_images=len(images))
        counts, produced_ids, produced_trajectories = Counter(), set(), set()
        temporary = records_path.with_name(records_path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                try:
                    record = convert_record(row, trajectories[row["trajectory"]["path"]], image_index,
                                            output, training_scenes=selected_scenes)
                except (ValueError, KeyError, IndexError) as error:
                    rejects.append({"trajectory": row["trajectory"]["path"], "low_idx": row["low_idx"],
                                    "stage": "native_conversion", "reason": str(error)})
                    continue
                require(record["id"] not in produced_ids, "duplicate training interaction identity")
                produced_ids.add(record["id"])
                produced_trajectories.add(record["trajectory_id"])
                counts[record["reference"]["next_expert_api_action"]["action"]] += 1
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(produced_ids, "no source-native training records passed conversion")
        if records_path.exists():
            require(sha256(records_path) == sha256(temporary), "existing final training records differ; do not overwrite")
            temporary.unlink()
        else:
            temporary.replace(records_path)
        atomic_json(output / "rejections.json", rejects)
        receipt = {"state": "scene_disjoint_source_native_training_records_converted", **binding,
                   "records": len(produced_ids), "trajectories": len(produced_trajectories),
                   "records_path": str(records_path), "records_sha256": sha256(records_path),
                   "action_counts": dict(counts), "training_scenes": sorted(selected_scenes),
                   "heldout_scenes": sorted(heldout), "physical_scene_overlap": [],
                   "excluded_trajectories": len(exclusions), "rejected_interactions": len(rejects),
                   "metadata_trajectories": len(metadata), "eligible_trajectories": len(trajectories),
                   "extracted_metadata_bytes": sum(item["bytes"] for item in metadata),
                   "extracted_preaction_image_bytes": sum(item["bytes"] for item in images),
                   "selected_images": len(images), "scene_selection_sha256": sha256(output / "scene_selection.json"),
                   "metadata_journal_sha256": sha256(output / "metadata_journal.jsonl"),
                   "images_journal_sha256": sha256(output / "images_journal.jsonl"),
                   "rejections_sha256": sha256(output / "rejections.json"), "converter_sha256": sha256(__file__),
                   "official_revision": REVISION, "official_generator_sha256": GENERATOR,
                   "elapsed_seconds": round(time.monotonic() - start, 2),
                   "python": platform.python_version(), "pillow": pillow_version, "command": sys.argv,
                   "new_model_calls": 0, "human_review_performed": False, "native_schema_complete": False,
                   "physical_contact_labels_supplied": False, "model_training_performed": False}
        atomic_json(final_manifest, receipt)
        phase = "complete"
        status(records=len(produced_ids), trajectories=len(produced_trajectories))
        return receipt
    except Exception as error:
        status(error_type=type(error).__name__, error=str(error), resumable=True)
        raise


def _sft_bound_records(records_path, manifest_path, partition):
    """Bind the complete source split; never silently sample or infer an approval."""
    records_path, manifest_path = Path(records_path).resolve(), Path(manifest_path).resolve()
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(records_path)
    require(manifest["records_sha256"] == digest, "SFT source records hash mismatch")
    expected_state = ("scene_disjoint_source_native_training_records_converted" if partition == "train"
                      else "source_native_validation_records_converted")
    require(manifest["state"] == expected_state, "wrong source manifest split/state")
    if "records_path" in manifest:
        require(Path(manifest["records_path"]).resolve() == records_path, "source manifest records path mismatch")
    rows = list(read_jsonl(records_path))
    require(rows and len(rows) == manifest["records"], "source record count mismatch")
    require(len({row["id"] for row in rows}) == len(rows), "duplicate source record identity")
    binding = {"records_path": str(records_path), "records_sha256": digest,
               "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash}
    return rows, manifest, binding


def _sft_record(record, partition, binding, load_trajectory, check_image):
    """Only source semantic labels enter the answer; raw simulator identities stay in metadata."""
    require(record["schema_version"] == "anvil-alfred-source-native-action-v1"
            and record["contract"] == "expert_plan_continuation_with_preaction_observation"
            and record["source"] == "ALFRED" and record["native_schema_complete"] is False,
            "unexpected source-native contract")
    split = record["source_split"]
    is_train = partition == "train"
    require((split == "train" if is_train else split in ALLOWED_SPLITS)
            and record["training_approved"] is is_train
            and record["partition"] == ("training_source_only" if is_train else "evaluation_source_only"),
            "heldout/training source partition mismatch")
    provenance, reference, inputs = record["provenance"], record["reference"], record["input"]
    require(provenance["official_revision"] == REVISION and provenance["temporal_phase"] == "before_action"
            and provenance["frame_position_within_low_step"] == 0, "not a pinned first-before observation")
    trajectory = load_trajectory(provenance["trajectory"]["path"], provenance["trajectory"]["sha256"])
    task, low_idx, high_idx = trajectory["task_id"], provenance["low_idx"], provenance["high_idx"]
    require(record["trajectory_id"] == task and record["scene_group"] == "FloorPlan" + str(trajectory["scene"]["scene_num"])
            and record["id"] == f"alfred:{split}:{task}:low{low_idx}", "source scene/task/step identity mismatch")
    require(type(low_idx) is int and 0 <= low_idx < len(trajectory["plan"]["low_actions"]), "invalid low step")
    member = PurePosixPath(provenance["source_frame_member"])
    require(len(member.parts) == 6 and member.parts[:2] == ("full_2.1.0", split)
            and member.parts[3:5] == (task, "raw_images") and ".." not in member.parts,
            "source frame member identity mismatch")
    group = [frame for frame in trajectory["images"] if frame["low_idx"] == low_idx]
    action = trajectory["plan"]["low_actions"][low_idx]
    original_row = {"low_idx": low_idx, "high_idx": high_idx, "source_action": action,
                    "frames": [{"metadata": frame, "member": str(member.parent / Path(frame["image_name"]).with_suffix(".jpg").name)}
                               for frame in group]}
    selected, counts, mode = select_preaction(original_row, trajectory)
    require(selected["member"] == str(member) and selected["metadata"] == provenance["source_image_metadata"]
            and list(counts) == provenance["before_middle_after_counts"]
            and mode == provenance["frame_count_mode"] and len(group) == provenance["frame_group_size"],
            "source temporal observation mismatch")
    image_path = Path(inputs["image"]).resolve()
    require(tuple(image_path.parts[-6:]) == member.parts, "image path does not identify source frame")
    check_image(str(image_path), inputs["image_sha256"], tuple(inputs["image_size"]))
    annotation = trajectory["turk_annotations"]["anns"][0]
    require(isinstance(inputs["goal"], str) and inputs["goal"].strip()
            and inputs["goal"] == annotation["task_desc"], "source goal mismatch")
    api = action["api_action"]
    require(api == reference["next_expert_api_action"], "next expert action differs from source")
    target_key = "receptacleObjectId" if api["action"] == "PutObject" else "objectId"
    target_id = api[target_key]
    role = "destination_receptacle" if target_key == "receptacleObjectId" else "interaction_object"
    require(reference["target_object_id"] == target_id and reference["target_object_type"] == target_id.split("|", 1)[0]
            and reference["target_role"] == role, "target identity/type differs from source")
    expected_steps = [{"high_idx": step["high_idx"], "discrete_action": step["discrete_action"],
                       "instruction": annotation["high_descs"][step["high_idx"]]
                       if step["high_idx"] < len(annotation["high_descs"]) else None}
                      for step in trajectory["plan"]["high_pddl"][high_idx:]
                      if step["planner_action"]["action"] != "End"]
    require(expected_steps and reference["remaining_high_level_steps"] == expected_steps,
            "missing or changed remaining source steps")
    steps = [step["discrete_action"] for step in expected_steps]
    for step in steps:
        require(set(step) == {"action", "args"} and isinstance(step["action"], str) and step["action"]
                and isinstance(step["args"], list) and all(isinstance(arg, str) and "|" not in arg for arg in step["args"]),
                "remaining actions must use source semantic arguments, not simulator identities")
    require(all(reference[key] is None for key in ("preconditions", "failure_modes", "target_contact_point_2d")),
            "unsupported action labels were added")
    target = {"target_object_type": reference["target_object_type"], "target_role": role,
              "next_expert_action": api["action"], "remaining_high_level_actions": steps}
    return {"id": record["id"], "images": [str(image_path)], "messages": [
        {"role": "user", "content": "<image>\n" + SFT_INSTRUCTION + inputs["goal"]},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True, allow_nan=False)}],
        "anvil_meta": {"contract": SFT_CONTRACT, "native_contract": SFT_CONTRACT,
                       "source_native_contract": record["schema_version"], "native_form": "action_cot", "family": "A",
                       "native_form_candidate": "action_cot", "native_schema_complete": False,
                       "source": "ALFRED", "source_record_id": record["id"], "source_split": split,
                       "partition": partition, "split": partition, "training_approved": is_train,
                       "scene_group": record["scene_group"], "group_id": "ALFRED:" + record["scene_group"], "trajectory_id": task,
                       "image_sha256": inputs["image_sha256"], "source_binding": binding,
                       "source_record_canonical_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                       "source_target_object_id": target_id, "provenance": provenance,
                       "verification": "Pinned source records, trajectory bytes, before-image bytes/dimensions, semantic targets and fixed physical-scene split checked; not model or human adjudication.",
                       "source_empty_argument_count": sum(arg == "" for step in steps for arg in step["args"]),
                       "rights": "local_only", "redistribution_status": "not_verified",
                       "scope": "One source expert continuation, not a unique optimal or guaranteed executable plan.",
                       "human_review_performed": False, "synthetic_cot_generated": False}}


def export_sft(train_records, train_manifest, evaluation_records, evaluation_manifest, output, contract):
    """Export fixed source-native train/evaluation splits without mutating source artifacts."""
    require(contract == SFT_CONTRACT, "explicit supported SFT contract required")
    output = Path(output).resolve()
    require(not output.exists(), "output directory already exists; use a new run directory")
    train, tm, tb = _sft_bound_records(train_records, train_manifest, "train")
    evaluation, em, eb = _sft_bound_records(evaluation_records, evaluation_manifest, "evaluation")
    require(Path(tm["heldout_records"]).resolve() == Path(eb["records_path"])
            and tm["heldout_records_sha256"] == eb["records_sha256"], "training exclusions bind a different heldout source")
    train_scenes, eval_scenes = ({r["scene_group"] for r in rows} for rows in (train, evaluation))
    require(train_scenes == set(tm["training_scenes"]) and eval_scenes == set(tm["heldout_scenes"]),
            "source scene inventory mismatch")
    require(not train_scenes & eval_scenes and tm["physical_scene_overlap"] == [], "heldout physical scene enters train")
    require(dict(Counter(r["source_split"] for r in evaluation)) == em["source_splits"], "evaluation split inventory mismatch")
    require(not {r["id"] for r in train} & {r["id"] for r in evaluation}, "cross-split record identity overlap")
    require(not {r["input"]["image_sha256"] for r in train} & {r["input"]["image_sha256"] for r in evaluation},
            "heldout image enters train")

    @lru_cache(maxsize=None)
    def load_trajectory(path, digest):
        require(sha256(path) == digest, "source trajectory hash mismatch")
        return json.loads(Path(path).read_text())

    @lru_cache(maxsize=None)
    def check_image(path, digest, size):
        require(size == (300, 300), "source image dimensions changed")
        require(sha256(path) == digest, "source image hash mismatch")
        with Image.open(path) as image:
            image.load()
            require(image.size == size, "source image dimensions mismatch")

    exported = {"train": [_sft_record(r, "train", tb, load_trajectory, check_image) for r in train],
                "evaluation": [_sft_record(r, "evaluation", eb, load_trajectory, check_image) for r in evaluation]}
    for binding in (tb, eb):
        require(sha256(binding["records_path"]) == binding["records_sha256"]
                and sha256(binding["manifest_path"]) == binding["manifest_sha256"], "source binding drift during export")
    output.mkdir(parents=True)
    files, counts = {}, {}
    for partition, rows in exported.items():
        path = output / f"{partition}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files[path.name] = sha256(path)
        counts[partition] = {"records": len(rows), "scenes": len({r["anvil_meta"]["scene_group"] for r in rows}),
                             "image_paths": len({r["images"][0] for r in rows}),
                             "unique_image_sha256": len({r["anvil_meta"]["image_sha256"] for r in rows}),
                             "rows_with_source_empty_arguments": sum(r["anvil_meta"]["source_empty_argument_count"] > 0 for r in rows)}
    receipt = {"state": "source_native_expert_continuation_sft_exported", "contract": SFT_CONTRACT,
               "files": files, "counts": counts, "source_binding": {"train": tb, "evaluation": eb},
               "training_scenes": sorted(train_scenes), "heldout_scenes": sorted(eval_scenes),
               "physical_scene_overlap": [], "image_sha_overlap": [], "all_images_hash_and_decode_verified": True,
               "all_source_trajectories_hash_verified": True, "converter_sha256": sha256(__file__),
               "official_revision": REVISION, "python": platform.python_version(), "pillow": pillow_version,
               "command": sys.argv, "rights": "local_only", "redistribution_status": "not_verified",
               "native_schema_complete": False, "model_training_performed": False,
               "human_review_performed": False, "new_model_calls": 0,
               "scope": "Single source expert continuation; no unique optimality, execution success, or complete Action-CoT claim."}
    atomic_json(output / "manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--interactions-root", type=Path)
    source.add_argument("--train-archive", type=Path, help="existing official full_2.1.0 archive; no download")
    source.add_argument("--sft-train-records", type=Path, help="export the complete pre-bound train and heldout sources")
    parser.add_argument("--sft-train-manifest", type=Path)
    parser.add_argument("--sft-evaluation-records", type=Path)
    parser.add_argument("--sft-evaluation-manifest", type=Path)
    parser.add_argument("--sft-contract", choices=[SFT_CONTRACT])
    parser.add_argument("--heldout-records", type=Path, help="required for scene-disjoint official train extraction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 converts all source interactions")
    args = parser.parse_args()
    if args.sft_train_records:
        if (not all((args.sft_train_manifest, args.sft_evaluation_records, args.sft_evaluation_manifest, args.sft_contract))
                or args.limit or args.heldout_records):
            parser.error("SFT export requires both manifests, evaluation records, and explicit --sft-contract; no sample limit")
        result = export_sft(args.sft_train_records, args.sft_train_manifest, args.sft_evaluation_records,
                            args.sft_evaluation_manifest, args.output, args.sft_contract)
    elif args.train_archive:
        if not args.heldout_records or args.limit:
            parser.error("--train-archive requires --heldout-records and does not accept a sample limit")
        result = run_train(args.train_archive, args.heldout_records, args.output)
    else:
        result = run(args.interactions_root, args.output, args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
