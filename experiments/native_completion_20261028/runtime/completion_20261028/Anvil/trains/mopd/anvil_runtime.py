"""Fresh-rollout token layouts and routing for the packaged Anvil contract.

The semantic evaluator is supplied explicitly and runs on the current sampled
payload. Missing semantic evidence abstains; it is never treated as a pass.
This module does not generate targets or qualify teachers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from trains.flywheel.validators import validate_structural


FAMILIES = ("G", "S", "A")
TAGS = {"G": "GROUND", "S": "RELATE", "A": "ACT"}
FORM_ALIASES = {"depth": "relative_depth", "dof": "gripper_dof", "action_cot": "action_plan"}
NATIVE_FORMS = {value: key for key, value in FORM_ALIASES.items()}
FIELD = re.compile(r'"(?:[^"\\]|\\.)*"\s*:')
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
ACTION = re.compile(r"\b(?:grasp|pinch|pull|turn|lift|open|close|push|release|place|approach|move|rotate)\b", re.I)
COORDINATES = {"bbox", "bbox_2d", "point_2d", "polygon", "contact_point_2d", "grasp_point_2d",
               "jaw_axis_2d", "target_point_2d", "wrist_deg", "rank", "relative_depth"}


def load_contract_module():
    path = Path(os.environ.get("ANVIL_CONTRACT_MODULE", str(
        Path(__file__).resolve().parents[2] / "engineering/experiments/contract_v2.py")))
    name = "_anvil_packaged_contract_v2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load Anvil contract: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _strict_json(text):
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(text, parse_constant=reject)


def decoded_token_spans(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Map actual sampled IDs without re-tokenizing their decoded string.

    Prefix decoding handles noncanonical BPE segmentations. Bytes completing one
    Unicode character can produce zero-width prefixes; those tokens attach to
    the character they complete rather than being silently discarded.
    """
    decode = lambda values: tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(ids)
    boundaries = [0]
    for stop in range(1, len(ids)):
        prefix = decode(ids[:stop])
        common = 0
        for left, right in zip(prefix, text):
            if left != right:
                break
            common += 1
        if common < boundaries[-1]:
            raise ValueError("token decoder rewrites an established text prefix")
        boundaries.append(common)
    boundaries.append(len(text))
    spans = []
    for index in range(len(ids)):
        start, end = boundaries[index:index + 2]
        if start == end and end < len(text):
            end += 1
        spans.append((start, end))
    return text, spans


def parse_stage_segments(text: str, stages: list[str], *, stop_on_invalid_json: bool = True):
    """Return complete declared segments, then one malformed suffix if present."""
    segments, cursor = [], 0
    for stage in stages:
        opening, closing = f"<{TAGS[stage]}>", f"</{TAGS[stage]}>"
        start = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(opening, cursor):
            segments.append((stage, start, len(text), None, "missing_or_out_of_order_delimiter"))
            return segments
        body_start = cursor + len(opening)
        close_at = text.find(closing, body_start)
        next_tag = re.search(r"</?(?:GROUND|RELATE|ACT)>", text[body_start:])
        if close_at < 0 or (next_tag is not None and body_start + next_tag.start() != close_at):
            segments.append((stage, start, len(text), None, "missing_repeated_or_out_of_order_delimiter"))
            return segments
        cursor = close_at + len(closing)
        try:
            payload = _strict_json(text[body_start:close_at])
        except (ValueError, json.JSONDecodeError):
            segments.append((stage, start, cursor, None, "invalid_json"))
            if stop_on_invalid_json:
                return segments
            continue
        segments.append((stage, start, cursor, payload, None))
    if text[cursor:].strip():
        # A repeated stage or other non-EOS trailing output is structural failure
        # at the final declared stage; a response cannot add an undeclared task.
        stage, start, _, _, _ = segments[-1]
        segments[-1] = (stage, start, len(text), None, "unexpected_trailing_output")
    return segments


def structural_ranges(text: str, *, coordinate_fields=COORDINATES,
                      literal_fields=frozenset({"predicate", "relation", "approach", "affordance"})) -> list[tuple[int, int]]:
    result = [(match.start(), match.end()) for match in re.finditer(r"<(?:GROUND|RELATE|ACT)>", text)]
    decoder = json.JSONDecoder()
    for match in FIELD.finditer(text):
        end = text.rfind('"', match.start(), match.end()) + 1
        result.append((match.start(), end))
        try:
            key = json.loads(text[match.start():end])
            start = match.end()
            while start < len(text) and text[start].isspace():
                start += 1
            _, length = decoder.raw_decode(text[start:])
        except (ValueError, json.JSONDecodeError):
            continue
        value = text[start:start + length]
        pattern = NUMBER if key in coordinate_fields else ACTION if key in {"goal", "steps"} else None
        if pattern is not None:
            result.extend((start + m.start(), start + m.end()) for m in pattern.finditer(value))
        elif key in literal_fields:
            result.append((start, start + length))
    return result


@dataclass
class AnvilTokenPlan:
    token_ids: Tensor
    stage_ids: Tensor
    selected_stages: Tensor
    structural_mask: Tensor
    decisions: list[dict]


def validate_teacher_eligibility(path, adapters, student_adapter):
    """Apply the existing path-bound gate before allocating teacher models."""
    path = Path(path or "")
    if not path.is_file():
        raise ValueError("Anvil requires a path-bound teacher eligibility report")
    eligibility = json.loads(path.read_text())
    if os.environ.get('ANVIL_DATA_CONTRACT') == 'anvil-source-native-train-v1':
        # A legacy pass cannot qualify newly trained native-format teachers.
        # This binds a separately obtained quality decision; it never creates it.
        train_manifest = Path(os.environ.get('TRAIN', '')) / 'manifest.json'
        if (eligibility.get('data_contract') != 'anvil-source-native-train-v1'
                or not train_manifest.is_file()
                or eligibility.get('training_export_manifest_sha256') != hashlib.sha256(train_manifest.read_bytes()).hexdigest()):
            raise ValueError('Native MTD requires a matching native training-export eligibility binding')
        for family, adapter in adapters.items():
            weights = Path(adapter) / 'adapter_model.safetensors'
            if (not weights.is_file() or (eligibility.get('teachers', {}).get(family) or {}).get('adapter_sha256')
                    != hashlib.sha256(weights.read_bytes()).hexdigest()):
                raise ValueError(f'Native {family} teacher bytes are not covered by eligibility')
        weights = Path(student_adapter or '') / 'adapter_model.safetensors'
        if not weights.is_file() or eligibility.get('joint_adapter_sha256') != hashlib.sha256(weights.read_bytes()).hexdigest():
            raise ValueError('Native initial student bytes are not covered by eligibility')
    if os.environ.get('ANVIL_EXPERIMENT_SCOPE') == 'GS_MECHANISM':
        # A separately declared mechanism experiment measures teacher quality;
        # it does not assert the original full-G/S/A superiority qualification.
        if eligibility.get('schema')!='anvil-gs-mechanism-binding-v1' or eligibility.get('full_GSA_qualified') is not False:
            raise ValueError('GS study requires an explicit scoped evidence binding')
        if set(adapters)!={'G','S'}:
            raise ValueError('GS study must not load or qualify an Action teacher')
        def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for family in ('G','S'):
            item=eligibility['teachers'][family]
            if Path(item['adapter']).resolve()!=Path(adapters[family]).resolve() or digest(Path(adapters[family])/'adapter_model.safetensors')!=item['sha256']:
                raise ValueError('GS teacher identity changed')
        initial=eligibility['student']
        if not student_adapter or Path(initial['adapter']).resolve()!=Path(student_adapter).resolve() or digest(Path(student_adapter)/'adapter_model.safetensors')!=initial['sha256']:
            raise ValueError('GS initial student identity changed')
        for name in ('training','context_evidence'):
            item=eligibility[name]
            if digest(item['path'])!=item['sha256']:raise ValueError('GS study evidence changed')
        if Path(os.environ.get('TRAIN','')).resolve()!=Path(eligibility['training']['path']).resolve():
            raise ValueError('GS study binding does not cover selected training source')
        evidence=json.loads(Path(eligibility['context_evidence']['path']).read_text())
        if not all(evidence[f][c]['state']=='complete' for f in ('S','joint') for c in ('student','shifted')):
            raise ValueError('GS context comparison is incomplete')
        return {'G':True,'S':True,'A':False}
    if str(eligibility.get("verdict")).upper() not in {"PASS", "APPROVED", "ELIGIBLE"}:
        raise ValueError("teacher eligibility verdict is not passing")
    qualified = {}
    for family in FAMILIES:
        item = (eligibility.get("teachers") or {}).get(family) or {}
        if (item.get("eligible") is not True or family not in adapters
                or Path(item.get("adapter", "")).resolve() != Path(adapters[family]).resolve()):
            raise ValueError(f"eligibility report does not cover selected {family} adapter")
        qualified[family] = True
    if not student_adapter or Path(eligibility.get("joint_adapter", "")).resolve() != Path(student_adapter).resolve():
        raise ValueError("eligibility report does not cover the student initialization")
    return qualified


def reference_geometry_evaluator(*, row, stage, form, payload, upstream, reference):
    """Measure a fresh stage against externally frozen, image-bound references.

    G uses label-aware IoU/F1/PCK; S uses mean pairwise depth order. The caller
    supplies native ``gsa_stage_contexts`` aligned with ``gsa_stages``. This
    callable does not qualify labels, calibrate thresholds or certify Teachers.
    Missing references/context abstain. A geometry additionally needs an explicit
    complete alternative-reference set; action-plan meaning always abstains here.
    """
    from eval.g_family_qualification import _label, _score_row, native_action_reference_scores
    from trains.flywheel.validators import validate

    canonical = FORM_ALIASES.get(form, form)
    native = NATIVE_FORMS.get(canonical, canonical)
    contract = load_contract_module()
    if contract.stage_for_form(canonical).value != stage:
        raise ValueError("semantic scorer stage/form mismatch")
    stages = row.get("gsa_stages") or []
    if stage not in stages:
        raise ValueError("semantic scorer stage is not declared")
    index = stages.index(stage)
    contexts = row.get("gsa_stage_contexts")
    if not isinstance(contexts, list) or len(contexts) != len(stages) or not isinstance(contexts[index], dict):
        return None
    expected_upstream = {edge["upstream"] for edge in row.get("gsa_edges", []) if edge["downstream"] == stage}
    if set(upstream) != expected_upstream:
        raise ValueError("semantic scorer did not receive exact passed direct predecessors")
    if not isinstance(reference, list) or not reference:
        return None
    context = contexts[index]
    reference_valid, _ = validate(native, reference, context)
    if not reference_valid:
        raise ValueError("semantic reference violates its frozen stage context")
    if native == "action_cot":
        return None  # Geometry cannot establish the semantics of an action plan.
    if stage == "A":
        coverage = row.get("gsa_reference_coverage")
        alternatives = row.get("gsa_reference_alternatives")
        if (not isinstance(coverage, list) or len(coverage) != len(stages)
                or coverage[index] != "complete_for_task" or not isinstance(alternatives, list)
                or len(alternatives) != len(stages) or not isinstance(alternatives[index], list)
                or not alternatives[index]):
            return None
        targets = alternatives[index]
        if reference not in targets or any(not validate(native, target, context)[0] for target in targets):
            raise ValueError("declared A alternative references are incomplete or invalid")
    else:
        targets = [reference]
    if stage == "S":
        gt = {_label(item["label"]): item["rank"] for item in reference}
        groundings = upstream.get("G")
        if groundings is None:
            return None
        names = [_label(item.get("label")) for item in groundings]
        if len(gt) != len(reference):
            raise ValueError("ambiguous entity identities in depth reference")
        if len(set(names)) != len(names) or set(names) != set(gt):
            # Establish reference applicability before judging the sampled S
            # payload. Otherwise an output over a reduced G entity set would be
            # incorrectly called a determinate S failure against oracle labels.
            return None
    valid, _ = validate(native, payload, context)
    if not valid:
        return 0.0
    if stage == "G":
        scored = _score_row({"gsa_form": native, "gsa_payload": reference, "gsa_context": context,
                             "sample_id": row.get("sample_id", row.get("canonical_uid", "linked-stage"))},
                            json.dumps(payload, allow_nan=False))
        metric = {"bbox": "bbox_iou", "detection": "primary", "mask": "mask_iou", "point": "hit_at_0_1"}[native]
        return float(scored[metric]) * (float(scored.get("label_match", 1)) if native != "detection" else 1.0)
    if stage == "S":
        gt = {_label(item["label"]): item["rank"] for item in reference}
        pred = {_label(item["label"]): item["rank"] for item in payload}
        if len(gt) != len(reference) or len(pred) != len(payload):
            raise ValueError("ambiguous entity identities in depth reference or prediction")
        if set(pred) != set(gt):
            return 0.0
        keys = sorted(gt)
        pairs = [(keys[a], keys[b]) for a in range(len(keys)) for b in range(a + 1, len(keys)) if gt[keys[a]] != gt[keys[b]]]
        if not pairs:
            return None
        return sum((gt[a] < gt[b]) == (pred[a] < pred[b]) and pred[a] != pred[b] for a, b in pairs) / len(pairs)
    metric = {"affordance": "single_reference_affordance_score", "dof": "single_reference_dof_score"}[native]
    return max(native_action_reference_scores({"gsa_form": native, "gsa_payload": target}, payload,
                                              usable=True)["scores"][metric] for target in targets)


class ContractV2BatchRouter:
    """Paper routing over declared metadata and freshly decoded model outputs.

    A stage evaluator accepts keyword arguments ``row, stage, form, payload,
    upstream, reference`` and returns a normalized score or None. ``upstream``
    contains only already-passed sampled stages. Its implementation and routing
    thresholds must be frozen/calibrated outside this runtime adapter.
    """

    def __init__(self, stage_evaluator: Callable | None = None, *, selector: str = "semantic"):
        if selector not in {"semantic", "structural", "declared", "flat"}:
            raise ValueError(f"unknown Anvil selector: {selector}")
        self.contract = load_contract_module()
        self.stage_evaluator = stage_evaluator
        self.selector = selector

    def _record(self, row: dict):
        c = self.contract
        if row.get("gsa_contract_version") != "gsa-contract-v2":
            raise ValueError("Anvil requires explicit gsa-contract-v2 records")
        family, form = row.get("gsa_family"), row.get("gsa_form")
        canonical_form = FORM_ALIASES.get(form, form)
        if family not in FAMILIES or c.stage_for_form(canonical_form).value != family:
            raise ValueError("immutable gsa_form and gsa_family disagree")
        messages = row.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant" or not isinstance(messages[-1].get("content"), str):
            raise ValueError("Anvil SFT requires a final reference assistant string")
        if any(message.get("role") == "assistant" for message in messages[:-1]):
            raise ValueError("Anvil currently requires one final assistant response per record")
        reference_text = messages[-1]["content"]
        meta = row.get("gsa_meta") or {}
        image_hash = meta.get("image_sha256")
        if not isinstance(image_hash, str) or not re.fullmatch("[0-9a-f]{64}", image_hash):
            raise ValueError("Anvil record lacks a byte-bound image identity")
        if meta.get("split") != "train":
            raise ValueError("Anvil updates require an explicit train split; held-out or unknown rows are rejected")
        if type(row.get("gsa_route_eligible", True)) is not bool:
            raise ValueError("gsa_route_eligible must be a boolean")
        stages = row.get("gsa_stages")
        linked = stages is not None
        if not linked:
            stages, forms, edges, spans = [family], {c.Stage(family): canonical_form}, (), {}
        else:
            if not isinstance(stages, list) or len(stages) < 2 or len(set(stages)) != len(stages):
                raise ValueError("linked gsa_stages must declare at least two unique stages")
            native_forms = row.get("gsa_stage_forms")
            if not isinstance(native_forms, list) or len(native_forms) != len(stages):
                raise ValueError("linked gsa_stage_forms must align with declared stages")
            forms = {c.Stage(stage): FORM_ALIASES.get(item, item) for stage, item in zip(stages, native_forms)}
            if stages[-1] != family or forms[c.Stage(stages[-1])] != canonical_form:
                raise ValueError("linked terminal stage disagrees with immutable output form/family")
            edges = tuple(c.DependencyEdge(c.Stage(edge["upstream"]), c.Stage(edge["downstream"]), edge["consumed_field"])
                          for edge in row.get("gsa_edges", []))
            if not edges:
                raise ValueError("linked records require explicit consumed dependency edges")
            segments = parse_stage_segments(reference_text, stages)
            if len(segments) != len(stages) or any(item[4] for item in segments):
                raise ValueError("reference serialization does not contain complete declared stages")
            spans = {c.Stage(stage): c.StageSpan(start, end) for stage, start, end, _, _ in segments}
            # Persisted reference spans are character offsets. Sampled token offsets
            # are always computed anew and never copied from these reference spans.
            expected = [{"start": item[1], "end": item[2]} for item in segments]
            if row.get("gsa_stage_span_unit") != "reference_characters" or row.get("gsa_stage_spans") != expected:
                raise ValueError("reference stage spans must match their explicit character units")
            for key in ("gsa_references", "gsa_thresholds", "gsa_uncertainty_bands"):
                if not isinstance(row.get(key), list) or len(row[key]) != len(stages):
                    raise ValueError(f"linked {key} must cover declared stages")
            for index, stage in enumerate(stages):
                c.StageEvaluation(c.Stage(stage), True, None, row["gsa_thresholds"][index],
                                  row["gsa_uncertainty_bands"][index]).state()
        uid = row.get("canonical_uid") or hashlib.sha256((image_hash + "\0" + reference_text).encode()).hexdigest()
        record = c.ContractRecord(c.RecordType.LINKED if linked else c.RecordType.ATOMIC,
                                  uid, image_hash, canonical_form, row.get("entity_namespace", uid), reference_text,
                                  tuple(c.Stage(stage) for stage in stages), forms, spans,
                                  None if linked else canonical_form, edges, row.get("gsa_route_eligible", True))
        record.validate()
        if [stage.value for stage in record.topological_order()] != stages:
            raise ValueError("declared serialization is not in dependency topological order")
        return record

    @staticmethod
    def _response(tokenizer, inputs, row_index, reference_prefix=""):
        positions = (inputs["labels"][row_index] != -100).nonzero().flatten().tolist()
        ids = inputs["input_ids"][row_index, positions].tolist()
        # End-of-turn special tokens do not extend a JSON stage or create a fault.
        eos = getattr(tokenizer, "eos_token_id", None)
        terminal = set(eos if isinstance(eos, (list, tuple)) else [eos])
        end = next((index for index, token in enumerate(ids) if token in terminal), len(ids))
        if end < len(ids):
            tail = [token for token in ids[end + 1:] if token not in terminal]
            if tokenizer.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip():
                raise ValueError("unexpected non-whitespace response after the end-of-turn token")
            ids, positions = ids[:end], positions[:end]
        text, spans = decoded_token_spans(tokenizer, ids)
        # Swift's non-thinking template inserts an empty thinking prefix in the
        # reference labels, but places it in the prompt for fresh generation.
        # Remove only that exact configured reference prefix, never sampled text.
        if reference_prefix and text.startswith(reference_prefix):
            start = len(reference_prefix)
            retained = [(max(0, left - start), right - start, position)
                        for (left, right), position in zip(spans, positions) if right > start]
            text = text[start:]
            spans = [(left, right) for left, right, _ in retained]
            positions = [position for _, _, position in retained]
        return text, spans, positions

    def plan(self, rows, inputs, tokenizer, *, reference: bool, qualified: dict[str, bool], reference_prefix=""):
        c = self.contract
        token_ids = inputs["input_ids"][:, 1:].detach().clone()
        stage_ids = torch.full_like(token_ids, -1)
        selected = torch.full((len(rows),), -1, dtype=torch.long, device=token_ids.device)
        structural = torch.zeros_like(token_ids, dtype=torch.bool)
        decisions = []
        if len(rows) != inputs["input_ids"].shape[0] or inputs["labels"].shape != inputs["input_ids"].shape:
            raise ValueError("row/encoded-token batch mismatch")
        for index, row in enumerate(rows):
            record = self._record(row)
            stages = [stage.value for stage in record.topological_order()]
            text, offsets, positions = self._response(tokenizer, inputs, index,
                                                       reference_prefix if reference else "")
            if reference and text != record.serialized_text:
                raise ValueError("encoded SFT response differs from the declared reference serialization")
            linked = record.record_type is c.RecordType.LINKED
            segments = (parse_stage_segments(text, stages,
                        stop_on_invalid_json=reference or self.selector != "declared")
                        if linked else [(stages[0], 0, len(text), None, None)])
            evaluations, passed_payloads = {}, {}
            for stage, start, end, payload, error in segments:
                # A BPE token can straddle a delimiter boundary. The later stage
                # owns that token, consistently in references and sampled traces.
                for (left, right), position in zip(offsets, positions):
                    if left < end and right > start and position > 0:
                        stage_ids[index, position - 1] = FAMILIES.index(stage)
                if reference or not linked or self.selector in {"declared", "flat"}:
                    continue
                stage_enum = c.Stage(stage)
                form = record.stage_forms[stage_enum]
                native = NATIVE_FORMS.get(form, form)
                structural_ok, _ = validate_structural(native, payload) if error is None else (False, error)
                score = None
                if structural_ok and self.selector == "semantic" and self.stage_evaluator is not None:
                    upstream = {previous.value: passed_payloads[previous.value]
                                for previous in record.predecessors(stage_enum)}
                    score = self.stage_evaluator(row=row, stage=stage, form=form, payload=payload,
                                                 upstream=upstream, reference=row["gsa_references"][stages.index(stage)])
                evaluation = c.StageEvaluation(stage_enum, bool(structural_ok), score,
                                               row["gsa_thresholds"][stages.index(stage)],
                                               row["gsa_uncertainty_bands"][stages.index(stage)])
                evaluations[stage_enum] = evaluation
                if self.selector == "structural":
                    # Structural passes are not semantic passes. Keep scanning
                    # structure without calling or fabricating an evaluator.
                    if not structural_ok:
                        break
                    continue
                if evaluation.state() != "pass":
                    break
                passed_payloads[stage] = payload
            ranges = structural_ranges(text)
            for (left, right), position in zip(offsets, positions):
                if position > 0 and any(left < end and right > start for start, end in ranges):
                    structural[index, position - 1] = True
            if reference:
                if not bool((stage_ids[index] >= 0).any()):
                    raise ValueError("reference has no trainable stage tokens")
                continue
            if self.selector == "semantic" or not linked or not record.route_eligible:
                decision = c.route(record, evaluations)
            elif self.selector in {"declared", "flat"}:
                decision = c.RouteDecision("declared_control", c.Stage(row["gsa_family"]),
                    "resolve_rollout_span", "declared_output_family")
            else:
                failed = next((stage for stage in record.topological_order()
                               if stage in evaluations and not evaluations[stage].structural_valid), None)
                decision = (c.RouteDecision("structural_invalid", failed, "resolve_rollout_span",
                                            f"first_structural_failure:{failed.value}")
                            if failed is not None else c.RouteDecision("structural_all_pass", None,
                                            "skip_mtd_retain_sft", "no_parse_or_contract_failure"))
            if self.selector == "flat":
                # The flat control retains the same reference SFT objective but
                # distills the entire sampled response from the declared teacher.
                # Response positions come from fresh IDs, never reference spans.
                stage_ids[index].fill_(-1)
                for position in positions:
                    if position > 0:
                        stage_ids[index, position - 1] = FAMILIES.index(row["gsa_family"])
            if decision.selected_stage is not None:
                stage_number = FAMILIES.index(decision.selected_stage.value)
                chosen = (stage_ids[index] == stage_number).nonzero().flatten()
                span = (c.RolloutTokenSpan(int(chosen[0]), int(chosen[-1]) + 1) if len(chosen)
                        else c.RolloutTokenSpan(token_ids.shape[1], token_ids.shape[1]))
                decision = c.allocate_mtd(decision, span, token_ids.shape[1],
                                          teacher_qualified=qualified.get(decision.selected_stage.value))
                if decision.mtd_action == "apply_mtd":
                    selected[index] = stage_number
            decisions.append({"uid": record.canonical_uid, "rollout_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "selector": self.selector,
                              "supervision_span": "full_response" if self.selector == "flat" else "selected_stage",
                              "state": decision.state, "stage": decision.selected_stage.value if decision.selected_stage else None,
                              "mtd_action": decision.mtd_action, "reason": decision.reason})
        return AnvilTokenPlan(token_ids, stage_ids, selected, structural, decisions)


class AnvilBatchRouter:
    """Dispatch unchanged v2 linked tasks and source-native atomic records.

    Native records retain their own schema and references; they are never
    converted into linked tasks or claimed to have strong-model adjudication.
    Atomic routing uses the declared family, even for malformed fresh output.
    QA replay belongs to reference-only SFT preparation, not this spatial MTD
    distribution. Teacher qualification is still required by the trainer.
    """

    def __init__(self, stage_evaluator=None, *, selector="semantic"):
        self.legacy = ContractV2BatchRouter(stage_evaluator, selector=selector)
        self.selector = selector

    def _native_plan(self, rows, inputs, tokenizer, *, reference, qualified, reference_prefix):
        from trains.flywheel.native_training import VERSION, require, validate_training_record
        token_ids = inputs["input_ids"][:, 1:].detach().clone()
        stage_ids = torch.full_like(token_ids, -1)
        selected = torch.full((len(rows),), -1, dtype=torch.long, device=token_ids.device)
        structural = torch.zeros_like(token_ids, dtype=torch.bool)
        decisions = []
        native_coordinates = COORDINATES | {"polygons", "region_bbox_2d", "value", "camera_from_gripper",
            "contact_point_camera", "contact_point_px", "approach_axis_camera", "jaw_axis_camera"}
        native_literals = {"predicate", "relation", "approach", "affordance", "next_expert_action", "action"}
        for index, row in enumerate(rows):
            require(row.get("gsa_contract_version") == VERSION,
                    "Native MTD requires spatial records; QA remains reference-only SFT replay")
            validate_training_record(row)
            text, offsets, positions = ContractV2BatchRouter._response(
                tokenizer, inputs, index, reference_prefix if reference else "")
            if reference:
                require(text == row["messages"][-1]["content"],
                        "Encoded native reference differs from source serialization")
            family = row["gsa_family"]; stage = FAMILIES.index(family)
            ranges = structural_ranges(text, coordinate_fields=native_coordinates, literal_fields=native_literals)
            for (left, right), position in zip(offsets, positions):
                require(position > 0 and bool(inputs["attention_mask"][index, position])
                        and inputs["labels"][index, position] == inputs["input_ids"][index, position],
                        "Native response labels are not attended next-token targets")
                stage_ids[index, position - 1] = stage
                if any(left < end and right > start for start, end in ranges):
                    structural[index, position - 1] = True
            nonempty = bool((stage_ids[index] == stage).any())
            if reference:
                require(nonempty, "Native reference has no trainable response tokens")
                continue
            eligible = nonempty and qualified.get(family) is True
            if eligible: selected[index] = stage
            decisions.append({"uid": row["canonical_uid"], "contract": VERSION,
                "rollout_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "selector": self.selector, "supervision_span": "atomic_response", "state": "atomic",
                "stage": family, "mtd_action": "apply_mtd" if eligible else "skip_mtd_retain_sft",
                "reason": "declared_native_family" if eligible else
                    "empty_rollout_span" if not nonempty else "teacher_not_qualified"})
        return AnvilTokenPlan(token_ids, stage_ids, selected, structural, decisions)

    def plan(self, rows, inputs, tokenizer, *, reference, qualified, reference_prefix=""):
        from trains.flywheel.native_training import VERSION, require
        require(bool(rows) and len(rows) == inputs["input_ids"].shape[0]
                and inputs["input_ids"].ndim == 2
                and inputs["labels"].shape == inputs["input_ids"].shape
                and inputs["attention_mask"].shape == inputs["input_ids"].shape,
                "Row/encoded-token batch mismatch")
        partitions = {"gsa-contract-v2": [], VERSION: []}
        for index, row in enumerate(rows):
            schema = row.get("gsa_contract_version")
            require(schema in partitions,
                    "Unsupported spatial MTD schema; QA replay remains in reference-only SFT preparation")
            partitions[schema].append(index)
        token_ids = inputs["input_ids"][:, 1:].detach().clone()
        stage_ids = torch.full_like(token_ids, -1)
        selected = torch.full((len(rows),), -1, dtype=torch.long, device=token_ids.device)
        structural = torch.zeros_like(token_ids, dtype=torch.bool)
        decisions = [None] * len(rows) if not reference else []
        for schema, indices in partitions.items():
            if not indices: continue
            tensor_indices = torch.tensor(indices, device=token_ids.device)
            batch = {key: inputs[key].index_select(0, tensor_indices)
                     for key in ("input_ids", "labels", "attention_mask")}
            planner = self.legacy.plan if schema == "gsa-contract-v2" else self._native_plan
            plan = planner([rows[i] for i in indices], batch, tokenizer, reference=reference,
                           qualified=qualified, reference_prefix=reference_prefix)
            require(torch.equal(plan.token_ids, token_ids[tensor_indices]), "Router changed sampled token IDs")
            stage_ids[tensor_indices] = plan.stage_ids
            selected[tensor_indices] = plan.selected_stages
            structural[tensor_indices] = plan.structural_mask
            if not reference:
                for i, decision in zip(indices, plan.decisions): decisions[i] = decision
        return AnvilTokenPlan(token_ids, stage_ids, selected, structural, decisions)
