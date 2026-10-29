"""Anvil-Gen v3 legacy formats with optional, explicit eleven native contracts.

The frozen v2 ``depth`` format is not renamed in place. Its historical results
remain depth-only; new spatial records carry explicit task and reference frames.
"""
from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

VERSION = "anvil-gen-v3"
FORMS = {
    "bbox": "G", "detection": "G", "mask": "G", "point": "G",
    "spatial": "S", "affordance": "A", "dof": "A", "action_cot": "A",
}
# Native contracts are not the eleven spatial task predicates.  Legacy records
# without an explicit native_form remain unclassified in native production counts.
NATIVE_CONTRACTS = {
    "bbox": {"family": "G", "form": "bbox"},
    "detection": {"family": "G", "form": "detection"},
    "mask": {"family": "G", "form": "mask"},
    "point": {"family": "G", "form": "point"},
    "depth_order": {"family": "S", "form": "spatial", "spatial_tasks": ["relative_depth"]},
    "metric_comparison": {"family": "S", "form": "spatial", "spatial_tasks": ["distance", "size"]},
    "direction": {"family": "S", "form": "spatial", "spatial_tasks": ["direction", "orientation", "viewpoint"]},
    "relation_graph": {"family": "S", "form": "spatial", "min_edges": 2},
    "affordance": {"family": "A", "form": "affordance"},
    "dof": {"family": "A", "form": "dof"},
    "action_cot": {"family": "A", "form": "action_cot"},
}
NativeForm = Literal["bbox", "detection", "mask", "point", "depth_order",
                     "metric_comparison", "direction", "relation_graph",
                     "affordance", "dof", "action_cot"]
SPATIAL_TASKS = {
    "relative_depth": {"nearer", "farther", "same_depth"},
    "direction": {"left_of", "right_of", "above", "below"},
    "distance": {"near", "far", "distance"},
    "size": {"larger", "smaller", "same_size"},
    "topology": {"overlaps", "disjoint", "touching"},
    "occlusion": {"occludes", "occluded_by"},
    "support": {"supports", "supported_by"},
    "containment": {"inside", "contains"},
    "orientation": {"parallel", "perpendicular", "facing", "away_from"},
    "viewpoint": {"left_of", "right_of", "in_front_of", "behind"},
    "motion": {"approaching", "receding", "stationary"},
}
SpatialTask = Literal["relative_depth", "direction", "distance", "size", "topology",
                      "occlusion", "support", "containment", "orientation", "viewpoint", "motion"]
Predicate = Annotated[str, Field(pattern="^("+"|".join(sorted(set().union(*SPATIAL_TASKS.values())))+")$")]
Text = Annotated[str, Field(min_length=1, max_length=2000, pattern=r"\S")]
Coord = Annotated[float, Field(ge=0, le=1000)]
Point = Annotated[list[Coord], Field(min_length=2, max_length=2)]
Box = Annotated[list[Coord], Field(min_length=4, max_length=4)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Entity(Strict):
    entity_id: Text
    label: Text
    bbox: Box

    @model_validator(mode="after")
    def ordered(self):
        if not (self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]):
            raise ValueError("unordered bbox")
        return self


class SourceRelation(Strict):
    task: SpatialTask
    subject_id: Text
    object_id: Text
    predicate: Predicate
    frame: Literal["image", "camera", "world", "observer"]
    truth: bool
    value: Annotated[float, Field(ge=0)] | None = None
    unit: Literal["m", "cm", "mm", "normalized_image"] | None = None

    @model_validator(mode="after")
    def known_relation(self):
        if self.task not in SPATIAL_TASKS or self.predicate not in SPATIAL_TASKS[self.task]:
            raise ValueError("unknown source relationship")
        if (self.value is None) != (self.unit is None):
            raise ValueError("source measurement requires both value and unit")
        if self.value is not None and (self.task != "distance" or self.predicate != "distance"):
            raise ValueError("source measurement only applies to metric distance")
        return self


class Evidence(Strict):
    evidence_id: Text
    kind: Literal["source_annotation", "sensor", "derived_geometry", "model"]
    description: Text
    entity_ids: list[Text] = Field(min_length=1)
    relation: SourceRelation | None = None


class Task(Strict):
    task_id: Text
    form: Literal["bbox", "detection", "mask", "point", "spatial", "affordance", "dof", "action_cot"]
    question: Text
    entity_ids: list[Text] = Field(min_length=1)
    spatial_task: SpatialTask | None = None
    requested_predicate: str | None = None
    observer_id: Text | None = None
    native_form: NativeForm | None = None

    @model_validator(mode="after")
    def typed(self):
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("duplicate task entity reference")
        if self.form == "detection" and not 2 <= len(self.entity_ids) <= 6:
            raise ValueError("detection requires 2–6 requested entities")
        graph = self.native_form == "relation_graph"
        if self.native_form is not None:
            native = NATIVE_CONTRACTS[self.native_form]
            if self.form != native["form"]:
                raise ValueError("native contract and legacy form disagree")
            if "spatial_tasks" in native and self.spatial_task not in native["spatial_tasks"]:
                raise ValueError("native contract and spatial task disagree")
        if self.form == "spatial":
            if (not graph and self.spatial_task not in SPATIAL_TASKS) or len(self.entity_ids) < 2:
                raise ValueError("spatial requires a supported task and two entities")
            if graph and (self.spatial_task is not None or self.requested_predicate is not None):
                raise ValueError("relation_graph uses per-edge types, not a single spatial task or predicate")
        elif self.spatial_task is not None:
            raise ValueError("spatial_task only applies to spatial")
        if self.requested_predicate is not None and (
                self.spatial_task not in SPATIAL_TASKS or
                self.requested_predicate not in SPATIAL_TASKS[self.spatial_task]):
            raise ValueError("requested predicate does not belong to spatial task")
        if not graph and (self.spatial_task == "viewpoint") != (self.observer_id is not None):
            raise ValueError("only viewpoint tasks require an explicit observer")
        return self


class Scene(Strict):
    source: Text
    scene_id: Text
    group_id: Text  # stable physical scene/episode identity, shared across sources
    images: list[Text] = Field(min_length=1, max_length=16)
    source_split: Literal["train", "validation", "test", "unknown"]
    training_approved: bool = False
    entities: list[Entity] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=2)

    @model_validator(mode="after")
    def references(self):
        ids = {e.entity_id for e in self.entities}
        if len(ids) != len(self.entities):
            raise ValueError("duplicate entity id")
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task id")
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence id")
        if len({t.form for t in self.tasks}) < 2:
            raise ValueError("cross-format adjudication requires two distinct forms")
        for item in [*self.tasks, *self.evidence]:
            if not set(item.entity_ids) <= ids:
                raise ValueError("unknown entity reference")
        for item in self.evidence:
            if item.relation and not {item.relation.subject_id, item.relation.object_id} <= set(item.entity_ids):
                raise ValueError("source relation references wrong evidence entities")
        return self


class BBox(Strict):
    label: Text
    bbox_2d: Box


class Mask(Strict):
    label: Text
    polygon: list[Point] = Field(min_length=6)


class PointAnswer(Strict):
    label: Text
    point_2d: Point


class Spatial(Strict):
    task: SpatialTask
    subject_id: Text
    object_id: Text
    predicate: Predicate
    frame: Literal["image", "camera", "world", "observer"]
    evidence_ids: list[Text] = Field(min_length=1)
    truth: bool = True
    value: Annotated[float, Field(ge=0)] | None = None
    unit: Literal["m", "cm", "mm", "normalized_image"] | None = None
    observer_id: str | None = None

    @model_validator(mode="after")
    def relation(self):
        if self.task not in SPATIAL_TASKS or self.predicate not in SPATIAL_TASKS[self.task]:
            raise ValueError("predicate does not belong to spatial task")
        if self.subject_id == self.object_id:
            raise ValueError("self relation")
        metric = self.task == "distance" and self.predicate == "distance"
        if metric != (self.value is not None and self.unit is not None):
            raise ValueError("metric distance requires value and unit")
        if not metric and (self.value is not None or self.unit is not None):
            raise ValueError("nonmetric relation cannot contain a measurement")
        if metric and (not self.truth or (self.unit != "normalized_image" and self.frame == "image")):
            raise ValueError("invalid metric measurement truth value or frame")
        if self.task == "direction" and self.frame != "image":
            raise ValueError("direction is image-frame; use viewpoint for other frames")
        if self.task == "viewpoint" and (self.frame != "observer" or not self.observer_id):
            raise ValueError("viewpoint requires an explicit observer")
        if self.task == "relative_depth" and self.frame != "camera":
            raise ValueError("relative depth is camera-frame")
        return self


class Affordance(BBox):
    affordance: Text
    contact_point_2d: Point


class Grasp(Strict):
    label: Text
    grasp_point_2d: Point
    jaw_axis_2d: Annotated[list[float], Field(min_length=2, max_length=2)]
    approach: Literal["top", "side", "front"]
    wrist_deg: Annotated[float, Field(ge=-90, le=90)]


class Action(Strict):
    goal: Text
    steps: list[Text] = Field(min_length=2)
    preconditions: list[Text] = Field(min_length=1)
    failure_modes: list[Text] = Field(min_length=1)
    target_point_2d: Point


ANSWER_TYPES = {
    "bbox": BBox, "detection": BBox, "mask": Mask, "point": PointAnswer,
    "spatial": Spatial, "affordance": Affordance, "dof": Grasp, "action_cot": Action,
}


class RuleStep(Strict):
    evidence_ids: list[Text] = Field(min_length=1)
    rule: Text
    conclusion: Annotated[str, Field(min_length=1, max_length=500)]


class Candidate(Strict):
    task_id: Text
    form: Literal["bbox", "detection", "mask", "point", "spatial", "affordance", "dof", "action_cot"]
    answer: list[dict] = Field(min_length=1)
    reasoning_trace: list[RuleStep] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def native_answer(self):
        parsed = TypeAdapter(list[ANSWER_TYPES[self.form]]).validate_python(self.answer)
        self.answer = [item.model_dump(exclude_none=True) for item in parsed]
        if self.form == "detection" and not 2 <= len(parsed) <= 6:
            raise ValueError("detection expects 2–6 entities")
        if self.form not in {"detection", "spatial"} and len(parsed) != 1:
            raise ValueError("single-target form expects exactly one item")
        return self


class Verdict(Strict):
    decision: Literal["accept", "reject", "abstain"]
    task_ids: list[Text] = Field(min_length=1)
    reason: Text


def answer_schema(form: str, *, student: bool = False, native_form: str | None = None) -> dict:
    schema = TypeAdapter(list[ANSWER_TYPES[form]]).json_schema()
    if native_form is not None:
        native = NATIVE_CONTRACTS[native_form]
        if native["form"] != form:
            raise ValueError("native contract and legacy form disagree")
        schema["minItems"] = native.get("min_edges", 1)
        if "spatial_tasks" in native:
            schema["$defs"]["Spatial"]["properties"]["task"]["enum"] = native["spatial_tasks"]
    if student and form == "spatial":
        spatial = schema["$defs"]["Spatial"]
        spatial["properties"].pop("evidence_ids")
        spatial["required"].remove("evidence_ids")
    return schema


def student_answer(candidate: Candidate) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "evidence_ids"}
            for item in candidate.answer]


def context(scene: Scene, task: Task, upstream: list[Candidate]) -> dict:
    entities = {e.entity_id: e for e in scene.entities}
    refs = [entities[k] for k in task.entity_ids]
    spatial = {c.form: c.answer for c in upstream if c.form in {"spatial", "affordance", "dof"}}
    return {"primary": {"object_class": refs[0].label, "bbox": refs[0].bbox},
            "groundings": [{"object_class": e.label, "bbox": e.bbox} for e in refs],
            "spatial": spatial}


def validate_candidate(candidate: Candidate, scene: Scene, task: Task,
                       upstream: list[Candidate]) -> None:
    """Schema + referential + available deterministic checks, not semantic proof."""
    if candidate.task_id != task.task_id or candidate.form != task.form:
        raise ValueError("candidate task mismatch")
    _validate_native_answer(task, candidate.answer)
    evidence = {e.evidence_id: e for e in scene.evidence}
    for step in candidate.reasoning_trace:
        if not set(step.evidence_ids) <= evidence.keys():
            raise ValueError("unknown trace evidence")
    if task.form != "spatial":
        from trains.flywheel.validators import VALIDATORS
        expected_labels = {e.label for e in scene.entities if e.entity_id in task.entity_ids}
        labels = {item.get("label") for item in candidate.answer}
        if task.form in {"bbox", "point", "mask", "dof"} and not labels <= expected_labels:
            raise ValueError("answer label does not match requested entity")
        if task.form == "detection" and labels != expected_labels:
            raise ValueError("detection entity set differs from requested entities")
        ok, reason = VALIDATORS[task.form](candidate.answer, context(scene, task, upstream))
        if not ok:
            raise ValueError(reason)
        return
    entities = {e.entity_id: e for e in scene.entities}
    for relation in candidate.answer:
        if task.native_form != "relation_graph" and relation["task"] != task.spatial_task:
            raise ValueError("spatial task mismatch")
        if task.requested_predicate and relation["predicate"] != task.requested_predicate:
            raise ValueError("requested spatial predicate mismatch")
        subject, obj = relation["subject_id"], relation["object_id"]
        if not {subject, obj} <= set(task.entity_ids):
            raise ValueError("unrequested spatial entity")
        if not set(relation["evidence_ids"]) <= evidence.keys():
            raise ValueError("unknown spatial evidence")
        covered = set().union(*(set(evidence[i].entity_ids) for i in relation["evidence_ids"]))
        if not {subject, obj} <= covered:
            raise ValueError("spatial evidence does not cover both entities")
        if task.requested_predicate and (subject, obj) != tuple(task.entity_ids[:2]):
            raise ValueError("requested spatial subject/object mismatch")
        if relation["task"] == "viewpoint" and relation.get("observer_id") != task.observer_id:
            raise ValueError("viewpoint observer mismatch")
        for evidence_id in relation["evidence_ids"]:
            item = evidence[evidence_id]
            if item.relation:
                expected = item.relation.model_dump(exclude_none=True)
                if any(relation.get(key) != value for key, value in expected.items()):
                    raise ValueError("relation contradicts source annotation")
        kinds = {evidence[i].kind for i in relation["evidence_ids"]}
        if relation["task"] in {"distance", "size", "orientation", "viewpoint", "motion"}:
            if not kinds & {"source_annotation", "sensor", "derived_geometry"}:
                raise ValueError("spatial task requires grounded source evidence")
        if task.native_form in {"metric_comparison", "relation_graph"} and relation["task"] in {"distance", "size"}:
            # A physical measurement is admitted only with explicit typed source
            # value/unit; text, ordinal depth and a model claim cannot supply it.
            image_metric = relation["predicate"] == "distance" and relation.get("unit") == "normalized_image"
            grounded = [evidence[i] for i in relation["evidence_ids"]
                        if evidence[i].kind in {"source_annotation", "sensor", "derived_geometry"}
                        and evidence[i].relation is not None]
            if not image_metric and not grounded:
                raise ValueError("native metric comparison requires typed source evidence")
            if relation["predicate"] == "distance" and not image_metric and not any(
                    e.relation.value == relation.get("value") and e.relation.unit == relation.get("unit")
                    for e in grounded):
                raise ValueError("native metric measurement requires explicit source value and unit")
        if relation["task"] == "motion" and len(scene.images) < 2:
            raise ValueError("motion requires ordered frames")
        if relation["task"] == "direction":
            a, b = entities[subject].bbox, entities[obj].bbox
            dx, dy = (a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2
            margins = {"left_of": -dx, "right_of": dx, "above": -dy, "below": dy}
            margin = margins[relation["predicate"]]
            if abs(margin) <= 1 or (margin > 0) != relation["truth"]:
                raise ValueError("direction contradicts source boxes or is ambiguous")
        if relation["predicate"] == "distance" and relation["unit"] == "normalized_image":
            if relation["frame"] != "image":
                raise ValueError("normalized_image measurement requires image frame")
            a, b = entities[subject].bbox, entities[obj].bbox
            expected = math.hypot((a[0]+a[2]-b[0]-b[2])/2000,
                                  (a[1]+a[3]-b[1]-b[3])/2000)
            if abs(relation["value"]-expected) > 0.001:
                raise ValueError("image-plane distance contradicts source boxes")


def _validate_native_answer(task: Task, answer: list[dict]) -> None:
    """Structural native admission; this does not calibrate a semantic evaluator."""
    if task.native_form is None or task.form != "spatial":
        return
    if task.native_form == "relation_graph":
        edges = {(r["task"], r["subject_id"], r["object_id"], r["predicate"],
                  r["frame"], r.get("observer_id")) for r in answer}
        if len(answer) < 2 or len(edges) != len(answer):
            raise ValueError("relation_graph requires at least two nonduplicate typed edges")
        covered = {r[key] for r in answer for key in ("subject_id", "object_id")}
        if covered != set(task.entity_ids):
            raise ValueError("relation_graph must preserve all requested entities")
    elif any(r["task"] not in NATIVE_CONTRACTS[task.native_form]["spatial_tasks"] for r in answer):
        raise ValueError("answer does not satisfy native spatial contract")


def student_prompt(scene: Scene, task: Task) -> str:
    """No source answers, oracle evidence, generator traces or verifier feedback."""
    refs = [{"entity_id": e.entity_id, "label": e.label,
             **({"bbox": e.bbox} if task.form not in {"bbox", "detection", "mask", "point"} else {})}
            for e in scene.entities if e.entity_id in task.entity_ids]
    return json.dumps({"contract": VERSION, "form": task.form,
                      "native_form": task.native_form,
                      "spatial_task": task.spatial_task, "question": task.question,
                      "requested_predicate": task.requested_predicate,
                      "observer_id": task.observer_id,
                      "entities": refs, "answer_schema": answer_schema(task.form, student=True, native_form=task.native_form),
                      "instruction": "Return only the answer JSON array. Coordinates use 0–1000."},
                     ensure_ascii=False)


def validate_training_record(record: dict) -> None:
    """Admission contract for v3 family-routed MOPD, not the frozen v2 router."""
    form = record.get("gsa_form")
    if record.get("gsa_contract_version") != VERSION or form not in FORMS:
        raise ValueError("unknown alignment contract")
    if record.get("gsa_family") != FORMS[form] or record.get("split") != "train":
        raise ValueError("alignment training requires a matching family and train split")
    meta = record.get("gsa_meta", {})
    if meta.get("verification") != "strong_verifier_accept" or not meta.get("verifier_model"):
        raise ValueError("missing independent verification receipt")
    messages = record.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("training requires one prompt and one reference answer")
    prompt = json.loads(messages[0]["content"])
    if prompt.get("contract") != VERSION or prompt.get("form") != form:
        raise ValueError("prompt and metadata disagree")
    if prompt.get("spatial_task") != record.get("spatial_task"):
        raise ValueError("spatial task and metadata disagree")
    native_form = record.get("gsa_native_form")
    if prompt.get("native_form") != native_form or meta.get("native_form") != native_form:
        raise ValueError("native contract and metadata disagree")
    task = Task.model_validate({"task_id": "reference", "form": form,
                               "question": prompt["question"],
                               "entity_ids": [e["entity_id"] for e in prompt["entities"]],
                               "spatial_task": record.get("spatial_task"),
                               "requested_predicate": prompt.get("requested_predicate"),
                               "observer_id": prompt.get("observer_id"), "native_form": native_form})
    payload = json.loads(messages[1]["content"])
    if form == "spatial":
        if not isinstance(payload, list) or not payload:
            raise ValueError("empty spatial reference")
        for item in payload:
            if "evidence_ids" in item or (native_form != "relation_graph" and item.get("task") != record.get("spatial_task")):
                raise ValueError("invalid student spatial reference")
        payload = [{**item, "evidence_ids": ["reference"]} for item in payload]
    parsed = Candidate.model_validate({"task_id": "reference", "form": form, "answer": payload,
                              "reasoning_trace": [{"evidence_ids": ["reference"],
                                                   "rule": "reference", "conclusion": "reference"}]})
    _validate_native_answer(task, parsed.answer)
