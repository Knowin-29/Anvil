"""Step5.6: Spatial QA 生成与校验（支持 bbox/point 答案）。"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from apps.data_flywheel.geometry import bbox_area, bbox_center, bbox_contains, bbox_iou

logger = logging.getLogger("flywheel.spatial_qa")

_PRED_TO_NL = {
    "left-of": "to the left of", "right-of": "to the right of", "above": "above", "below": "below",
    "on-top-of": "on top of", "under": "under", "front": "in front of", "behind": "behind",
    "inside": "inside", "next-to": "next to", "occluding": "occluding", "occluded-by": "occluded by",
    "stacked-on": "stacked on top of", "supporting": "supporting", "larger-than": "larger than",
    "smaller-than": "smaller than", "near": "near", "overlapping": "overlapping with",
}
_PRED_TO_NL_ZH = {
    "left-of": "在左边", "right-of": "在右边", "above": "在上方", "below": "在下方",
    "on-top-of": "在顶上", "under": "在下面", "front": "在前方", "behind": "在后方",
    "inside": "在里面", "next-to": "在旁边", "occluding": "遮挡了", "occluded-by": "被遮挡",
    "stacked-on": "堆叠在上面", "supporting": "支撑着", "larger-than": "比...大", "smaller-than": "比...小",
    "near": "靠近", "overlapping": "重叠",
}

_POSITION_PREDS = {"left-of", "right-of", "above", "below"}
_SPATIAL_3D_PREDS = {"on-top-of", "under", "front", "behind"}
_OCCLUSION_PREDS = {"occluding", "occluded-by"}
_STACKING_PREDS = {"stacked-on", "supporting"}
_CONTAINMENT_PREDS = {"inside"}
_SIZE_PREDS = {"larger-than", "smaller-than"}
_LOCATE_PREDS = _POSITION_PREDS | _SPATIAL_3D_PREDS | {"inside", "near", "next-to"}

_POSITION_OPTIONS_EN = ["to the left of", "to the right of", "above", "below"]
_POSITION_OPTIONS_ZH = ["在左边", "在右边", "在上方", "在下方"]
_SPATIAL_3D_OPTIONS_EN = ["on top of", "under", "in front of", "behind"]
_SPATIAL_3D_OPTIONS_ZH = ["在顶上", "在下面", "在前方", "在后方"]


def _obj_name(objects: List[Dict], idx: int) -> str:
    name = objects[idx]["name"]
    for prefix in ("a ", "an ", "the "):
        if name.lower().startswith(prefix):
            return name[len(prefix):]
    return name


def _obj_ref(objects: List[Dict], idx: int) -> Dict:
    o = objects[idx]
    return {"name": _obj_name(objects, idx), "id": o["id"], "bbox": o["bbox"]}


def _label(idx: int) -> str:
    return f"({chr(65 + idx)})"


def _nl(pred: str, lang: str = "en") -> str:
    return (_PRED_TO_NL_ZH if lang == "zh" else _PRED_TO_NL).get(pred, pred)


def _with_answer_fields(qa: Dict[str, Any]) -> Dict[str, Any]:
    qa.setdefault("answer_format", "text")
    qa.setdefault("answer_bbox", None)
    qa.setdefault("answer_point", None)
    qa.setdefault("supports_training", True)
    qa.setdefault("verified", True)
    qa.setdefault("verify_score", 1.0)
    qa.setdefault("verify_reasons", [])
    return qa


def _build_thinking_position(subj_name: str, obj_name: str, subj_bbox: List, obj_bbox: List, pred: str, lang: str) -> str:
    sc = bbox_center(subj_bbox)
    oc = bbox_center(obj_bbox)
    if lang == "zh":
        return f"{subj_name}框中心({sc[0]:.0f},{sc[1]:.0f})，{obj_name}框中心({oc[0]:.0f},{oc[1]:.0f})，据此判断{subj_name}{_nl(pred,'zh')}{obj_name}。"
    return f"{subj_name} center=({sc[0]:.0f},{sc[1]:.0f}), {obj_name} center=({oc[0]:.0f},{oc[1]:.0f}); thus {subj_name} is {_nl(pred)} {obj_name}."


def _build_thinking_3d(subj_name: str, obj_name: str, subj_obj: Dict, obj_obj: Dict, pred: str, lang: str) -> str:
    s_depth, o_depth = subj_obj.get("mean_depth", 0), obj_obj.get("mean_depth", 0)
    if lang == "zh":
        return f"{subj_name}深度{s_depth:.2f}m，{obj_name}深度{o_depth:.2f}m；结合3D信息，{subj_name}{_nl(pred,'zh')}{obj_name}。"
    return f"{subj_name} depth={s_depth:.2f}m, {obj_name} depth={o_depth:.2f}m; using 3D cues, {subj_name} is {_nl(pred)} {obj_name}."


def _build_thinking_simple(subj_name: str, obj_name: str, pred: str, reason: str, lang: str) -> str:
    if lang == "zh":
        return f"分析{subj_name}与{obj_name}关系：{reason}因此{subj_name}{_nl(pred,'zh')}{obj_name}。"
    return f"Reasoning over {subj_name} and {obj_name}: {reason}Therefore {subj_name} is {_nl(pred)} {obj_name}."


def _gen_position_qa(sg: Dict, rel: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    subj, obj, pred = _obj_name(objects, rel["subject"]), _obj_name(objects, rel["object"]), rel["predicate"]
    options = list(_POSITION_OPTIONS_ZH if lang == "zh" else _POSITION_OPTIONS_EN)
    pred_map = {"left-of": options[0], "right-of": options[1], "above": options[2], "below": options[3]}
    correct = pred_map.get(pred)
    if not correct:
        return None
    random.shuffle(options)
    ci = options.index(correct)
    opts_text = " ".join(f"{_label(i)} {o}" for i, o in enumerate(options))
    q = f"{subj}相对于{obj}的位置关系是什么？{opts_text}" if lang == "zh" else f"Where is the {subj} relative to the {obj}? {opts_text}"
    qa = {
        "image": sg["image"], "question": q, "thinking": _build_thinking_position(subj, obj, objects[rel["subject"]]["bbox"], objects[rel["object"]]["bbox"], pred, lang),
        "answer": f"{_label(ci)} {correct}", "type": "position", "predicate": pred,
        "subject": _obj_ref(objects, rel["subject"]), "object": _obj_ref(objects, rel["object"]), "is_negative": False,
    }
    return _with_answer_fields(qa)


def _gen_spatial_3d_qa(sg: Dict, rel: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    subj, obj, pred = _obj_name(objects, rel["subject"]), _obj_name(objects, rel["object"]), rel["predicate"]
    options = list(_SPATIAL_3D_OPTIONS_ZH if lang == "zh" else _SPATIAL_3D_OPTIONS_EN)
    pred_map = {"on-top-of": options[0], "under": options[1], "front": options[2], "behind": options[3]}
    correct = pred_map.get(pred)
    if not correct:
        return None
    random.shuffle(options)
    ci = options.index(correct)
    opts_text = " ".join(f"{_label(i)} {o}" for i, o in enumerate(options))
    q = f"{subj}和{obj}在3D空间中的关系是什么？{opts_text}" if lang == "zh" else f"What is the 3D relation between {subj} and {obj}? {opts_text}"
    qa = {
        "image": sg["image"], "question": q, "thinking": _build_thinking_3d(subj, obj, objects[rel["subject"]], objects[rel["object"]], pred, lang),
        "answer": f"{_label(ci)} {correct}", "type": "spatial_3d", "predicate": pred,
        "subject": _obj_ref(objects, rel["subject"]), "object": _obj_ref(objects, rel["object"]), "is_negative": False,
    }
    return _with_answer_fields(qa)


def _gen_yesno_qa(sg: Dict, rel: Dict, objects: List[Dict], qa_type: str, q_en: str, q_zh: str, is_negative: bool, lang: str) -> Dict[str, Any]:
    subj, obj, pred = _obj_name(objects, rel["subject"]), _obj_name(objects, rel["object"]), rel["predicate"]
    if lang == "zh":
        q, ans = q_zh.format(subj=subj, obj=obj) + " (A) 是 (B) 否", ("(B) 否" if is_negative else "(A) 是")
        reason = "该关系不存在。" if is_negative else f"该关系通过{'3D深度' if rel.get('source') == '3d' else '2D边界框'}确认。"
    else:
        q, ans = q_en.format(subj=subj, obj=obj) + " (A) yes (B) no", ("(B) no" if is_negative else "(A) yes")
        reason = "No such relation exists. " if is_negative else f"Confirmed by {'3D depth' if rel.get('source') == '3d' else '2D bbox'} analysis. "
    qa = {
        "image": sg["image"], "question": q, "thinking": _build_thinking_simple(subj, obj, pred, reason, lang),
        "answer": ans, "type": qa_type, "predicate": pred,
        "subject": _obj_ref(objects, rel["subject"]), "object": _obj_ref(objects, rel["object"]), "is_negative": is_negative,
    }
    return _with_answer_fields(qa)


def _gen_proximity_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    rels = [r for r in sg["relations"] if r["predicate"] == "near"]
    if len(objects) < 3 or not rels:
        return None
    rel = random.choice(rels)
    anchor_idx, correct_idx = rel["subject"], rel["object"]
    anchor, correct = _obj_name(objects, anchor_idx), _obj_name(objects, correct_idx)
    others = [i for i in range(len(objects)) if i not in (anchor_idx, correct_idx)]
    if not others:
        return None
    candidates = [correct_idx] + random.sample(others, min(2, len(others)))
    random.shuffle(candidates)
    ci = candidates.index(correct_idx)
    opts = " ".join(f"{_label(i)} {_obj_name(objects, c)}" for i, c in enumerate(candidates))
    q = f"哪个物体离{anchor}最近？{opts}" if lang == "zh" else f"Which object is closest to {anchor}? {opts}"
    qa = {
        "image": sg["image"], "question": q, "thinking": _build_thinking_simple(anchor, correct, "near", "通过关系图可知其最近。" if lang == "zh" else "Relation graph indicates nearest object. ", lang),
        "answer": f"{_label(ci)} {correct}", "type": "proximity", "predicate": "near",
        "subject": _obj_ref(objects, anchor_idx), "object": _obj_ref(objects, correct_idx), "is_negative": False,
    }
    return _with_answer_fields(qa)


def _gen_comparison_qa(sg: Dict, rel: Dict, objects: List[Dict], lang: str) -> Dict[str, Any]:
    subj, obj, pred = _obj_name(objects, rel["subject"]), _obj_name(objects, rel["object"]), rel["predicate"]
    sa, oa = bbox_area(objects[rel["subject"]]["bbox"]), bbox_area(objects[rel["object"]]["bbox"])
    q = f"哪个更大？(A) {subj} (B) {obj}" if lang == "zh" else f"Which is larger? (A) {subj} (B) {obj}"
    ans = f"(A) {subj}" if pred == "larger-than" else f"(B) {obj}"
    reason = f"{subj}面积{sa:.0f}px²，{obj}面积{oa:.0f}px²。 " if lang == "zh" else f"{subj} area={sa:.0f}px², {obj} area={oa:.0f}px². "
    qa = {
        "image": sg["image"], "question": q, "thinking": _build_thinking_simple(subj, obj, pred, reason, lang),
        "answer": ans, "type": "comparison", "predicate": pred,
        "subject": _obj_ref(objects, rel["subject"]), "object": _obj_ref(objects, rel["object"]), "is_negative": False,
    }
    return _with_answer_fields(qa)


def _gen_multi_hop_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    adj: Dict[int, List[Tuple[int, str]]] = {}
    for r in sg["relations"]:
        adj.setdefault(r["subject"], []).append((r["object"], r["predicate"]))
    for a in adj:
        for b, p1 in adj[a]:
            for c, p2 in adj.get(b, []):
                if c == a:
                    continue
                na, nb, nc = _obj_name(objects, a), _obj_name(objects, b), _obj_name(objects, c)
                if lang == "zh":
                    q = f"若{na}{_nl(p1,'zh')}{nb}且{nb}{_nl(p2,'zh')}{nc}，{na}与{nc}关系如何？"
                    thinking = f"{na}{_nl(p1,'zh')}{nb}，{nb}{_nl(p2,'zh')}{nc}，可得{na}与{nc}存在间接空间关联。"
                    ans = f"{na}通过{nb}间接关联{nc}"
                else:
                    q = f"If {na} is {_nl(p1)} {nb} and {nb} is {_nl(p2)} {nc}, what about {na} and {nc}?"
                    thinking = f"{na} is {_nl(p1)} {nb}, and {nb} is {_nl(p2)} {nc}; thus {na} is indirectly related to {nc}."
                    ans = f"{na} is indirectly related to {nc} via {nb}"
                qa = {
                    "image": sg["image"], "question": q, "thinking": thinking, "answer": ans, "type": "multi_hop",
                    "predicate": f"{p1}+{p2}", "subject": _obj_ref(objects, a), "object": _obj_ref(objects, c), "is_negative": False,
                }
                return _with_answer_fields(qa)
    return None


def _gen_counting_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    on_top_count: Dict[int, List[int]] = {}
    inside_count: Dict[int, List[int]] = {}
    for r in sg["relations"]:
        if r["predicate"] == "on-top-of":
            on_top_count.setdefault(r["object"], []).append(r["subject"])
        elif r["predicate"] == "inside":
            inside_count.setdefault(r["object"], []).append(r["subject"])
    for target, items in list(on_top_count.items()) + list(inside_count.items()):
        tname, count = _obj_name(objects, target), len(items)
        rel_type = "on top of" if target in on_top_count and items == on_top_count.get(target) else "inside"
        if lang == "zh":
            rel_zh = "上面" if rel_type == "on top of" else "里面"
            q, ans = f"{tname}的{rel_zh}有几个物体？", f"{count}个"
        else:
            q, ans = f"How many objects are {rel_type} the {tname}?", str(count)
        qa = {
            "image": sg["image"], "question": q,
            "thinking": f"统计与{tname}存在'{rel_type}'关系的物体数量为{count}。" if lang == "zh" else f"Count objects with '{rel_type}' relation to {tname}: {count}.",
            "answer": ans, "type": "counting", "predicate": rel_type.replace(" ", "-"),
            "subject": _obj_ref(objects, target), "object": {"name": f"{count} objects", "id": -1, "bbox": []}, "is_negative": False,
        }
        return _with_answer_fields(qa)
    return None


def _gen_action_feasibility_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    if not objects:
        return None
    obj = random.choice(objects)
    action = random.choice(["pick", "push", "stack"])
    can_do = obj.get(f"can_{action}", False)
    name = _obj_name(objects, obj["id"])
    if lang == "zh":
        action_zh = {"pick": "拿起", "push": "推动", "stack": "堆叠"}[action]
        q, ans = f"能否{action_zh}{name}？(A) 能 (B) 不能", ("(A) 能" if can_do else "(B) 不能")
    else:
        action_en = {"pick": "pick up", "push": "push", "stack": "stack on"}[action]
        q, ans = f"Can you {action_en} {name}? (A) yes (B) no", ("(A) yes" if can_do else "(B) no")
    qa = {
        "image": sg["image"], "question": q,
        "thinking": f"分析{name}的属性可得该操作{'可行' if can_do else '不可行'}。" if lang == "zh" else f"Object properties indicate this action is {'feasible' if can_do else 'not feasible'}.",
        "answer": ans, "type": "action_feasibility", "predicate": f"can_{action}",
        "subject": _obj_ref(objects, obj["id"]), "object": {"name": action, "id": -1, "bbox": []}, "is_negative": not can_do,
    }
    return _with_answer_fields(qa)


def _gen_bbox_single_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    if not objects:
        return None
    idx = random.randrange(len(objects))
    ref = _obj_ref(objects, idx)
    q = f"{ref['name']} 在哪里？请返回 bbox。" if lang == "zh" else f"Where is the {ref['name']}? Return bbox."
    qa = {
        "image": sg["image"], "question": q, "thinking": f"目标物体ID={ref['id']}，直接返回其检测框。" if lang == "zh" else f"Use object id={ref['id']} and return its detection bbox.",
        "answer": str(ref["bbox"]), "type": "bbox_single", "predicate": "locate",
        "subject": ref, "object": {"name": "bbox", "id": -1, "bbox": ref["bbox"]}, "is_negative": False,
        "answer_format": "bbox", "answer_bbox": ref["bbox"], "answer_point": None, "supports_training": True,
    }
    return _with_answer_fields(qa)


def _gen_point_single_qa(sg: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    if not objects:
        return None
    idx = random.randrange(len(objects))
    ref = _obj_ref(objects, idx)
    c = bbox_center(ref["bbox"])
    point = [round(float(c[0]), 2), round(float(c[1]), 2)]
    q = f"{ref['name']} 的目标点在哪里？请返回 point(x,y)。" if lang == "zh" else f"Where is the target point of {ref['name']}? Return point(x,y)."
    qa = {
        "image": sg["image"], "question": q, "thinking": "使用bbox中心点作为监督point。" if lang == "zh" else "Use bbox center as supervision point.",
        "answer": str(point), "type": "point_single", "predicate": "locate",
        "subject": ref, "object": {"name": "point", "id": -1, "bbox": ref["bbox"]}, "is_negative": False,
        "answer_format": "point", "answer_bbox": None, "answer_point": point, "supports_training": True,
    }
    return _with_answer_fields(qa)


def _gen_bbox_relational_qa(sg: Dict, rel: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    if rel["predicate"] not in _LOCATE_PREDS:
        return None
    subj, obj = _obj_ref(objects, rel["subject"]), _obj_ref(objects, rel["object"])
    q = f"{subj['name']}{_nl(rel['predicate'],'zh')}的物体在哪？请返回bbox。" if lang == "zh" else f"Which object is {_nl(rel['predicate'])} {subj['name']}? Return its bbox."
    qa = {
        "image": sg["image"], "question": q,
        "thinking": f"由关系边({subj['id']}->{obj['id']}, {rel['predicate']})确定目标为{obj['name']}。" if lang == "zh" else f"Use relation edge ({subj['id']}->{obj['id']}, {rel['predicate']}) to locate {obj['name']}.",
        "answer": str(obj["bbox"]), "type": "bbox_relational", "predicate": rel["predicate"],
        "subject": subj, "object": obj, "is_negative": False,
        "answer_format": "bbox", "answer_bbox": obj["bbox"], "answer_point": None, "supports_training": True,
    }
    return _with_answer_fields(qa)


def _gen_point_relational_qa(sg: Dict, rel: Dict, objects: List[Dict], lang: str) -> Optional[Dict]:
    if rel["predicate"] not in _LOCATE_PREDS:
        return None
    subj, obj = _obj_ref(objects, rel["subject"]), _obj_ref(objects, rel["object"])
    c = bbox_center(obj["bbox"])
    point = [round(float(c[0]), 2), round(float(c[1]), 2)]
    q = f"{subj['name']}{_nl(rel['predicate'],'zh')}的物体目标点在哪？返回point。" if lang == "zh" else f"Where is the point of object {_nl(rel['predicate'])} {subj['name']}? Return point."
    qa = {
        "image": sg["image"], "question": q,
        "thinking": "先定位关系目标框，再取中心点作为point监督。" if lang == "zh" else "Locate relation target bbox, then use center as point supervision.",
        "answer": str(point), "type": "point_relational", "predicate": rel["predicate"],
        "subject": subj, "object": obj, "is_negative": False,
        "answer_format": "point", "answer_bbox": None, "answer_point": point, "supports_training": True,
    }
    return _with_answer_fields(qa)


def _gen_negative_qa(sg: Dict, objects: List[Dict], existing_rels: Dict[str, Set], lang: str) -> List[Dict]:
    negatives: List[Dict] = []
    n = len(objects)
    if n < 2:
        return negatives
    for _ in range(n * 2):
        i, j = random.sample(range(n), 2)
        pk = f"{i}-{j}"
        if pk not in existing_rels.get("occlusion", set()):
            negatives.append(_gen_yesno_qa(sg, {"subject": i, "object": j, "predicate": "occluding", "source": "2d"}, objects, "occlusion", "Is {subj} occluded by {obj}?", "{subj}是否被{obj}遮挡？", True, lang))
        if pk not in existing_rels.get("stacking", set()):
            negatives.append(_gen_yesno_qa(sg, {"subject": i, "object": j, "predicate": "stacked-on", "source": "2d"}, objects, "stacking", "Is {subj} stacked on {obj}?", "{subj}是否堆叠在{obj}上？", True, lang))
        if pk not in existing_rels.get("containment", set()):
            negatives.append(_gen_yesno_qa(sg, {"subject": i, "object": j, "predicate": "inside", "source": "2d"}, objects, "containment", "Is {subj} inside {obj}?", "{subj}是否在{obj}里？", True, lang))
    return negatives


def _verify_qa(qa: Dict[str, Any], verify_iou_thresh: float, verify_point_margin: float) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 1.0
    target_bbox = qa.get("object", {}).get("bbox")
    answer_bbox = qa.get("answer_bbox")
    answer_point = qa.get("answer_point")

    if qa.get("answer_format") == "bbox":
        if not isinstance(answer_bbox, list) or len(answer_bbox) != 4 or not isinstance(target_bbox, list) or len(target_bbox) != 4:
            reasons.append("bbox_missing_or_invalid")
            score = 0.0
        else:
            iou = bbox_iou(answer_bbox, target_bbox)
            contains = bbox_contains(target_bbox, answer_bbox) or bbox_contains(answer_bbox, target_bbox)
            score = max(float(iou), 0.7 if contains else 0.0)
            if iou < verify_iou_thresh and not contains:
                reasons.append(f"bbox_iou_too_low:{iou:.3f}")

    if qa.get("answer_format") == "point":
        if not isinstance(answer_point, list) or len(answer_point) != 2 or not isinstance(target_bbox, list) or len(target_bbox) != 4:
            reasons.append("point_missing_or_invalid")
            score = 0.0
        else:
            x, y = float(answer_point[0]), float(answer_point[1])
            x1, y1, x2, y2 = [float(v) for v in target_bbox]
            inside = (x1 - verify_point_margin) <= x <= (x2 + verify_point_margin) and (y1 - verify_point_margin) <= y <= (y2 + verify_point_margin)
            score = 1.0 if inside else 0.0
            if not inside:
                reasons.append("point_outside_target_bbox")

    verified = len(reasons) == 0
    qa["verified"] = verified
    qa["verify_score"] = round(score, 4)
    qa["verify_reasons"] = reasons
    if not verified:
        qa["supports_training"] = False
    return qa


def generate_spatial_qa(
    scene_graph_jsonl: str,
    output_jsonl: str = "data/output/flywheel/spatial_qa.jsonl",
    max_qa_per_image: int = 20,
    include_negatives: bool = True,
    negative_ratio: float = 0.3,
    enable_thinking: bool = True,
    lang: str = "en",
    enable_bbox_single: bool = True,
    enable_bbox_relational: bool = True,
    enable_point_single: bool = True,
    enable_point_relational: bool = True,
    enable_verifier: bool = True,
    verify_iou_thresh: float = 0.7,
    verify_point_margin: float = 0.0,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    scene_graphs: List[Dict] = []
    with open(scene_graph_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scene_graphs.append(json.loads(line))

    all_qa: List[Dict] = []
    type_counts: Dict[str, int] = {}
    verified_count = 0

    for sg in scene_graphs:
        objects, relations = sg["objects"], sg["relations"]
        image_qa: List[Dict] = []
        existing_rels: Dict[str, Set] = {"occlusion": set(), "stacking": set(), "containment": set()}

        for rel in relations:
            pred = rel["predicate"]
            pk = f"{rel['subject']}-{rel['object']}"
            qa: Optional[Dict[str, Any]] = None
            if pred in _POSITION_PREDS:
                qa = _gen_position_qa(sg, rel, objects, lang)
            elif pred in _SPATIAL_3D_PREDS:
                qa = _gen_spatial_3d_qa(sg, rel, objects, lang)
            elif pred in _OCCLUSION_PREDS:
                existing_rels["occlusion"].add(pk)
                qa = _gen_yesno_qa(sg, rel, objects, "occlusion", "Is {subj} occluded by {obj}?", "{subj}是否被{obj}遮挡？", False, lang)
            elif pred in _STACKING_PREDS:
                existing_rels["stacking"].add(pk)
                qa = _gen_yesno_qa(sg, rel, objects, "stacking", "Is {subj} stacked on {obj}?", "{subj}是否堆叠在{obj}上？", False, lang)
            elif pred in _CONTAINMENT_PREDS:
                existing_rels["containment"].add(pk)
                qa = _gen_yesno_qa(sg, rel, objects, "containment", "Is {subj} inside {obj}?", "{subj}是否在{obj}里？", False, lang)
            elif pred in _SIZE_PREDS:
                qa = _gen_comparison_qa(sg, rel, objects, lang)
            if qa:
                image_qa.append(qa)

            if enable_bbox_relational:
                qa = _gen_bbox_relational_qa(sg, rel, objects, lang)
                if qa:
                    image_qa.append(qa)
            if enable_point_relational:
                qa = _gen_point_relational_qa(sg, rel, objects, lang)
                if qa:
                    image_qa.append(qa)

        extra_qas = [
            _gen_proximity_qa(sg, objects, lang),
            _gen_multi_hop_qa(sg, objects, lang),
            _gen_counting_qa(sg, objects, lang),
            _gen_action_feasibility_qa(sg, objects, lang),
        ]
        image_qa.extend([x for x in extra_qas if x is not None])

        if enable_bbox_single:
            qa = _gen_bbox_single_qa(sg, objects, lang)
            if qa:
                image_qa.append(qa)
        if enable_point_single:
            qa = _gen_point_single_qa(sg, objects, lang)
            if qa:
                image_qa.append(qa)

        if include_negatives:
            neg = _gen_negative_qa(sg, objects, existing_rels, lang)
            max_neg = max(1, int(max(1, len(image_qa)) * negative_ratio))
            if neg:
                image_qa.extend(random.sample(neg, min(max_neg, len(neg))))

        if len(image_qa) > max_qa_per_image:
            image_qa = random.sample(image_qa, max_qa_per_image)

        if enable_verifier:
            image_qa = [_verify_qa(qa, verify_iou_thresh, verify_point_margin) for qa in image_qa]

        if not enable_thinking:
            for qa in image_qa:
                qa.pop("thinking", None)

        for qa in image_qa:
            type_counts[qa["type"]] = type_counts.get(qa["type"], 0) + 1
            if qa.get("verified", True):
                verified_count += 1
        all_qa.extend(image_qa)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    logger.info(f"Spatial QA: {len(all_qa)} 条, verified={verified_count}, 类型分布: {type_counts}")
    return {
        "count": len(all_qa),
        "verified_count": verified_count,
        "type_counts": type_counts,
        "output_jsonl": output_jsonl,
    }
