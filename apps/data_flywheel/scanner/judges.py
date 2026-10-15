"""多模型 Judge 实现 — 按能力维度做精准交叉验证。

验证策略:
- G (Grounding): 裁剪 bbox 区域 → 问 VLM "这是什么?" → 对比标注 phrase
- A (Attribute): 裁剪 bbox → 问 VLM 物体属性(颜色/形状) → 对比标注值
- S (Spatial): 几何验证 bbox 坐标 + VLM 确认相对位置
- Aff (Affordance): 在图上画 contact_point → 问 VLM 该点所在物体和可操作性
- F (Instruction): 检查指令+答案逻辑一致性 + bbox 可见性
- H (Hallucination): 验证 absent_object 确实不在图中
- R (Reasoning): 检查推理链逻辑合理性
- P (Planning): 验证步骤中的物体都存在且可操作
- E (Multiturn): 验证多轮对话前后一致性
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import JudgeEndpoint

logger = logging.getLogger("scanner.judges")

SCORE_PATTERN = re.compile(r"(?:score|质量分|分数)[:\s]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
SCORE_PATTERN_BRACKET = re.compile(r"\[([0-9]*\.?[0-9]+)\]")


def _encode_image_to_base64(image_path: str, max_size: int = 1280) -> Optional[str]:
    """将本地图片编码为 base64 data URL。"""
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return None


def _crop_bbox_to_base64(image_path: str, bbox: List[int], padding: int = 10, max_size: int = 512) -> Optional[str]:
    """裁剪 bbox 区域并编码为 base64。"""
    if not os.path.exists(image_path) or len(bbox) < 4:
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(w, bbox[2] + padding)
        y2 = min(h, bbox[3] + padding)
        crop = img.crop((x1, y1, x2, y2))
        if max(crop.size) > max_size:
            ratio = max_size / max(crop.size)
            crop = crop.resize((int(crop.size[0] * ratio), int(crop.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to crop bbox: {e}")
        return None


def _draw_point_on_image(image_path: str, point: List[float], max_size: int = 1280) -> Optional[str]:
    """在图片上画一个红色圆点，返回 base64。"""
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image, ImageDraw
        import io
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            point = [point[0] * ratio, point[1] * ratio]
        draw = ImageDraw.Draw(img)
        r = max(8, int(min(img.size) * 0.015))
        cx, cy = int(point[0]), int(point[1])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red", outline="white", width=2)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to draw point: {e}")
        return None


def _draw_bbox_on_image(image_path: str, bbox: List[int], label: str = "", max_size: int = 1280) -> Optional[str]:
    """在图片上画 bbox 框，返回 base64。"""
    if not os.path.exists(image_path) or len(bbox) < 4:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        img = Image.open(image_path).convert("RGB")
        scale = 1.0
        if max(img.size) > max_size:
            scale = max_size / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = [int(c * scale) for c in bbox]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if label:
            draw.text((x1, max(0, y1 - 15)), label, fill="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to draw bbox: {e}")
        return None


def _parse_judge_output(text: str) -> Dict[str, Any]:
    """解析 Judge 模型输出。"""
    score = 0.5
    reason = ""
    action = "keep"

    score_match = re.search(r"\[score\]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    else:
        m = SCORE_PATTERN.search(text) or SCORE_PATTERN_BRACKET.search(text)
        if m:
            score = float(m.group(1))

    reason_match = re.search(r"\[reason\]\s*(.+?)(?:\n|\[|$)", text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    action_match = re.search(r"\[action\]\s*(keep|delete)", text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).lower()

    score = max(0.0, min(1.0, score))
    return {"score": score, "reason": reason, "action": action}


def _geometric_verify_spatial(content: Dict) -> Tuple[float, str]:
    """几何验证空间关系标注 — 不需要 VLM，纯坐标计算。"""
    relations = content.get("relations", [])
    if not relations:
        return 0.5, "no_relations_to_verify"

    passed = 0
    total = 0
    for rel in relations:
        subj_bbox = rel.get("subject_bbox", [])
        obj_bbox = rel.get("object_bbox", [])
        predicate = rel.get("predicate", "")

        if len(subj_bbox) < 4 or len(obj_bbox) < 4:
            continue

        total += 1
        sx, sy = (subj_bbox[0] + subj_bbox[2]) / 2, (subj_bbox[1] + subj_bbox[3]) / 2
        ox, oy = (obj_bbox[0] + obj_bbox[2]) / 2, (obj_bbox[1] + obj_bbox[3]) / 2

        ok = False
        if "left" in predicate and sx < ox:
            ok = True
        elif "right" in predicate and sx > ox:
            ok = True
        elif "above" in predicate and sy < oy:
            ok = True
        elif "below" in predicate and sy > oy:
            ok = True
        elif "near" in predicate or "next" in predicate:
            dist = ((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5
            avg_size = (subj_bbox[2] - subj_bbox[0] + obj_bbox[2] - obj_bbox[0]) / 2
            if dist < avg_size * 3:
                ok = True
        else:
            ok = True  # 无法验证的关系默认通过

        if ok:
            passed += 1

    if total == 0:
        return 0.5, "no_verifiable_relations"
    score = passed / total
    reason = f"geometric_check: {passed}/{total} relations consistent"
    return score, reason


# ========== 按能力维度的验证 Prompt 模板 ==========

GROUNDING_VERIFY_PROMPT = """观察图片中红框标注的区域。
问题: 红框内的物体是什么?

请直接回答物体名称（一个词或短语），不要其他解释。"""

GROUNDING_CROP_VERIFY_PROMPT = """这是从一张大图中裁剪出来的局部区域。
问题: 这个区域中主要的物体是什么?

请直接回答物体名称（英文，一个词或短语）。"""

ATTRIBUTE_VERIFY_PROMPT = """观察图片中红框标注的物体。
问题: 这个物体的{attr_type}是什么?

请直接回答属性值（一个词），不要其他解释。"""

AFFORDANCE_VERIFY_PROMPT = """观察图片中红色圆点标记的位置。
问题1: 红点所在的是什么物体?
问题2: 在红点这个位置，执行"{action}"操作是否合理?

请按格式回答:
[object] <物体名>
[reasonable] <yes/no>
[reason] <一句话理由>"""

HALLUCINATION_VERIFY_PROMPT = """仔细观察图片的全部内容。
问题: 图片中是否存在"{object}"这个物体?

请直接回答 yes 或 no。"""

PLANNING_VERIFY_PROMPT = """观察图片内容。以下是一个操作规划:
指令: {instruction}
涉及物体: {objects}

问题: 图片中是否能看到所有涉及的物体？这个规划是否在图片场景中可执行?

请按格式回答:
[objects_visible] <yes/no>
[executable] <yes/no>
[reason] <一句话理由>"""


@dataclass
class JudgeResult:
    """单个 Judge 的评估结果。"""
    judge_name: str
    score: float
    reason: str
    action: str
    raw_output: str
    verification_type: str = "generic"
    error: Optional[str] = None


class VLMJudge:
    """基于 VLM API 的单模型 Judge — 按能力维度做针对性交叉验证。"""

    def __init__(self, endpoint: JudgeEndpoint, max_image_size: int = 1280):
        self.endpoint = endpoint
        self.max_image_size = max_image_size
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _call_vlm(self, messages_content: List[Dict], max_tokens: int = 256) -> str:
        """统一调用 VLM 接口。"""
        payload = {
            "model": self.endpoint.model,
            "messages": [{"role": "user", "content": messages_content}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        client = await self._get_client()
        url = f"{self.endpoint.base_url}/v1/chat/completions"
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def judge(self, annotation: Dict[str, Any]) -> JudgeResult:
        """按能力维度分派验证策略。"""
        cap = annotation.get("capability_tag", "")
        try:
            if cap == "G":
                return await self._verify_grounding(annotation)
            elif cap == "A":
                return await self._verify_attribute(annotation)
            elif cap == "S":
                return await self._verify_spatial(annotation)
            elif cap == "Aff":
                return await self._verify_affordance(annotation)
            elif cap == "H":
                return await self._verify_hallucination(annotation)
            elif cap == "P":
                return await self._verify_planning(annotation)
            elif cap == "F":
                return await self._verify_instruction(annotation)
            else:
                return await self._verify_generic(annotation)
        except Exception as e:
            logger.warning(f"Judge error for {cap}: {e}")
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.5, reason="",
                action="keep", raw_output="", verification_type="error", error=str(e),
            )

    async def _verify_grounding(self, ann: Dict) -> JudgeResult:
        """G 验证: 裁剪 bbox → 问 VLM 是什么 → 对比标注 phrase。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        phrase = content.get("phrase", "")
        boxes = content.get("boxes", [[]])
        bbox = boxes[0] if boxes else []

        if not image_path or not bbox or len(bbox) < 4:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.2, reason="missing_image_or_bbox",
                action="delete", raw_output="", verification_type="grounding_skip",
            )

        # 策略1: 裁剪 bbox 区域问 VLM
        crop_b64 = _crop_bbox_to_base64(image_path, bbox)
        if crop_b64:
            messages = [
                {"type": "image_url", "image_url": {"url": crop_b64}},
                {"type": "text", "text": GROUNDING_CROP_VERIFY_PROMPT},
            ]
            vlm_answer = await self._call_vlm(messages, max_tokens=50)
            vlm_answer_lower = vlm_answer.lower().strip().rstrip(".")

            # 计算匹配度
            phrase_lower = phrase.lower().strip()
            match_score = self._compute_name_match(phrase_lower, vlm_answer_lower)

            reason = f"VLM says '{vlm_answer_lower}', annotation says '{phrase_lower}'"
            action = "keep" if match_score >= 0.5 else "delete"
            return JudgeResult(
                judge_name=self.endpoint.name, score=match_score, reason=reason,
                action=action, raw_output=vlm_answer, verification_type="grounding_crop_qa",
            )

        return JudgeResult(
            judge_name=self.endpoint.name, score=0.3, reason="crop_failed",
            action="keep", raw_output="", verification_type="grounding_fallback",
        )

    async def _verify_attribute(self, ann: Dict) -> JudgeResult:
        """A 验证: 裁剪物体 → 问属性 → 对比。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        bbox = content.get("bbox", [])
        attr_type = content.get("attribute_type", "")
        attr_value = content.get("attribute_value", "")
        obj_name = content.get("object", "")

        if not image_path or not bbox or len(bbox) < 4:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="missing_data",
                action="keep", raw_output="", verification_type="attribute_skip",
            )

        # 裁剪 bbox 问属性
        crop_b64 = _crop_bbox_to_base64(image_path, bbox)
        if not crop_b64:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="crop_failed",
                action="keep", raw_output="", verification_type="attribute_fallback",
            )

        prompt = ATTRIBUTE_VERIFY_PROMPT.format(attr_type=attr_type)
        # 用全图 + bbox 标注
        full_b64 = _draw_bbox_on_image(image_path, bbox, obj_name, self.max_image_size)
        img_to_use = full_b64 or crop_b64

        messages = [
            {"type": "image_url", "image_url": {"url": img_to_use}},
            {"type": "text", "text": prompt},
        ]
        vlm_answer = await self._call_vlm(messages, max_tokens=50)
        vlm_lower = vlm_answer.lower().strip()
        attr_lower = attr_value.lower().strip()

        # 模糊匹配
        if attr_lower in vlm_lower or vlm_lower in attr_lower:
            score = 0.9
        elif any(w in vlm_lower for w in attr_lower.split()):
            score = 0.7
        else:
            score = 0.3

        reason = f"attr={attr_type}, annotation='{attr_value}', VLM='{vlm_lower}'"
        action = "keep" if score >= 0.5 else "delete"
        return JudgeResult(
            judge_name=self.endpoint.name, score=score, reason=reason,
            action=action, raw_output=vlm_answer, verification_type="attribute_qa",
        )

    async def _verify_spatial(self, ann: Dict) -> JudgeResult:
        """S 验证: 几何校验坐标一致性 + VLM 确认。"""
        content = ann.get("content", {})
        geo_score, geo_reason = _geometric_verify_spatial(content)

        if geo_score < 0.5:
            return JudgeResult(
                judge_name=self.endpoint.name, score=geo_score, reason=geo_reason,
                action="delete", raw_output="", verification_type="spatial_geometric",
            )

        # 如果几何通过，额外用 VLM 验证一个关系
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        question = content.get("question", "")
        answer = content.get("answer", "")

        if image_path and question:
            full_b64 = _encode_image_to_base64(image_path, self.max_image_size)
            if full_b64:
                messages = [
                    {"type": "image_url", "image_url": {"url": full_b64}},
                    {"type": "text", "text": f"请观察图片回答: {question}\n直接给出答案。"},
                ]
                vlm_answer = await self._call_vlm(messages, max_tokens=100)
                # 简单一致性检查
                answer_words = set(answer.lower().split())
                vlm_words = set(vlm_answer.lower().split())
                overlap = len(answer_words & vlm_words) / max(len(answer_words), 1)
                vlm_score = min(1.0, overlap * 1.5 + 0.3)
                combined = (geo_score + vlm_score) / 2
                reason = f"geo={geo_score:.2f}, vlm_consistency={vlm_score:.2f}"
                return JudgeResult(
                    judge_name=self.endpoint.name, score=combined, reason=reason,
                    action="keep" if combined >= 0.5 else "delete",
                    raw_output=vlm_answer, verification_type="spatial_combined",
                )

        return JudgeResult(
            judge_name=self.endpoint.name, score=geo_score, reason=geo_reason,
            action="keep" if geo_score >= 0.5 else "delete",
            raw_output="", verification_type="spatial_geometric",
        )

    async def _verify_affordance(self, ann: Dict) -> JudgeResult:
        """Aff 验证: 在图上画 contact_point → 问 VLM 该点所在物体和可操作性。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        contact_point = content.get("contact_point", [])
        action_type = content.get("action", "")
        obj_name = content.get("object_name", "")

        if not image_path or not contact_point or len(contact_point) < 2:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="missing_data",
                action="keep", raw_output="", verification_type="affordance_skip",
            )

        # 在图上画点
        point_b64 = _draw_point_on_image(image_path, contact_point, self.max_image_size)
        if not point_b64:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="draw_failed",
                action="keep", raw_output="", verification_type="affordance_fallback",
            )

        prompt = AFFORDANCE_VERIFY_PROMPT.format(action=action_type)
        messages = [
            {"type": "image_url", "image_url": {"url": point_b64}},
            {"type": "text", "text": prompt},
        ]
        vlm_answer = await self._call_vlm(messages, max_tokens=150)

        # 解析
        score = 0.5
        reason = ""
        obj_match = re.search(r"\[object\]\s*(.+?)(?:\n|\[|$)", vlm_answer)
        reasonable_match = re.search(r"\[reasonable\]\s*(yes|no)", vlm_answer, re.IGNORECASE)

        if obj_match:
            vlm_obj = obj_match.group(1).strip().lower()
            if self._compute_name_match(obj_name.lower(), vlm_obj) >= 0.5:
                score += 0.25
            else:
                score -= 0.2

        if reasonable_match:
            if reasonable_match.group(1).lower() == "yes":
                score += 0.25
            else:
                score -= 0.3

        score = max(0.0, min(1.0, score))
        reason = f"point_on='{obj_match.group(1).strip() if obj_match else '?'}', reasonable={'yes' if reasonable_match and reasonable_match.group(1).lower() == 'yes' else 'no'}"
        action = "keep" if score >= 0.5 else "delete"
        return JudgeResult(
            judge_name=self.endpoint.name, score=score, reason=reason,
            action=action, raw_output=vlm_answer, verification_type="affordance_point_qa",
        )

    async def _verify_hallucination(self, ann: Dict) -> JudgeResult:
        """H 验证: 确认 absent_object 确实不在图中。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        absent_obj = content.get("absent_object", "")
        ann_type = content.get("type", "")

        if not image_path or not absent_obj:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.5, reason="missing_data",
                action="keep", raw_output="", verification_type="hallucination_skip",
            )

        full_b64 = _encode_image_to_base64(image_path, self.max_image_size)
        if not full_b64:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="image_load_failed",
                action="keep", raw_output="", verification_type="hallucination_fallback",
            )

        prompt = HALLUCINATION_VERIFY_PROMPT.format(object=absent_obj)
        messages = [
            {"type": "image_url", "image_url": {"url": full_b64}},
            {"type": "text", "text": prompt},
        ]
        vlm_answer = await self._call_vlm(messages, max_tokens=30)
        vlm_lower = vlm_answer.lower().strip()

        # 标注说物体不存在，VLM 应回答 "no"
        if ann_type == "negative":
            if "no" in vlm_lower and "yes" not in vlm_lower:
                score = 0.95
                reason = f"VLM confirms '{absent_obj}' absent"
            elif "yes" in vlm_lower:
                score = 0.1
                reason = f"VLM says '{absent_obj}' IS present — annotation wrong!"
            else:
                score = 0.5
                reason = f"VLM ambiguous: '{vlm_lower}'"
        else:
            score = 0.6
            reason = "non-negative type, basic check"

        action = "delete" if score < 0.3 else "keep"
        return JudgeResult(
            judge_name=self.endpoint.name, score=score, reason=reason,
            action=action, raw_output=vlm_answer, verification_type="hallucination_presence_qa",
        )

    async def _verify_planning(self, ann: Dict) -> JudgeResult:
        """P 验证: 确认规划中涉及的物体都在图中可见。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        instruction = content.get("instruction", "")
        objects = content.get("objects_involved", [])

        if not image_path or not objects:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.5, reason="missing_data",
                action="keep", raw_output="", verification_type="planning_skip",
            )

        full_b64 = _encode_image_to_base64(image_path, self.max_image_size)
        if not full_b64:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.3, reason="image_load_failed",
                action="keep", raw_output="", verification_type="planning_fallback",
            )

        prompt = PLANNING_VERIFY_PROMPT.format(
            instruction=instruction, objects=", ".join(objects)
        )
        messages = [
            {"type": "image_url", "image_url": {"url": full_b64}},
            {"type": "text", "text": prompt},
        ]
        vlm_answer = await self._call_vlm(messages, max_tokens=150)

        visible_match = re.search(r"\[objects_visible\]\s*(yes|no)", vlm_answer, re.IGNORECASE)
        exec_match = re.search(r"\[executable\]\s*(yes|no)", vlm_answer, re.IGNORECASE)

        score = 0.5
        if visible_match and visible_match.group(1).lower() == "yes":
            score += 0.25
        elif visible_match and visible_match.group(1).lower() == "no":
            score -= 0.3
        if exec_match and exec_match.group(1).lower() == "yes":
            score += 0.2
        elif exec_match and exec_match.group(1).lower() == "no":
            score -= 0.2

        score = max(0.0, min(1.0, score))
        reason = f"visible={'yes' if visible_match and visible_match.group(1).lower() == 'yes' else 'no'}, exec={'yes' if exec_match and exec_match.group(1).lower() == 'yes' else 'no'}"
        return JudgeResult(
            judge_name=self.endpoint.name, score=score, reason=reason,
            action="keep" if score >= 0.4 else "delete",
            raw_output=vlm_answer, verification_type="planning_visibility_qa",
        )

    async def _verify_instruction(self, ann: Dict) -> JudgeResult:
        """F 验证: 指令中的物体是否可见 + bbox 合理性。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        obj_name = content.get("object", "")
        expected = content.get("expected_output", {})
        bbox = expected.get("bbox", []) if isinstance(expected, dict) else []

        if not image_path or not obj_name:
            return JudgeResult(
                judge_name=self.endpoint.name, score=0.5, reason="missing_data",
                action="keep", raw_output="", verification_type="instruction_skip",
            )

        # 用 bbox 裁剪验证物体
        if bbox and len(bbox) >= 4:
            crop_b64 = _crop_bbox_to_base64(image_path, bbox)
            if crop_b64:
                messages = [
                    {"type": "image_url", "image_url": {"url": crop_b64}},
                    {"type": "text", "text": GROUNDING_CROP_VERIFY_PROMPT},
                ]
                vlm_answer = await self._call_vlm(messages, max_tokens=50)
                match_score = self._compute_name_match(obj_name.lower(), vlm_answer.lower().strip())
                reason = f"instruction_obj='{obj_name}', VLM_crop='{vlm_answer.strip()}'"
                return JudgeResult(
                    judge_name=self.endpoint.name, score=match_score, reason=reason,
                    action="keep" if match_score >= 0.5 else "delete",
                    raw_output=vlm_answer, verification_type="instruction_bbox_qa",
                )

        # 没有 bbox 就用全图验证物体可见性
        full_b64 = _encode_image_to_base64(image_path, self.max_image_size)
        if full_b64:
            messages = [
                {"type": "image_url", "image_url": {"url": full_b64}},
                {"type": "text", "text": f"图片中是否有 '{obj_name}'? 回答 yes 或 no。"},
            ]
            vlm_answer = await self._call_vlm(messages, max_tokens=30)
            if "yes" in vlm_answer.lower():
                score = 0.8
            else:
                score = 0.3
            return JudgeResult(
                judge_name=self.endpoint.name, score=score,
                reason=f"obj='{obj_name}', visible={'yes' in vlm_answer.lower()}",
                action="keep" if score >= 0.5 else "delete",
                raw_output=vlm_answer, verification_type="instruction_visibility",
            )

        return JudgeResult(
            judge_name=self.endpoint.name, score=0.4, reason="no_image",
            action="keep", raw_output="", verification_type="instruction_fallback",
        )

    async def _verify_generic(self, ann: Dict) -> JudgeResult:
        """通用验证 (E/R 等): 全图 + 标注内容一起问 VLM。"""
        content = ann.get("content", {})
        image_path = content.get("image_path", "") or ann.get("image_path", "")
        cap = ann.get("capability_tag", "")

        content_json = json.dumps(
            {k: v for k, v in content.items() if k != "image_path"},
            ensure_ascii=False, indent=2
        )[:1200]

        prompt = (
            f"你是数据质量审核员。请对照图片验证以下[{cap}]标注的质量。\n"
            f"标注内容:\n```json\n{content_json}\n```\n\n"
            f"请基于图片内容判断标注是否合理，按格式回答:\n"
            f"[score] <0-1>\n[reason] <一句话，引用图片视觉证据>\n[action] <keep/delete>"
        )

        messages_content: List[Dict] = []
        if image_path:
            img_b64 = _encode_image_to_base64(image_path, self.max_image_size)
            if img_b64:
                messages_content.append({"type": "image_url", "image_url": {"url": img_b64}})
        messages_content.append({"type": "text", "text": prompt})

        vlm_answer = await self._call_vlm(messages_content, max_tokens=200)
        parsed = _parse_judge_output(vlm_answer)
        return JudgeResult(
            judge_name=self.endpoint.name, score=parsed["score"], reason=parsed["reason"],
            action=parsed["action"], raw_output=vlm_answer, verification_type="generic_qa",
        )

    @staticmethod
    def _compute_name_match(expected: str, actual: str) -> float:
        """计算物体名称匹配度 (支持同义词/子串)。"""
        if not expected or not actual:
            return 0.3

        expected = expected.lower().strip()
        actual = actual.lower().strip()

        if expected == actual:
            return 1.0
        if expected in actual or actual in expected:
            return 0.85

        # 词级别重叠
        exp_words = set(expected.split())
        act_words = set(actual.split())
        if exp_words & act_words:
            overlap = len(exp_words & act_words) / max(len(exp_words), len(act_words))
            return 0.5 + overlap * 0.4

        # 常见同义词
        synonyms = {
            "knob": {"dial", "rotary", "switch", "control"},
            "button": {"key", "switch", "control"},
            "screen": {"display", "monitor", "panel"},
            "washing machine": {"washer", "laundry machine"},
            "machine": {"appliance", "device"},
        }
        for key, syns in synonyms.items():
            if (expected == key and actual in syns) or (actual == key and expected in syns):
                return 0.75

        return 0.2


class EnsembleJudge:
    """多模型集成 Judge — 加权平均多个 VLM 的评分。"""

    def __init__(self, endpoints: List[JudgeEndpoint], max_image_size: int = 1280):
        self.judges = [VLMJudge(ep, max_image_size) for ep in endpoints]
        self.weights = [ep.weight for ep in endpoints]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    async def close(self):
        for j in self.judges:
            await j.close()

    async def judge(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """集成评估：并行调用所有 Judge，加权平均。"""
        tasks = [j.judge(annotation) for j in self.judges]
        results: List[JudgeResult] = await asyncio.gather(*tasks)

        weighted_score = 0.0
        details = []
        for r, w in zip(results, self.weights):
            weighted_score += r.score * w
            details.append({
                "judge": r.judge_name,
                "score": r.score,
                "reason": r.reason,
                "action": r.action,
                "verification_type": r.verification_type,
                "error": r.error,
            })

        final_action = "delete" if any(r.action == "delete" for r in results) else "keep"
        if weighted_score < 0.3:
            final_action = "delete"

        return {
            "ensemble_score": round(weighted_score, 4),
            "action": final_action,
            "judge_details": details,
        }

    async def judge_batch(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量评估。"""
        results = []
        for ann in annotations:
            r = await self.judge(ann)
            results.append(r)
        return results
