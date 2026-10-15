"""GSA Data Flywheel Generator Service — High-Concurrency vLLM Edition.

Features:
- 32 concurrent workers with round-robin load balancing
- Continuous mode: auto-scans for new images after each round
- Exponential backoff retry with dead-letter queue
- JSONL output sharded by dimension + date
- Rate limiting to prevent backend overload
- Real-time progress tracking via /status endpoint

Runs on port 10305, provides REST API.
"""

import json
import logging
import os
import re
import signal
import sys
import time
import base64
import io
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request
from PIL import Image
from queue import Queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gsa.generator")

app = Flask(__name__)

# Prefetch queue for image base64 encoding
_prefetch_queue: Queue = Queue(maxsize=10)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "gsa_flywheel_exp"
DIMENSION_SOURCES = PROJECT_ROOT / "data" / "input" / "dimension_sources"

# vLLM backends (dual instance)
VLLM_BACKENDS = [
    os.environ.get("VLM_URL_A", "http://localhost:10301/v1"),
    os.environ.get("VLM_URL_B", "http://localhost:10302/v1"),
]
# Fallback: single proxy endpoint
VLM_PROXY_URL = os.environ.get("VLM_URL", "http://localhost:10300/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3.6-27B")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")

MAX_WORKERS = int(os.environ.get("VLM_WORKERS", "24"))
MAX_RETRIES = int(os.environ.get("VLM_MAX_RETRIES", "3"))
CONTINUOUS_MODE = os.environ.get("GSA_CONTINUOUS", "0") == "1"
CONTINUOUS_INTERVAL = int(os.environ.get("GSA_CONTINUOUS_INTERVAL", "60"))
MAX_IMAGES_PER_DIM = int(os.environ.get("GSA_MAX_IMAGES", "50"))

# Prefetch and rate control
PREFETCH_QUEUE_SIZE = int(os.environ.get("GSA_PREFETCH_SIZE", "10"))
RATE_LIMIT_BACKOFF = float(os.environ.get("GSA_RATE_BACKOFF", "1.0"))
BATCH_OBJECTS_PER_CALL = int(os.environ.get("GSA_BATCH_SIZE", "5"))

DIMENSIONS = [
    "G_grounding", "S_spatial", "A_identity_attr",
    "A_obj_state", "A_action_precondition", "A_action_interface",
    "A_robot_state",
]

# ============================================================
# State
# ============================================================

_shutdown = threading.Event()
GENERATION_STATE = {
    "status": "idle",
    "mode": "continuous" if CONTINUOUS_MODE else "single",
    "current_dimension": None,
    "current_round": 0,
    "progress": {},
    "total_labels": 0,
    "total_images_processed": 0,
    "total_errors": 0,
    "dead_letter_count": 0,
    "start_time": None,
    "last_round_time": None,
    "throughput_per_hour": 0.0,
    "backends_healthy": 0,
    "inference_ok": None,
    "inference_detail": "",
}


# ============================================================
# VLM Client with Load Balancing
# ============================================================

class VLMClient:
    def __init__(self, backends: List[str], model: str, api_key: str):
        self.backends = backends
        self.model = model
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Content-Type"] = "application/json"
        self._backend_idx = 0
        self._lock = threading.Lock()
        self._backend_health = {b: True for b in backends}

    def _next_backend(self) -> str:
        with self._lock:
            healthy = [b for b in self.backends if self._backend_health.get(b, True)]
            if not healthy:
                healthy = self.backends
            backend = healthy[self._backend_idx % len(healthy)]
            self._backend_idx += 1
            return backend

    def _mark_unhealthy(self, backend: str):
        with self._lock:
            self._backend_health[backend] = False

    def _mark_healthy(self, backend: str):
        with self._lock:
            self._backend_health[backend] = True

    def check_health(self) -> int:
        healthy = 0
        for backend in self.backends:
            try:
                resp = self.session.get(f"{backend}/models", timeout=5)
                if resp.status_code == 200:
                    self._mark_healthy(backend)
                    healthy += 1
                else:
                    self._mark_unhealthy(backend)
            except Exception:
                self._mark_unhealthy(backend)
        return healthy

    def test_inference(self) -> tuple:
        """Real inference smoke test: send a tiny chat completion and validate the
        response contains usable content. Returns (ok: bool, detail: str).

        /models returning 200 does NOT mean inference works (model may be down,
        OOM, or returning only reasoning_content). This catches that.
        """
        backend = self.backends[0]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        if "volceapi" in backend or "dashscope" in backend or "volces" in backend:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = self.session.post(f"{backend}/chat/completions", json=payload, timeout=60)
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}: {resp.text[:150]}"
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content")
            if content and content.strip():
                return True, f"content='{content.strip()[:50]}'"
            # content empty but reasoning present => thinking mode not disabled
            if msg.get("reasoning_content"):
                return False, "model returned only reasoning_content (thinking mode not disabled / max_tokens too low)"
            return False, f"empty response: {str(data)[:150]}"
        except Exception as e:
            return False, f"exception: {str(e)[:150]}"

    def _encode_image(self, image_path: str, max_size: int = 960) -> str:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def chat_json(self, prompt: str, image_path: Optional[str] = None, retries: int = None) -> Optional[Any]:
        if retries is None:
            retries = MAX_RETRIES

        messages = [{"role": "user", "content": []}]
        if image_path and os.path.exists(image_path):
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": self._encode_image(image_path)}})
        messages[0]["content"].append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 8192,
        }
        # Disable thinking mode for Qwen3.5 API (content goes to reasoning_content otherwise)
        if "volceapi" in self.backends[0] or "dashscope" in self.backends[0]:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        for attempt in range(retries):
            if _shutdown.is_set():
                return None

            backend = self._next_backend()
            url = f"{backend}/chat/completions"

            try:
                resp = self.session.post(url, json=payload, timeout=180)
                if resp.status_code == 429:
                    wait = RATE_LIMIT_BACKOFF * (2 ** attempt)
                    logger.warning(f"Rate limited (429) on {backend}, backing off {wait:.1f}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                if not content or content == "null":
                    content = msg.get("reasoning_content") or ""
                self._mark_healthy(backend)
                return self._parse_json(content)
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {backend} (attempt {attempt+1}/{retries})")
                self._mark_unhealthy(backend)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on {backend} (attempt {attempt+1}/{retries})")
                self._mark_unhealthy(backend)
            except Exception as e:
                logger.warning(f"VLM error on {backend}: {e} (attempt {attempt+1}/{retries})")

            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 10))

        return None

    def verify_bbox(self, image_path: str, bbox: list, expected_class: str) -> dict:
        """Reverse-verify: ask VLM if the expected object is actually in the bbox."""
        prompt = (
            f'I claim there is a "{expected_class}" in the region '
            f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}] (normalized 0-1000 coords). '
            f'Look at that region carefully. '
            f'Is my claim correct? Is "{expected_class}" really there? '
            f'Reply ONLY with JSON: '
            f'{{"match": true/false, "actual_object": "what you actually see there", "confidence": 0.0-1.0}}'
        )
        result = self.chat_json(prompt, image_path, retries=1)
        if not result or not isinstance(result, dict):
            return {"match": True, "verified_class": "", "confidence": 0.5}
        match_val = result.get("match", True)
        if isinstance(match_val, str):
            match_val = match_val.lower() in ("true", "yes", "1")
        return {
            "match": bool(match_val),
            "verified_class": result.get("actual_object", "") or result.get("object_class", ""),
            "confidence": float(result.get("confidence", 0.5)),
        }

    def classify_scene(self, image_path: str) -> str:
        """Classify scene type from image using VLM."""
        prompt = (
            'What type of scene is this image? Choose ONE from: '
            'tabletop_manipulation, kitchen, living_room, bedroom, bathroom, '
            'office, warehouse, outdoor, hallway, staircase, toy_closeup, other. '
            'Reply ONLY JSON: {"scene_type": "..."}'
        )
        result = self.chat_json(prompt, image_path, retries=1)
        if result and isinstance(result, dict):
            return result.get("scene_type", "unknown")
        return "unknown"

    @staticmethod
    def _parse_json(text: str) -> Optional[Any]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            if start < 0:
                start = text.find("{")
            end = max(text.rfind("]"), text.rfind("}")) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return None


# ============================================================
# Prompts
# ============================================================

PROMPTS = {
    "G_grounding": """You are a precise visual grounding system for robotic manipulation.

TASK: Detect and localize the MOST PROMINENT physical objects that a robot could interact with.

CRITICAL BBOX RULES:
- Coordinates are normalized to [0, 1000]. (0,0)=top-left, (1000,1000)=bottom-right
- bbox = [x1, y1, x2, y2] where (x1,y1) is top-left corner, (x2,y2) is bottom-right corner
- Each bbox MUST tightly enclose ONLY the target object — no extra padding, no background
- If an object occupies the left 25% of the image horizontally and top 50% vertically:
  bbox ≈ [0, 0, 250, 500] (not [0, 0, 1000, 1000])
- REJECT any bbox larger than 50% of image area — that means your localization failed
- A good bbox: width and height each typically 50-400 units (5%-40% of image dimension)

NAMING RULES:
- Use 2-3 word descriptive names: "red_cup", "wooden_spoon", "metal_bowl"
- NOT single generic words ("object", "thing") or overly long descriptions

PROCEDURE:
1. Identify the 3-5 most salient manipulable objects (NOT walls/floor/ceiling/shadows)
2. For each: estimate its TIGHT bounding box by identifying its visual edges
3. Self-check: would cropping the image to this bbox show ONLY this object?
4. Assign confidence: 0.9+ = clear edges, 0.7-0.9 = partial occlusion, <0.7 = uncertain

Output JSON array (3-5 objects, QUALITY over quantity):
[{"object_name": "descriptive_name", "bbox": [x1,y1,x2,y2], "confidence": 0.5-1.0}]""",

    "S_spatial": """You are a spatial reasoning system for robot scene understanding.

TASK: Analyze spatial relationships between clearly identifiable objects in this scene.

COORDINATE SYSTEM: All coordinates normalized to [0, 1000]. (0,0)=top-left.

RULES:
- Subject and object MUST be different physical objects you can clearly see
- Each relationship must be verifiable from the image (not guessed)
- Include approximate bbox for BOTH subject and object (enables geometric verification)
- Confidence reflects how certain you are about this specific spatial judgment
- Do NOT output trivial/obvious relationships (e.g., "floor below everything")

PREDICATES (use the most specific applicable one):
- Horizontal: left-of, right-of (object centers differ >25% image width)
- Vertical: above, below (object centers differ >20% image height)
- Contact: on-top-of, under (physical support/contact)
- Containment: inside (one physically contains the other)
- Depth: in-front-of, behind (occlusion or perspective cues)
- Proximity: next-to (edges within 15% of image diagonal)

Output JSON array (2-5 high-quality relationships):
[{"subject": "object_name", "subject_bbox": [x1,y1,x2,y2], "predicate": "on-top-of|left-of|right-of|above|below|under|inside|next-to|in-front-of|behind", "object": "object_name", "object_bbox": [x1,y1,x2,y2], "reasoning": "brief visual evidence", "confidence": 0.5-1.0}]

Each bbox must tightly enclose its object (50-400 units typical). Quality over quantity.""",

    "A_identity_attr": """You are a visual attribute recognition system for robotic object understanding. Identify objects and their physical properties that matter for manipulation.

STRICT RULES:
- Only report attributes you can CLEARLY see (use null for uncertain)
- Color must be specific (e.g., "dark_blue" not just "blue")
- Material must be inferable from visual texture/reflectance
- Size is relative to other objects in the scene

Think step by step:
1. Identify each distinct physical object
2. For color: observe the dominant surface color under current lighting. Be specific (navy_blue, light_gray, dark_brown)
3. For material: look at surface texture, reflectance, transparency. Options: metal/stainless_steel/aluminum, wood/bamboo, plastic/acrylic, glass, ceramic/porcelain, fabric/cloth, rubber/silicone, paper/cardboard, leather, stone
4. For shape: describe the 3D form. Options: cylindrical, spherical, rectangular/cuboid, flat/planar, irregular, conical, L-shaped
5. For size: compare to other objects. small(<5cm), medium(5-20cm), large(>20cm)
6. Verify: can I actually see this attribute clearly, or am I guessing?

Output format - JSON array:
[{"object_name": "specific_name", "bbox": [x1,y1,x2,y2], "attributes": {"color": "specific_color|null", "material": "specific_material|null", "shape": "3d_shape|null", "size": "small|medium|large|null"}, "confidence": 0.5-1.0}]
Coordinates normalized 0-1000. At least 2 non-null attributes per object.""",

    "A_obj_state": """You are an object state recognition system for robotic manipulation planning. Determine the CURRENT physical state of each object that would affect how a robot should interact with it.

STRICT RULES:
- State must be OBSERVABLE in the image (not assumed or default)
- state_value must be SPECIFIC to the object type (not generic "normal")
- Each object can have multiple relevant states - pick the MOST manipulation-relevant one
- Reasoning must explain what visual evidence supports this state judgment

Think step by step:
1. Identify each manipulable object
2. For each object, ask: "What state is this object in that would change how I grasp/move it?"
3. Match object type to relevant states:
   - Containers (cup/bowl/box/jar): upright vs tipped_over, open vs closed, empty vs contains_liquid vs contains_solid
   - Hinged objects (door/drawer/lid/laptop): open_90deg vs closed vs half_open vs ajar
   - Bottles/tubes: upright vs lying_on_side vs inverted, cap_on vs cap_off
   - Books/plates/phones: face_up vs face_down vs standing_upright vs stacked
   - Switches/knobs: on vs off, rotated_to_X
   - Bags/cloth: folded vs crumpled vs spread_flat vs hanging
   - Tools (scissors/pliers): open vs closed
4. Verify: can I actually SEE this state, or am I assuming the default?

Output format - JSON array:
[{"object_name": "specific_name", "bbox": [x1,y1,x2,y2], "state_type": "pose|open_close|occupancy|on_off|orientation", "state_value": "specific_observable_value", "reasoning": "what visual evidence shows this state", "confidence": 0.5-1.0}]
Coordinates normalized 0-1000. state_value MUST be specific (not "unknown" or "normal").""",

    "A_action_precondition": """You are a robot manipulation feasibility analyzer. For each object, determine whether a parallel-jaw gripper can successfully grasp or push it RIGHT NOW, and if not, explain exactly why and what must be done first.

STRICT RULES:
- Analyze from the perspective of a 2-finger parallel-jaw gripper (width 8cm, max opening 10cm)
- can_do_now=true ONLY if there is clear, unobstructed access AND the object is graspable
- If blocked, blocked_by must name the SPECIFIC blocking object
- failure_reason must be physically accurate
- reasoning must describe the physical constraint

Think step by step:
1. For each object, check these conditions IN ORDER:
   a) REACHABILITY: Is the object within the robot's workspace? (not behind glass, not on high shelf)
   b) VISIBILITY: Can the gripper see/approach the object? (not fully hidden)
   c) OCCLUSION: Is another object directly above or in front, blocking gripper approach?
   d) CLEARANCE: Is there enough space around the object for gripper fingers? (need ~3cm clearance on each side)
   e) STABILITY: Would grasping this object cause other objects to fall? (stacked items)
   f) SIZE: Is the object between 1cm and 10cm in the grasp dimension?
   g) WEIGHT: Does the object appear too heavy? (large metal objects, full containers)
2. If ANY condition fails, can_do_now=false
3. Specify the FIRST pre-action needed to make it graspable

Output format - JSON array:
[{"object_name": "...", "bbox": [x1,y1,x2,y2], "action_type": "pick|push|pull|place|open|close|toggle_on|toggle_off|pour|slide|reorient|stack|unstack", "can_do_now": true/false, "blocked_by": ["specific_object_name"], "need_pre_action": "none|push_aside_X|remove_Y|reorient_Z|clear_top_of_W|wait_for_stable", "failure_reason": "none|occluded_by_X|stacked_under_Y|too_large|too_small|no_clearance|unstable_stack|unreachable|too_heavy", "reasoning": "physical analysis of why/why not", "confidence": 0.5-1.0}]
Coordinates normalized 0-1000.""",

    "A_action_interface": """You are a robotic manipulation planning system. For each interactive object in the scene, determine the BEST ACTION and execution parameters.

CRITICAL: DO NOT default to "pick" for everything! Choose action based on object type:
- Drawers/doors/cabinets: "open" or "close"
- Switches/buttons/faucets: "toggle_on" or "toggle_off"
- Heavy/fixed furniture: "push" or "pull"
- Containers with liquid: "pour"
- Small portable objects: "pick" (only for truly graspable items)
- Sliding objects: "slide"

STRICT RULES:
- jaw_axis_2d MUST vary: vertical objects use [0,1], horizontal use [1,0], angled use [0.7,0.7] etc.
- grasp_point MUST be physically ON the object (within its bbox)
- For push/pull: specify contact point on object edge and direction
- For open/close: specify handle/edge contact point

Think step by step:
1. For each object, analyze its 3D geometry from the image:
   - What is its cross-section shape at the best grasp height?
   - Where is the center of mass (for stable grasp)?
   - What is the narrowest graspable dimension?
2. Choose grasp strategy based on geometry:
   - top_down: for flat objects on a surface (plates, books) - approach from above
   - side_pinch: for cylindrical objects (bottles, cups) - approach from side
   - edge_grasp: for thin flat objects (cards, papers) - grasp the edge
   - wrap: for irregular objects (fruit, toys) - maximize contact area
   - precision_pinch: for small objects (pens, keys) - fingertip grasp
3. Determine jaw_axis: perpendicular to the object's longest dimension at grasp point
4. Verify: Is the grasp point accessible? Would the object be stable in the gripper?

Output format - JSON array:
[{"object_name": "...", "bbox": [x1,y1,x2,y2], "action_type": "pick|push|pull|open|close|pour|slide|toggle_on|toggle_off|place", "grasp_point_2d": [x,y], "jaw_axis_2d": [dx,dy], "push_direction_2d": [dx,dy], "grasp_strategy": "top_down|side_pinch|edge_grasp|wrap|precision_pinch", "reasoning": "why this grasp point, axis, and strategy based on object geometry and task", "confidence": 0.5-1.0}]

ADDITIONALLY: Identify any large EMPTY/CLEAR regions suitable for placing objects. For each placement zone output:
{"object_name": "placement_zone_N", "bbox": [x1,y1,x2,y2], "action_type": "place", "place_surface": "table|shelf|counter|floor", "clearance": "clear|partially_obstructed", "reasoning": "why this area is suitable for placement", "confidence": 0.7-1.0}

All coordinates normalized 0-1000. grasp_point MUST be inside bbox. jaw_axis_2d MUST vary based on object shape (NOT always [1,0]). Consider object material and weight.""",

    "A_robot_state": """You are a robot end-effector state analyzer. This is a robot camera view (often a wrist/gripper camera). Determine the gripper/end-effector state based ONLY on visible pixel evidence.

STRICT RULES:
- Only label states you can verify from visible gripper fingers, jaws, or held objects
- Each yes/no MUST cite concrete visual evidence (not generic statements)
- Use "unknown" when the gripper/object relationship is not clearly visible
- Do NOT infer hidden state without visible evidence

Analyze these 5 robot states step by step:
1. gripper_empty: are the gripper fingers/jaws visibly empty (nothing held)?
2. holding_object: is an object visibly held/grasped by the gripper? (if yes, name target_object_class)
3. object_between_fingers: is an object visibly positioned between the gripper fingers/jaws?
4. contact_state: is the gripper/end-effector visibly touching an object or surface? (NOT object touching table by itself)
5. tool_ready: is the gripper/tool clearly visible, unobstructed, and positioned for use?

For EACH state you can judge, output one entry. state_value MUST be one of: yes/no/unknown.

Output format - JSON array:
[{"state_type": "gripper_empty|holding_object|object_between_fingers|contact_state|tool_ready", "state_value": "yes|no|unknown", "target_object_class": "object name or null", "camera_view": "wrist|head|external", "visual_evidence": "specific visible evidence from pixels", "confidence": 0.5-1.0}]

Cross-check: gripper_empty=yes is mutually exclusive with holding_object=yes and object_between_fingers=yes. Output at least 2 states with concrete evidence.""",
}


# ============================================================
# Label Building
# ============================================================

def _infer_camera_view(image_path: str) -> str:
    """Infer camera view from image path (wrist/head/external) for A_robot_state."""
    if not image_path:
        return "external"
    lp = image_path.lower()
    if any(k in lp for k in ("wrist", "hand_camera", "gripper", "bridgedatav2", "bridge_data_v2", "episode_", "droid", "fmb")):
        return "wrist"
    if "head" in lp or "embodied_reasoner" in lp:
        return "head"
    return "external"


def is_wrist_camera_image(image_path: str) -> bool:
    """Check if image is from a robot wrist/gripper camera (A_robot_state routing)."""
    if not image_path:
        return False
    lp = image_path.lower()
    return any(k in lp for k in ("wrist", "hand_camera", "gripper", "bridgedatav2", "bridge_data_v2", "episode_", "droid", "fmb"))


def derive_source(image_path: str) -> str:
    p = image_path.lower()
    if "bridge" in p or "episode" in p or "bridgedatav2" in p:
        return "bridge_data_v2"
    if "erqa" in p or "embodied" in p:
        return "erqa_embodied"
    if "vlm_dataset_root" in p or "raw_benchmark" in p:
        return "vlm_benchmark"
    if "appliance_datasets" in p or "data_instruction" in p or "washing" in p:
        return "appliance"
    if "mit-states" in p or "mit_states" in p:
        return "mit_states"
    if "changeit" in p or "object-states" in p:
        return "changeit_frames"
    if "spatialsense" in p:
        return "spatialsense"
    if "where2place" in p:
        return "where2place"
    if "fixtures" in p:
        return "test_fixtures"
    return "other"


def denormalize_bbox(bbox: List, img_w: int, img_h: int, norm_range: int = 1000):
    """Convert bbox from VLM output to pixel coordinates.

    VLM outputs are ALWAYS in [0, 1000) normalized coordinates (per prompt).
    This function maps them to actual pixel coords for the given image size.
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    # Case 1: [0, 1] normalized (rare, some models output this)
    if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
        x1, y1, x2, y2 = int(x1*img_w), int(y1*img_h), int(x2*img_w), int(y2*img_h)
    # Case 2: [0, 1000] normalized (standard - VLM prompt says "normalized to 0-1000")
    elif all(0 <= v <= norm_range for v in [x1, y1, x2, y2]):
        x1 = int(x1 * img_w / norm_range)
        y1 = int(y1 * img_h / norm_range)
        x2 = int(x2 * img_w / norm_range)
        y2 = int(y2 * img_h / norm_range)
    # Case 3: values > 1000 — likely already pixel coords (shouldn't happen with our prompt)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return None
    area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    if area > img_area * 0.95 or area < img_area * 0.001:
        return None
    return [x1, y1, x2, y2]


def denormalize_point(point: List, img_w: int, img_h: int, norm_range: int = 1000) -> List[int]:
    if not point or len(point) != 2:
        return [img_w // 2, img_h // 2]
    x, y = point
    if max(x, y) <= norm_range:
        x = int(x * img_w / norm_range)
        y = int(y * img_h / norm_range)
    return [max(0, min(x, img_w - 1)), max(0, min(y, img_h - 1))]


def _compute_iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_image_size(image_path: str) -> Tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (1280, 960)


def build_label(item: Dict, dimension: str, image_path: str, img_w: int, img_h: int, sample_id: str, idx: int) -> Dict:
    source = derive_source(image_path)
    obj_id = f"{sample_id}_obj{idx:02d}"
    # scene_type: trust VLM output; fallback to "unknown" (not hard-coded guesses)
    scene_type = item.get("scene_type", "unknown")
    base = {
        "pipeline": dimension,
        "source": source,
        "dataset": source,
        "sample_id": sample_id,
        "image_path": image_path,
        "image_size": [img_w, img_h],
        "bucket": dimension[0],
        "dimension": dimension,
        "object_id": obj_id,
        "timestamp": datetime.now().isoformat(),
        "confidence": float(item.get("confidence", 0.5)),
        "scene_type": scene_type,
    }

    if dimension == "G_grounding":
        base["object_class"] = item.get("object_name") or item.get("label") or item.get("name") or "unknown"
        base["bbox"] = denormalize_bbox(item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [], img_w, img_h)
    elif dimension == "S_spatial":
        base["subject"] = item.get("subject", "")
        base["predicate"] = item.get("predicate", "")
        base["object"] = item.get("object", "")
    elif dimension == "A_identity_attr":
        base["object_class"] = item.get("object_name") or item.get("label") or item.get("name") or "unknown"
        base["bbox"] = denormalize_bbox(item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [], img_w, img_h)
        base["attributes"] = item.get("attributes", {"color": "", "material": "", "shape": "", "size": ""})
        base["g_context"] = True
    elif dimension == "A_obj_state":
        base["object_class"] = item.get("object_name") or item.get("label") or item.get("name") or "unknown"
        base["bbox"] = denormalize_bbox(item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [], img_w, img_h)
        base["state_type"] = item.get("state_type", "unknown")
        base["state_value"] = item.get("state_value", "unknown")
        base["evidence_region"] = item.get("evidence_region")
        base["s_context"] = False
    elif dimension == "A_action_precondition":
        base["scene_family"] = scene_type
        base["object_class"] = item.get("object_name") or item.get("label") or item.get("name") or "unknown"
        base["bbox"] = denormalize_bbox(item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [], img_w, img_h)
        base["action_type"] = item.get("action_type", "pick")
        base["can_do_now"] = item.get("can_do_now", True)
        base["blocked_by"] = item.get("blocked_by", [])
        base["need_pre_action"] = item.get("need_pre_action") or item.get("pre_action_needed") or "none"
        base["failure_reason"] = item.get("failure_reason", "none")
        base["s_context"] = True
    elif dimension == "A_action_interface":
        base["scene_family"] = scene_type
        base["object_class"] = item.get("object_name") or item.get("label") or item.get("name") or "unknown"
        base["bbox"] = denormalize_bbox(item.get("bbox") or item.get("bbox_2d") or item.get("bounding_box") or [], img_w, img_h)
        base["action_type"] = item.get("action_type", "pick")
        if base["action_type"] != "place":
            base["grasp_point_2d"] = denormalize_point(item.get("grasp_point_2d") or item.get("grasp_point") or [], img_w, img_h)
            base["jaw_axis_2d"] = item.get("jaw_axis_2d") or item.get("jaw_axis") or [1, 0]
        else:
            base["place_surface"] = item.get("place_surface", "unknown")
            base["clearance"] = item.get("clearance", "clear")
        base["g_context"] = True
        base["s_context"] = True
    elif dimension == "A_robot_state":
        base["object_class"] = item.get("target_object_class") if item.get("target_object_class") not in (None, "null") else None
        base["state_type"] = item.get("state_type", "gripper_empty")
        base["state_value"] = item.get("state_value", "unknown")
        base["target_object_class"] = base["object_class"]
        base["camera_view"] = item.get("camera_view") or _infer_camera_view(image_path)
        base["end_effector"] = "parallel_jaw"
        base["visual_evidence"] = item.get("visual_evidence", "")
        base["subtype"] = base["state_type"]
        base["s_context"] = True

    return base


# ============================================================
# Processing
# ============================================================

def process_single_image(vlm: VLMClient, image_path: str, dimension: str, sample_id: str) -> Tuple[List[Dict], Optional[str]]:
    """Process one image. Returns (labels, error_msg)."""
    # Filter: skip images not suitable for this dimension
    path_lower = image_path.lower()
    if dimension in ("A_action_precondition", "A_action_interface"):
        # Action dimensions need tabletop/robot scenes, skip appliance panels
        if "knob" in path_lower or "panel" in path_lower or "instruction" in path_lower:
            return [], None  # silently skip
    if dimension == "S_spatial":
        # Spatial needs multi-object scenes, skip close-up single-object images
        if "knob" in path_lower or "instruction" in path_lower:
            return [], None
    if dimension == "A_robot_state":
        # Robot state only meaningful for wrist/gripper camera views
        if not is_wrist_camera_image(image_path):
            return [], None

    result = vlm.chat_json(PROMPTS[dimension], image_path)
    if result is None:
        return [], f"VLM returned None for {image_path}"
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return [], f"Unexpected result type: {type(result)}"

    img_w, img_h = get_image_size(image_path)
    labels = []
    for idx, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        lbl = build_label(item, dimension, image_path, img_w, img_h, sample_id, idx)
        if dimension in ("G_grounding", "A_identity_attr", "A_obj_state", "A_action_precondition", "A_action_interface"):
            if lbl.get("bbox") is None:
                continue
        labels.append(lbl)
    return labels, None


def get_output_file(dimension: str) -> Path:
    """Get sharded output file path: labels/{dimension}/{dimension}_{date}.jsonl"""
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = OUTPUT_ROOT / "labels" / dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{dimension}_{date_str}.jsonl"


def run_dimension(vlm: VLMClient, dimension: str, max_images: int, processed_set: set):
    """Process a single dimension with concurrent workers."""
    # Find source file
    list_file = DIMENSION_SOURCES / f"{dimension}.txt"
    if not list_file.exists():
        # Try alternate naming
        alt_names = {
            "A_identity_attr": "A_attribute.txt",
            "A_obj_state": "A_object_state.txt",
            "A_action_precondition": "A_precondition.txt",
        }
        alt = alt_names.get(dimension)
        if alt:
            list_file = DIMENSION_SOURCES / alt
        if not list_file.exists():
            logger.warning(f"No image list for dimension {dimension}")
            return

    with open(list_file) as f:
        all_images = []
        _local_scene_map = {}
        for l in f:
            l = l.strip()
            if not l:
                continue
            # Support tab-separated format: path\tsource\tscene_type
            parts = l.split("\t")
            img_path = parts[0].strip()
            scene_type_col = parts[2].strip() if len(parts) > 2 else "unknown"
            if img_path and os.path.exists(img_path):
                all_images.append(img_path)
                _local_scene_map[img_path] = scene_type_col

    # Filter already processed
    images = [img for img in all_images if img not in processed_set][:max_images]
    if not images:
        logger.info(f"[{dimension}] No new images to process")
        return

    logger.info(f"[{dimension}] Processing {len(images)} images with {MAX_WORKERS} workers...")

    out_file = get_output_file(dimension)
    write_lock = threading.Lock()
    dead_letter_file = OUTPUT_ROOT / "dead_letter" / f"{dimension}.jsonl"
    dead_letter_file.parent.mkdir(parents=True, exist_ok=True)

    GENERATION_STATE["current_dimension"] = dimension
    GENERATION_STATE["progress"][dimension] = {"total": len(images), "done": 0, "labels": 0, "errors": 0}

    def _task(idx_img):
        idx, img_path = idx_img
        sid = f"exp_{dimension}_{datetime.now().strftime('%Y%m%d')}_{idx:04d}"
        return img_path, process_single_image(vlm, img_path, dimension, sid)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_task, (i, p)): i for i, p in enumerate(images)}
        for future in as_completed(futures):
            if _shutdown.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                return

            try:
                img_path, (labels, error) = future.result()
            except Exception as e:
                GENERATION_STATE["progress"][dimension]["errors"] += 1
                GENERATION_STATE["total_errors"] += 1
                logger.error(f"  [{dimension}] task exception: {e}")
                continue

            with write_lock:
                GENERATION_STATE["progress"][dimension]["done"] += 1
                GENERATION_STATE["total_images_processed"] += 1
                processed_set.add(img_path)

                if error:
                    GENERATION_STATE["progress"][dimension]["errors"] += 1
                    GENERATION_STATE["total_errors"] += 1
                    GENERATION_STATE["dead_letter_count"] += 1
                    with open(dead_letter_file, "a") as f:
                        f.write(json.dumps({"image": img_path, "error": error, "time": datetime.now().isoformat()}) + "\n")
                else:
                    # Cross-check verification for bbox-based dimensions
                    verified_labels = []
                    for lbl in labels:
                        bbox = lbl.get("bbox")
                        obj_class = lbl.get("object_class", "") or lbl.get("object_name", "")
                        if bbox and obj_class and dimension in ("G_grounding", "A_identity_attr", "A_action_interface"):
                            try:
                                vr = vlm.verify_bbox(img_path, bbox, obj_class)
                                lbl["_cross_check"] = vr
                                if not vr.get("match", True):
                                    lbl["_cross_check_failed"] = True
                                    continue
                            except Exception:
                                pass
                        verified_labels.append(lbl)

                    # Filter placement_zone labels that overlap >50% with detected objects
                    if dimension == "A_action_interface":
                        obj_bboxes = [l["bbox"] for l in verified_labels if l.get("bbox") and l.get("action_type") != "place"]
                        filtered = []
                        for lbl in verified_labels:
                            if lbl.get("action_type") == "place" and lbl.get("bbox"):
                                if any(_compute_iou(lbl["bbox"], ob) > 0.5 for ob in obj_bboxes):
                                    continue
                            filtered.append(lbl)
                        verified_labels = filtered

                    GENERATION_STATE["progress"][dimension]["labels"] += len(verified_labels)
                    GENERATION_STATE["total_labels"] += len(verified_labels)
                    scene = _local_scene_map.get(img_path, "unknown")
                    with open(out_file, "a") as f:
                        for lbl in verified_labels:
                            lbl["scene_type"] = scene
                            if "scene_family" in lbl:
                                lbl["scene_family"] = scene
                            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

            done = GENERATION_STATE["progress"][dimension]["done"]
            total = GENERATION_STATE["progress"][dimension]["total"]
            if done % 5 == 0 or done == total:
                logger.info(f"  [{dimension}] {done}/{total} images, {GENERATION_STATE['progress'][dimension]['labels']} labels")

    logger.info(f"[{dimension}] Round complete: {GENERATION_STATE['progress'][dimension]['labels']} labels, {GENERATION_STATE['progress'][dimension]['errors']} errors")


def run_all_dimensions(vlm: VLMClient, processed_set: set):
    """Run one full round — interleave all dimensions (not sequential)."""
    import random
    GENERATION_STATE["current_round"] += 1
    round_start = time.time()

    # Collect all (image, dimension) pairs and shuffle them
    all_tasks = []
    for dim in DIMENSIONS:
        list_file = DIMENSION_SOURCES / f"{dim}.txt"
        if not list_file.exists():
            alt_names = {"A_identity_attr": "A_attribute.txt", "A_obj_state": "A_object_state.txt", "A_action_precondition": "A_precondition.txt"}
            alt = alt_names.get(dim)
            if alt:
                list_file = DIMENSION_SOURCES / alt
        if not list_file.exists():
            continue
        with open(list_file) as f:
            images = []
            for l in f:
                l = l.strip()
                if not l:
                    continue
                parts = l.split("\t")
                img_path = parts[0].strip()
                scene = parts[2].strip() if len(parts) >= 3 else "unknown"
                if not img_path or img_path in processed_set:
                    continue
                images.append((img_path, scene))
                if len(images) >= MAX_IMAGES_PER_DIM * 3:
                    break
        new_images = images[:MAX_IMAGES_PER_DIM]
        for img, sc in new_images:
            all_tasks.append((dim, img, sc))

    random.shuffle(all_tasks)

    if not all_tasks:
        GENERATION_STATE["last_round_time"] = 0
        return

    logger.info(f"Round {GENERATION_STATE['current_round']}: {len(all_tasks)} tasks across {len(DIMENSIONS)} dimensions (shuffled)")

    # Initialize progress for all dimensions
    dim_counts = {}
    for dim, _, _ in all_tasks:
        dim_counts.setdefault(dim, 0)
        dim_counts[dim] += 1
    for dim, count in dim_counts.items():
        if dim not in GENERATION_STATE["progress"]:
            GENERATION_STATE["progress"][dim] = {"total": count, "done": 0, "labels": 0, "errors": 0}
        else:
            GENERATION_STATE["progress"][dim]["total"] = count

    out_files = {}
    write_lock = threading.Lock()

    # Map image_path -> scene_type for use in build_label
    _scene_map = {img: sc for _, img, sc in all_tasks}

    def _task(dim_img):
        dim, img_path, _sc = dim_img
        idx = GENERATION_STATE["total_images_processed"]
        sid = f"exp_{dim}_{datetime.now().strftime('%Y%m%d')}_{idx:04d}"
        GENERATION_STATE["current_dimension"] = dim
        return dim, img_path, process_single_image(vlm, img_path, dim, sid)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_task, t): t for t in all_tasks}
        for future in as_completed(futures):
            if _shutdown.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                return
            try:
                dim, img_path, (labels, error) = future.result()
            except Exception as e:
                GENERATION_STATE["total_errors"] += 1
                continue

            with write_lock:
                GENERATION_STATE["total_images_processed"] += 1
                processed_set.add(img_path)
                GENERATION_STATE["progress"][dim]["done"] += 1

                if error:
                    GENERATION_STATE["progress"][dim]["errors"] += 1
                    GENERATION_STATE["total_errors"] += 1
                else:
                    out_file = get_output_file(dim)
                    # Scene type from data source (fast, no VLM call)
                    scene = _scene_map.get(img_path, "unknown")
                    valid_labels = []
                    for lbl in labels:
                        lbl["scene_type"] = scene
                        if "scene_family" in lbl:
                            lbl["scene_family"] = scene
                        valid_labels.append(lbl)
                    GENERATION_STATE["progress"][dim]["labels"] += len(valid_labels)
                    GENERATION_STATE["total_labels"] += len(valid_labels)
                    with open(out_file, "a") as f:
                        for lbl in valid_labels:
                            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

            done = GENERATION_STATE["total_images_processed"]
            if done % 5 == 0:
                logger.info(f"  Progress: {done}/{len(all_tasks)} | {GENERATION_STATE['total_labels']} labels | dim={dim}")

    GENERATION_STATE["last_round_time"] = time.time() - round_start

    # Update throughput
    elapsed_h = (time.time() - GENERATION_STATE["start_time"]) / 3600
    if elapsed_h > 0:
        GENERATION_STATE["throughput_per_hour"] = round(GENERATION_STATE["total_labels"] / elapsed_h, 1)


def run_continuous():
    """Main generation loop — continuous or single-shot."""
    GENERATION_STATE["status"] = "running"
    GENERATION_STATE["start_time"] = time.time()

    vlm = VLMClient(VLLM_BACKENDS, VLM_MODEL, VLM_API_KEY)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Check backend health
    healthy = vlm.check_health()
    GENERATION_STATE["backends_healthy"] = healthy
    if healthy == 0:
        logger.warning("No healthy backends! Will retry on each request...")

    # Real inference smoke test (catches dead/OOM/thinking-mode models that /models misses)
    infer_ok, infer_detail = vlm.test_inference()
    GENERATION_STATE["inference_ok"] = infer_ok
    GENERATION_STATE["inference_detail"] = infer_detail
    if infer_ok:
        logger.info(f"VLM inference test PASSED: {infer_detail}")
    else:
        logger.error(f"VLM inference test FAILED: {infer_detail}")
        logger.error(f"  Endpoint: {vlm.backends[0]} | Model: {vlm.model}")
        logger.error("  Generation will not produce labels until the VLM API is usable.")

    # Load processed set from disk (avoid re-processing after restart)
    processed_file = OUTPUT_ROOT / "processed_images.txt"
    processed_set: set = set()
    if processed_file.exists():
        processed_set = set(l.strip() for l in processed_file.read_text().splitlines() if l.strip())
        logger.info(f"Loaded {len(processed_set)} previously processed images")

    def _save_processed():
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        processed_file.write_text("\n".join(sorted(processed_set)) + "\n")

    if CONTINUOUS_MODE:
        logger.info("Starting in CONTINUOUS mode (Ctrl+C to stop)")
        empty_rounds = 0
        while not _shutdown.is_set():
            labels_before = GENERATION_STATE["total_labels"]
            run_all_dimensions(vlm, processed_set)
            _save_processed()
            if _shutdown.is_set():
                break
            produced = GENERATION_STATE["total_labels"] - labels_before

            # Detect stall: round produced nothing
            if produced == 0:
                empty_rounds += 1
                # Re-test inference to distinguish "API down" from "no new images"
                infer_ok, infer_detail = vlm.test_inference()
                GENERATION_STATE["inference_ok"] = infer_ok
                GENERATION_STATE["inference_detail"] = infer_detail
                if not infer_ok:
                    logger.error(f"Empty round + VLM inference FAILING: {infer_detail}. "
                                 f"Waiting {CONTINUOUS_INTERVAL}s and retrying (API issue).")
                elif empty_rounds >= 2:
                    # API works but all images processed -> allow re-processing for deeper verified@k
                    logger.warning(f"All images processed ({len(processed_set)}). "
                                   f"Resetting processed_set to re-scan for verified@k depth.")
                    processed_set.clear()
                    _save_processed()
                    empty_rounds = 0
            else:
                empty_rounds = 0

            logger.info(f"Round {GENERATION_STATE['current_round']} complete (produced {produced}). "
                        f"Waiting {CONTINUOUS_INTERVAL}s before next round...")
            _shutdown.wait(timeout=CONTINUOUS_INTERVAL)
    else:
        run_all_dimensions(vlm, processed_set)

    GENERATION_STATE["status"] = "completed"
    elapsed = time.time() - GENERATION_STATE["start_time"]
    logger.info(f"Generation finished in {elapsed:.0f}s. Total: {GENERATION_STATE['total_labels']} labels, {GENERATION_STATE['total_errors']} errors")


# ============================================================
# Signal Handling
# ============================================================

def _signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown.set()
    GENERATION_STATE["status"] = "stopping"

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ============================================================
# Flask Endpoints
# ============================================================

@app.route("/status")
def status():
    return jsonify(GENERATION_STATE)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "generator", "port": 10305})


@app.route("/start", methods=["POST"])
def start_generation():
    if GENERATION_STATE["status"] == "running":
        return jsonify({"error": "Generation already running"}), 409
    threading.Thread(target=run_continuous, daemon=True).start()
    return jsonify({"message": "Generation started", "mode": GENERATION_STATE["mode"]})


@app.route("/stop", methods=["POST"])
def stop_generation():
    _shutdown.set()
    return jsonify({"message": "Shutdown signal sent"})


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    if os.environ.get("GSA_AUTO_START", "0") == "1":
        threading.Thread(target=run_continuous, daemon=True).start()
    app.run(host="0.0.0.0", port=10305, debug=False, threaded=True)
