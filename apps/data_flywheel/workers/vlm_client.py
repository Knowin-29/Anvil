"""VLMClient — 共享同步 VLM 调用客户端。

所有 Worker 通过此客户端与 VLM API 交互。
支持: 本地模型 / 火山引擎 MLLM API / 任意 OpenAI 兼容 API。
支持: 图片编码、自动重试、JSON 解析容错。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

logger = logging.getLogger("flywheel.vlm_client")

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)(?:```|\Z)", re.DOTALL)
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class VLMClient:
    """同步 VLM 调用客户端，所有 Worker 共享。

    支持两种模式:
    1. 本地模式: base_url="http://localhost:10301", 无需 api_key
    2. 远程 API 模式: base_url="https://xxx/v1", api_key="sk-xxx"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:10301",
        model: str = "qwen2.5-vl-32b",
        api_key: Optional[str] = None,
        max_image_size: int = 960,
        timeout: int = 300,
        max_retries: int = 3,
    ):
        self.base_url = self.expand_env(base_url).rstrip("/")
        self.model = self.expand_env(model)
        self.api_key = self.expand_env(api_key or "") or None
        self.max_image_size = max_image_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._thread_local = threading.local()


    @staticmethod
    def expand_env(value: Any) -> Any:
        """Expand ${VAR} and ${VAR:-default} placeholders in config strings."""
        if not isinstance(value, str):
            return value

        def repl(match: re.Match) -> str:
            name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(name, default)

        return _ENV_RE.sub(repl, value)

    def _get_session(self) -> requests.Session:
        """Return a per-thread Session; requests.Session is not thread-safe."""
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            if self.api_key:
                session.headers["Authorization"] = f"Bearer {self.api_key}"
            self._thread_local.session = session
        return session

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _encode_image(self, image_path: str) -> str:
        """将本地图片编码为 base64 data URL。"""
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > self.max_image_size:
            ratio = self.max_image_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _encode_pil_image(self, img: Image.Image) -> str:
        """将 PIL Image 编码为 base64 data URL。"""
        img = img.convert("RGB")
        if max(img.size) > self.max_image_size:
            ratio = self.max_image_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def denormalize_bbox(bbox: List, img_w: int, img_h: int, norm_range: int = 1000) -> List[int]:
        """Convert Qwen VL normalized coords (0-1000) to pixel coords."""
        if not bbox or len(bbox) != 4:
            return bbox
        return [
            int(bbox[0] / norm_range * img_w),
            int(bbox[1] / norm_range * img_h),
            int(bbox[2] / norm_range * img_w),
            int(bbox[3] / norm_range * img_h),
        ]

    @staticmethod
    def denormalize_point(point: List, img_w: int, img_h: int, norm_range: int = 1000) -> List[int]:
        """Convert Qwen VL normalized point (0-1000) to pixel coords."""
        if not point or len(point) != 2:
            return point
        return [
            int(point[0] / norm_range * img_w),
            int(point[1] / norm_range * img_h),
        ]

    def query(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        pil_image: Optional[Image.Image] = None,
        pil_images: Optional[List[Image.Image]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.4,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Alias for chat() — used by fine-grained A pipelines."""
        return self.chat(
            prompt,
            image_path=image_path,
            image_paths=image_paths,
            pil_image=pil_image,
            pil_images=pil_images,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def chat(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        pil_image: Optional[Image.Image] = None,
        pil_images: Optional[List[Image.Image]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        require_complete: bool = False,
    ) -> str:
        """发送图片+prompt，返回文本响应。"""
        content: List[Dict[str, Any]] = []

        all_image_paths: List[str] = []
        if image_path:
            all_image_paths.append(image_path)
        if image_paths:
            all_image_paths.extend(image_paths)

        all_pil_images: List[Image.Image] = []
        if pil_image is not None:
            all_pil_images.append(pil_image)
        if pil_images:
            all_pil_images.extend(pil_images)

        for path in all_image_paths:
            data_url = self._encode_image(path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        for img in all_pil_images:
            data_url = self._encode_pil_image(img)
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        content.append({"type": "text", "text": prompt})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = self._chat_completions_url()
        self._thread_local.response_metadata = {}
        self._thread_local.attempt_receipts = []
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                resp = self._get_session().post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                self._thread_local.response_metadata = {
                    "model": data.get("model", self.model), "usage": data.get("usage"),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "inference": data.get("inference"),
                }
                self._thread_local.attempt_receipts.append({
                    **self._thread_local.response_metadata, "attempt": attempt + 1,
                    "seconds": time.monotonic() - started,
                    "content": data["choices"][0]["message"].get("content"),
                })
                if require_complete and data["choices"][0].get("finish_reason") != "stop":
                    raise ValueError("incomplete model response")
                msg = data["choices"][0]["message"]
                # Handle thinking-mode models: content may be null, use reasoning_content
                content_text = msg.get("content")
                if content_text is None and not require_complete:
                    content_text = msg.get("reasoning_content", "")
                return content_text or ""
            except Exception as e:
                if len(self._thread_local.attempt_receipts) <= attempt:
                    self._thread_local.attempt_receipts.append({"attempt": attempt + 1,
                        "seconds": time.monotonic() - started, "error_type": type(e).__name__})
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"VLM call failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"VLM call failed after {self.max_retries+1} attempts: {e}")
                    raise

    def query(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        pil_image: Optional[Image.Image] = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Compatibility wrapper for older GSA pipelines."""
        raw = self.chat(
            prompt=prompt,
            image_path=image_path,
            pil_image=pil_image,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt or "Return valid JSON only.",
        )
        parsed = self._parse_json(raw)
        if parsed:
            return json.dumps(parsed, ensure_ascii=False)
        return raw

    def chat_json(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        pil_image: Optional[Image.Image] = None,
        pil_images: Optional[List[Image.Image]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.4,
        system_prompt: Optional[str] = None,
    ) -> Union[Dict, List]:
        """发送并解析 JSON 响应（容错：支持 thinking-mode + markdown code block）。"""
        raw = self.chat(prompt, image_path=image_path, image_paths=image_paths,
                        pil_image=pil_image, pil_images=pil_images,
                        max_tokens=max_tokens, temperature=temperature,
                        system_prompt=system_prompt)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> Union[Dict, List]:
        """解析 JSON，支持 thinking-mode reasoning 输出 + markdown code block + 截断修复。"""
        text = text.strip()
        if not text:
            return {}

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取
        match = _JSON_BLOCK_RE.search(text)
        if match:
            block = match.group(1).strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                fixed = VLMClient._fix_truncated_json(block)
                if fixed is not None:
                    return fixed

        # 从 thinking-mode 输出中提取最后一个完整 JSON 对象/数组
        # 策略: 从后往前找最后一个 } 或 ]，然后向前匹配
        json_candidates = []
        # 找所有 {...} 块
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{' and depth == 0:
                start = i
                depth = 1
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    json_candidates.append(text[start:i+1])
                    start = -1

        # 找所有 [...] 块
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '[' and depth == 0:
                start = i
                depth = 1
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0 and start >= 0:
                    json_candidates.append(text[start:i+1])
                    start = -1

        # 从后往前尝试解析（最后出现的 JSON 通常是最终答案）
        for candidate in reversed(json_candidates):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue

        # 尝试修复截断
        for i, ch in enumerate(text):
            if ch in "[{":
                remaining = text[i:]
                fixed = VLMClient._fix_truncated_json(remaining)
                if fixed is not None:
                    return fixed
                break

        logger.warning("json_parse_failed_or_truncated")
        return {}

    @staticmethod
    def _fix_truncated_json(text: str) -> Optional[Union[Dict, List]]:
        """尝试修复被截断的 JSON（VLM max_tokens 不够时常见）。"""
        text = text.strip()
        if not text:
            return None

        # 策略1: 如果是数组，尝试在最后一个完整对象后截断
        if text.startswith("["):
            # 找到最后一个 }, 的位置
            last_complete = -1
            depth = 0
            in_string = False
            escape_next = False
            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        last_complete = i

            if last_complete > 0:
                truncated = text[:last_complete + 1] + "]"
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass

        # 策略2: 如果是对象，尝试补全闭合括号
        if text.startswith("{"):
            open_braces = text.count("{") - text.count("}")
            open_brackets = text.count("[") - text.count("]")
            # 去掉最后不完整的 key-value
            last_comma = text.rfind(",")
            if last_comma > 0:
                candidate = text[:last_comma]
                candidate += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        return None

    def get_image_size(self, image_path: str) -> tuple:
        """获取图片原始尺寸 (width, height)。"""
        img = Image.open(image_path)
        return img.size

    def chat_batch(
        self,
        requests: List[Dict[str, Any]],
        max_workers: int = 4,
    ) -> List[str]:
        """并发执行多个 chat 请求。
        
        Args:
            requests: [{"prompt": str, "image_path": str, "temperature": float}, ...]
            max_workers: 并发线程数
            
        Returns:
            对应顺序的响应列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(requests)

        def _call(idx: int, req: Dict) -> tuple:
            resp = self.chat(
                prompt=req["prompt"],
                image_path=req.get("image_path"),
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 4000),
            )
            return idx, resp

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call, i, r): i for i, r in enumerate(requests)}
            for future in as_completed(futures):
                try:
                    idx, resp = future.result()
                    results[idx] = resp
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"Batch request {idx} failed: {e}")
                    results[idx] = ""

        return results
