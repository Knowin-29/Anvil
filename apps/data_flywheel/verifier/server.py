"""Verifier 推理服务 — FastAPI + vllm 部署。

POST /verify: 单条验证
POST /verify/batch: 批量验证
GET /health: 健康检查

启动: python -m apps.data_flywheel.verifier.server --model checkpoints/verifier_grpo_v0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("flywheel.verifier.server")

_SCORE_PATTERN = re.compile(r"\[Score:\s*([\d.]+)\]")


class VerifierInference:
    """Verifier 推理引擎，支持 vllm / transformers / API 三种后端。"""

    def __init__(
        self,
        model_path: str = "checkpoints/verifier_grpo_v0",
        backend: str = "vllm",
        cache_dir: str = ".cache/verifier",
    ):
        self.model_path = model_path
        self.backend = backend
        self.cache_dir = cache_dir
        self._model = None
        self._cache: Dict[str, Dict] = {}

        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self):
        cache_file = os.path.join(self.cache_dir, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached verifier results")
            except Exception:
                self._cache = {}

    def _save_cache(self):
        cache_file = os.path.join(self.cache_dir, "cache.json")
        with open(cache_file, "w") as f:
            json.dump(self._cache, f)

    def _cache_key(self, capability_tag: str, input_text: str) -> str:
        content = f"{capability_tag}:{input_text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_model(self):
        if self._model is not None:
            return

        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
                self._model = LLM(model=self.model_path, trust_remote_code=True)
                self._sampling_params = SamplingParams(
                    temperature=0.1, max_tokens=256, top_p=0.9
                )
                logger.info(f"Loaded vllm model: {self.model_path}")
                return
            except ImportError:
                logger.warning("vllm not available, falling back to transformers")
                self.backend = "transformers"

        if self.backend == "transformers":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True, device_map="auto"
                )
                logger.info(f"Loaded transformers model: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._model = "fallback"

    def verify(self, capability_tag: str, input_text: str, output_text: str) -> Dict[str, Any]:
        """验证单条标注。"""
        cache_key = self._cache_key(capability_tag, input_text + output_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"[Capability: {capability_tag}]\n"
            f"Input: {input_text[:500]}\n"
            f"Output: {output_text[:500]}\n"
            f"Rate the quality of this annotation (0-1):"
        )

        result = self._infer(prompt)
        self._cache[cache_key] = result

        if len(self._cache) % 100 == 0:
            self._save_cache()

        return result

    def verify_batch(self, items: List[Dict]) -> List[Dict]:
        """批量验证。"""
        results: List[Dict] = []
        for item in items:
            r = self.verify(
                capability_tag=item.get("capability_tag", ""),
                input_text=item.get("input", ""),
                output_text=item.get("output", ""),
            )
            results.append(r)
        return results

    def _infer(self, prompt: str) -> Dict[str, Any]:
        """执行推理。"""
        self._load_model()

        if self._model == "fallback" or self._model is None:
            return self._fallback_verify(prompt)

        if self.backend == "vllm":
            outputs = self._model.generate([prompt], self._sampling_params)
            text = outputs[0].outputs[0].text
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            output_ids = self._model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            text = self._tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return self._parse_output(text)

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """解析模型输出。"""
        match = _SCORE_PATTERN.search(text)
        score = float(match.group(1)) if match else 0.5
        score = max(0.0, min(1.0, score))

        rationale = text
        if match:
            rationale = text[match.end():].strip()

        return {
            "score": score,
            "rationale": rationale,
            "raw_output": text,
        }

    def _fallback_verify(self, prompt: str) -> Dict[str, Any]:
        """降级: 基于规则的简单验证。"""
        return {
            "score": 0.5,
            "rationale": "Fallback mode: model not available, returning neutral score.",
            "raw_output": "",
            "fallback": True,
        }


def create_app(model_path: str = "checkpoints/verifier_grpo_v0", backend: str = "vllm"):
    """创建 FastAPI 应用。"""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        logger.error("fastapi not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="Rex-Embodied Verifier", version="0.1.0")
    verifier = VerifierInference(model_path=model_path, backend=backend)

    class VerifyRequest(BaseModel):
        capability_tag: str
        input: str
        output: str

    class BatchVerifyRequest(BaseModel):
        items: List[Dict[str, str]]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path, "backend": backend}

    @app.post("/verify")
    def verify(req: VerifyRequest):
        result = verifier.verify(req.capability_tag, req.input, req.output)
        return result

    @app.post("/verify/batch")
    def verify_batch(req: BatchVerifyRequest):
        results = verifier.verify_batch(req.items)
        return {"results": results}

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verifier 推理服务")
    parser.add_argument("--model", type=str, default="checkpoints/verifier_grpo_v0")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    app = create_app(model_path=args.model, backend=args.backend)
    if app is None:
        return

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
