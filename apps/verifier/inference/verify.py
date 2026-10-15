"""Verifier Inference — 加载训练好的 Verifier 模型进行标注质量评分。"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger("verifier.inference")

SYSTEM_PROMPT = """You are a quality verifier for embodied AI training data. Given an annotation, evaluate its quality and return a JSON object with:
- "score": float between 0 and 1 (1 = perfect quality)
- "reason": brief explanation of the score
- "issues": list of specific issues found (empty if none)"""


class VerifierInference:
    """Verifier 推理引擎。"""

    def __init__(
        self,
        base_model_path: str = "/path/to/environment/hf_models/Qwen/Qwen3-32B",
        lora_path: Optional[str] = None,
        device_map: str = "auto",
    ):
        logger.info(f"Loading base model: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        logger.info("Verifier model loaded")

    def build_prompt(self, annotation: Dict) -> str:
        """构建验证 prompt。"""
        cap = annotation.get("capability_tag", "")
        sub_tag = annotation.get("sub_tag", "")
        content = annotation.get("content", {})
        content_str = json.dumps(content, ensure_ascii=False, indent=2)[:1500]

        user_msg = f"""Evaluate the quality of this {cap} ({sub_tag}) annotation:

```json
{content_str}
```

Consider:
1. Is the content factually plausible for the described scene?
2. Is the format correct and complete?
3. Are there any logical inconsistencies?
4. Is the difficulty level appropriate?

Return your evaluation as JSON: {{"score": 0.0-1.0, "reason": "...", "issues": [...]}}"""
        return user_msg

    @torch.no_grad()
    def verify(self, annotation: Dict, max_tokens: int = 512) -> Dict:
        """对单条标注进行质量评分。"""
        user_msg = self.build_prompt(annotation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Parse response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            return {
                "score": float(result.get("score", 0.5)),
                "reason": result.get("reason", ""),
                "issues": result.get("issues", []),
                "raw_response": response[:200],
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "score": 0.5,
                "reason": "Failed to parse verifier output",
                "issues": ["parse_error"],
                "raw_response": response[:200],
            }

    def verify_batch(self, annotations: List[Dict], max_tokens: int = 512) -> List[Dict]:
        """批量验证。"""
        results = []
        for ann in annotations:
            result = self.verify(ann, max_tokens=max_tokens)
            results.append(result)
        return results


def main():
    """CLI 入口：对 JSONL 文件进行批量验证。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Verifier Inference")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output scores JSONL file")
    parser.add_argument("--model", default="/path/to/environment/hf_models/Qwen/Qwen3-32B")
    parser.add_argument("--lora", default=None, help="LoRA adapter path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    verifier = VerifierInference(
        base_model_path=args.model,
        lora_path=args.lora,
        device_map=args.device_map,
    )

    annotations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    if args.max_samples:
        annotations = annotations[:args.max_samples]

    logger.info(f"Verifying {len(annotations)} annotations...")

    with open(args.output, "w") as out_f:
        for i, ann in enumerate(annotations):
            result = verifier.verify(ann)
            record = {
                "sample_id": ann.get("sample_id", ""),
                "capability_tag": ann.get("capability_tag", ""),
                "sub_tag": ann.get("sub_tag", ""),
                "verifier_score": result["score"],
                "verifier_reason": result["reason"],
                "verifier_issues": result["issues"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(annotations)}] avg_score={sum(r.get('score',0.5) for r in [result])/1:.2f}")

    logger.info(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
