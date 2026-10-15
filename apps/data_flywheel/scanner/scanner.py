"""Quality Scanner — 后台守护进程，持续扫描已生成标注并调用多模型评估。

核心流程:
1. 扫描 annotations_dir 下所有 *.jsonl
2. 对比 *.scores.jsonl (sidecar) 确定未评估行
3. 逐批读取未评估标注，发给 EnsembleJudge
4. 将评分结果 append 到 sidecar
5. 循环（可设置 scan_interval 秒轮询）

用法:
    python -m apps.data_flywheel.scanner.scanner --config scripts/configs/scanner.yaml
    python -m apps.data_flywheel.scanner.scanner --annotations-dir data/output/flywheel/v3_run_1778980547/annotations
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import ScannerConfig, JudgeEndpoint
from .judges import EnsembleJudge

logger = logging.getLogger("scanner")


class QualityScanner:
    """持续扫描已生成数据并进行质量评估的守护进程。"""

    def __init__(self, config: ScannerConfig):
        self.config = config
        self.judge = EnsembleJudge(config.judges, config.max_image_size)
        self._running = False
        self._stats = {
            "total_scanned": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_delete": 0,
        }
        self._image_map: Dict[str, str] = {}
        self._load_image_map()

    def _load_image_map(self):
        """从 manifest.jsonl 加载 sample_id → image_path 映射。"""
        ann_dir = Path(self.config.annotations_dir)
        manifest_path = ann_dir.parent / "manifest.jsonl"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}, images will be unavailable!")
            return
        count = 0
        with open(manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id", "")
                    img = obj.get("image_path", "")
                    if sid and img:
                        self._image_map[sid] = img
                        count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded image map: {count} entries from manifest")

    async def start(self):
        """启动扫描循环。"""
        self._running = True
        logger.info(f"Scanner started — dir={self.config.annotations_dir}, "
                    f"interval={self.config.scan_interval_seconds}s, "
                    f"judges={len(self.config.judges)}")

        while self._running:
            try:
                scanned = await self._scan_once()
                if scanned == 0:
                    logger.info(f"No new annotations to scan. Sleeping {self.config.scan_interval_seconds}s...")
                else:
                    logger.info(f"Scanned {scanned} annotations this round. Stats: {self._stats}")
            except Exception as e:
                logger.error(f"Scan round error: {e}", exc_info=True)

            await asyncio.sleep(self.config.scan_interval_seconds)

        await self.judge.close()
        logger.info("Scanner stopped.")

    def stop(self):
        """停止扫描。"""
        self._running = False

    async def _scan_once(self) -> int:
        """执行一轮完整扫描，返回本轮评估的标注数。"""
        ann_dir = Path(self.config.annotations_dir)
        if not ann_dir.exists():
            logger.warning(f"Annotations dir not found: {ann_dir}")
            return 0

        total_scanned = 0
        jsonl_files = sorted(ann_dir.rglob("*.jsonl"))

        if self.config.capabilities:
            allowed = set(self.config.capabilities)
            jsonl_files = [f for f in jsonl_files if f.parent.name in allowed]

        # 排除 sidecar 文件本身
        jsonl_files = [f for f in jsonl_files if not f.name.endswith(self.config.score_suffix)]

        for jsonl_path in jsonl_files:
            n = await self._scan_file(jsonl_path)
            total_scanned += n

        return total_scanned

    async def _scan_file(self, jsonl_path: Path) -> int:
        """扫描单个 JSONL 文件，评估未打分的行。"""
        score_path = jsonl_path.with_suffix(self.config.score_suffix)

        scored_ids = self._load_scored_ids(score_path)
        pending = self._load_pending(jsonl_path, scored_ids)

        if not pending:
            return 0

        logger.info(f"Scanning {jsonl_path.name}: {len(pending)} pending / {len(scored_ids)} already scored")

        scanned = 0
        batch: List[Dict[str, Any]] = []
        line_indices: List[int] = []

        for line_idx, annotation in pending:
            batch.append(annotation)
            line_indices.append(line_idx)

            if len(batch) >= self.config.batch_size:
                await self._evaluate_and_write(batch, line_indices, score_path)
                scanned += len(batch)
                batch = []
                line_indices = []

        if batch:
            await self._evaluate_and_write(batch, line_indices, score_path)
            scanned += len(batch)

        return scanned

    def _load_scored_ids(self, score_path: Path) -> Set[str]:
        """加载已评分的 sample_id+line_idx 集合。"""
        scored = set()
        if not score_path.exists():
            return scored
        try:
            with open(score_path, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        key = f"{obj.get('sample_id', '')}_{obj.get('line_idx', 0)}"
                        scored.add(key)
        except Exception as e:
            logger.warning(f"Error reading score file {score_path}: {e}")
        return scored

    def _load_pending(self, jsonl_path: Path, scored_ids: Set[str]) -> List[tuple]:
        """加载未评分的标注行，并注入 image_path (从 manifest 映射)。"""
        pending = []
        max_samples = self.config.max_samples_per_file
        try:
            with open(jsonl_path, "r") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        ann = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = f"{ann.get('sample_id', '')}_{idx}"
                    if key not in scored_ids:
                        # 注入 image_path: 优先从 content 取，fallback 从 manifest 映射
                        if "image_path" not in ann.get("content", {}):
                            sid = ann.get("sample_id", "")
                            if sid in self._image_map:
                                ann.setdefault("content", {})["image_path"] = self._image_map[sid]
                        ann["_line_idx"] = idx
                        pending.append((idx, ann))
                    if max_samples > 0 and len(pending) >= max_samples:
                        break
        except Exception as e:
            logger.warning(f"Error reading {jsonl_path}: {e}")
        return pending

    async def _evaluate_and_write(
        self,
        batch: List[Dict[str, Any]],
        line_indices: List[int],
        score_path: Path,
    ):
        """评估一批标注并写入 sidecar。"""
        sem = asyncio.Semaphore(self.config.concurrency)

        async def _eval_one(ann: Dict, line_idx: int) -> Dict[str, Any]:
            async with sem:
                result = await self.judge.judge(ann)
                result["sample_id"] = ann.get("sample_id", "")
                result["capability_tag"] = ann.get("capability_tag", "")
                result["line_idx"] = line_idx
                result["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                return result

        tasks = [_eval_one(ann, idx) for ann, idx in zip(batch, line_indices)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        with open(score_path, "a") as f:
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Eval exception: {r}")
                    continue
                self._update_stats(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _update_stats(self, result: Dict[str, Any]):
        self._stats["total_scanned"] += 1
        score = result.get("ensemble_score", 0.5)
        if score >= self.config.pass_threshold:
            self._stats["total_passed"] += 1
        else:
            self._stats["total_failed"] += 1
        if result.get("action") == "delete" or score < self.config.delete_threshold:
            self._stats["total_delete"] += 1

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)


def load_config_from_yaml(path: str) -> ScannerConfig:
    """从 YAML 加载配置。"""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    judges = []
    for j in data.get("judges", []):
        judges.append(JudgeEndpoint(**j))

    return ScannerConfig(
        annotations_dir=data.get("annotations_dir", ScannerConfig.annotations_dir),
        score_suffix=data.get("score_suffix", ScannerConfig.score_suffix),
        judges=judges if judges else ScannerConfig().judges,
        batch_size=data.get("batch_size", 8),
        concurrency=data.get("concurrency", 4),
        pass_threshold=data.get("pass_threshold", 0.6),
        delete_threshold=data.get("delete_threshold", 0.3),
        scan_interval_seconds=data.get("scan_interval_seconds", 60),
        max_image_size=data.get("max_image_size", 1280),
        capabilities=data.get("capabilities", None),
        resume=data.get("resume", True),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quality Scanner — 持续扫描标注数据并评估质量")
    parser.add_argument("--config", type=str, help="YAML 配置文件路径")
    parser.add_argument("--annotations-dir", type=str, help="标注目录 (覆盖配置)")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:10301", help="VLM 服务 URL")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-32b", help="模型名")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--interval", type=int, default=60, help="扫描间隔(秒)")
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    parser.add_argument("--delete-threshold", type=float, default=0.3)
    parser.add_argument("--once", action="store_true", help="只扫描一轮就退出")
    parser.add_argument("--capabilities", type=str, nargs="*", help="只扫描指定能力维度")
    parser.add_argument("--max-samples", type=int, default=0, help="每个文件最多扫描N条 (0=不限)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = ScannerConfig(
            judges=[JudgeEndpoint(name=args.model, base_url=args.vlm_url, model=args.model)],
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            scan_interval_seconds=args.interval,
            pass_threshold=args.pass_threshold,
            delete_threshold=args.delete_threshold,
            capabilities=args.capabilities,
        )

    if args.annotations_dir:
        config.annotations_dir = args.annotations_dir
    if args.max_samples:
        config.max_samples_per_file = args.max_samples

    scanner = QualityScanner(config)

    def _sighandler(signum, frame):
        logger.info("Received stop signal, shutting down...")
        scanner.stop()

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    async def _run():
        if args.once:
            n = await scanner._scan_once()
            logger.info(f"Single scan complete: {n} annotations evaluated. Stats: {scanner.stats}")
        else:
            await scanner.start()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
