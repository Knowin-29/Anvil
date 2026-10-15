"""飞轮运行日志 — 带日期的结构化日志，支持 GUI 实时监控。"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


class FlywheelLogger:
    """飞轮运行日志管理器。

    日志文件按日期命名: logs/flywheel_YYYYMMDD_HHMMSS.jsonl
    每行一条 JSON 事件，GUI 可实时 tail 读取。
    """

    def __init__(self, log_dir: str = "logs/flywheel"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"flywheel_{timestamp}.jsonl")
        self._start_time = time.time()
        self._sample_count = 0
        self._annotation_count = 0

        self._write_event("run_start", {
            "timestamp": timestamp,
            "log_file": self.log_file,
        })

    def log_sample(self, sample_id: str, source_type: str, image_path: Optional[str] = None):
        """记录样本处理。"""
        self._sample_count += 1
        self._write_event("sample_processed", {
            "sample_id": sample_id,
            "source_type": source_type,
            "image_path": image_path,
            "total_samples": self._sample_count,
        })

    def log_annotations(self, sample_id: str, annotations: list, capabilities: list):
        """记录标注生成。"""
        n = len(annotations)
        self._annotation_count += n
        self._write_event("annotations_generated", {
            "sample_id": sample_id,
            "count": n,
            "capabilities": capabilities,
            "total_annotations": self._annotation_count,
        })

    def log_verifier(self, sample_id: str, scores: Dict[str, float]):
        """记录 Verifier 评分。"""
        self._write_event("verifier_scored", {
            "sample_id": sample_id,
            "scores": scores,
        })

    def log_error(self, message: str, details: Optional[Dict] = None):
        """记录错误。"""
        self._write_event("error", {
            "message": message,
            "details": details or {},
        })

    def log_progress(self, current: int, total: int, capability_stats: Dict[str, int]):
        """记录进度（每 N 个样本）。"""
        elapsed = time.time() - self._start_time
        speed = current / elapsed if elapsed > 0 else 0
        self._write_event("progress", {
            "current": current,
            "total": total,
            "elapsed_seconds": round(elapsed, 1),
            "speed_samples_per_sec": round(speed, 3),
            "capability_stats": capability_stats,
        })

    def log_complete(self, result: Dict[str, Any]):
        """记录运行完成。"""
        elapsed = time.time() - self._start_time
        self._write_event("run_complete", {
            "elapsed_seconds": round(elapsed, 1),
            "total_samples": self._sample_count,
            "total_annotations": self._annotation_count,
            "result": result,
        })

    def _write_event(self, event_type: str, data: Dict[str, Any]):
        """写入一条日志事件。"""
        event = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def list_logs(log_dir: str = "logs/flywheel") -> list:
        """列出所有日志文件（按时间倒序）。"""
        if not os.path.isdir(log_dir):
            return []
        files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
        return sorted(files, reverse=True)

    @staticmethod
    def read_log(log_path: str, tail: int = 50) -> list:
        """读取日志最后 N 行。"""
        if not os.path.exists(log_path):
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = lines[-tail:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        return events
