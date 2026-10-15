"""Phase E: Mock hardware acceptance — start service, run grasp decision, generate report."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "hardware_acceptance.md"
RESULT_PATH = PROJECT_ROOT / "reports" / "hardware_acceptance.json"
SERVICE_URL = os.getenv("REX_SERVICE_URL", "http://127.0.0.1:8101")
TEST_IMAGE = os.getenv(
    "REX_TEST_IMAGE",
    "/path/to/Rex-Omni/tutorials/detection_example/test_images/cafe.jpg",
)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "services.app"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ok = False
        for _ in range(60):
            try:
                r = requests.get(f"{SERVICE_URL}/health", timeout=2)
                if r.ok:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ok:
            print("[ERROR] service health check failed after 60 retries")
            return 1

        payload = {
            "image_path": TEST_IMAGE,
            "target_object": "cup",
            "candidate_objects": ["laptop", "person"],
            "backend": "transformers",
            "robot_exec_mode": "mock",
        }
        r = requests.post(f"{SERVICE_URL}/grasp/optimize", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        output = {
            "timestamp": datetime.now().isoformat(),
            "service_url": SERVICE_URL,
            "request": payload,
            "response": data,
            "success": True,
        }
        RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        report = f"""# Phase E Hardware Mock Acceptance

- 时间: {output["timestamp"]}
- 服务地址: `{SERVICE_URL}`
- 测试图: `{TEST_IMAGE}`
- 健康检查: `PASS`
- 抓取决策链路: `PASS`

## 请求
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## 响应（截断）
```json
{json.dumps(data, ensure_ascii=False, indent=2)[:4000]}
```
"""
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"[INFO] report: {REPORT_PATH}")
        print(f"[INFO] result: {RESULT_PATH}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
