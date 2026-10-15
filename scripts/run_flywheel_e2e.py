"""端到端飞轮验收 — 启服务 → 跑飞轮 → 评估 → 检查报告。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_URL = os.getenv("REX_SERVICE_URL", "http://127.0.0.1:8101")
TEST_IMAGE = os.getenv(
    "REX_TEST_IMAGE",
    "/path/to/Rex-Omni/tutorials/detection_example/test_images/cafe.jpg",
)


def ensure_demo_images() -> str:
    demo_dir = str(PROJECT_ROOT / "data" / "input" / "images")
    os.makedirs(demo_dir, exist_ok=True)
    target = os.path.join(demo_dir, "cafe.jpg")
    if not os.path.exists(target) and os.path.exists(TEST_IMAGE):
        import shutil
        shutil.copy2(TEST_IMAGE, target)
    return demo_dir


def wait_for_service(url: str, retries: int = 60) -> bool:
    for _ in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> int:
    demo_dir = ensure_demo_images()

    # 启动服务
    print("[INFO] 启动服务...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "services.app"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        if not wait_for_service(SERVICE_URL):
            print("[ERROR] 服务启动失败")
            return 1
        print("[INFO] 服务就绪")

        # 跑飞轮
        print("[INFO] 触发飞轮...")
        r = requests.post(f"{SERVICE_URL}/flywheel/run", json={
            "image_dir": demo_dir,
            "rounds": 1,
            "max_images": 3,
            "augment_per_image": 1,
            "filter_adjectives": True,
            "backend": "transformers",
        }, timeout=1200)
        r.raise_for_status()
        flywheel_data = r.json()
        print(f"[INFO] 飞轮结果: ok={flywheel_data.get('ok')}")

        # 检查状态
        r = requests.get(f"{SERVICE_URL}/flywheel/status", timeout=30)
        r.raise_for_status()
        status_data = r.json()
        print(f"[INFO] 飞轮状态: {json.dumps(status_data.get('data', {}).get('status', 'unknown'))}")

        # 检查输出文件
        output_root = str(PROJECT_ROOT / "data" / "output" / "flywheel")
        summary_path = os.path.join(output_root, "flywheel_summary.json")
        checks = {
            "summary": os.path.exists(summary_path),
            "round_0_grounding": os.path.exists(os.path.join(output_root, "round_0", "grounding.jsonl")),
            "round_0_referring": os.path.exists(os.path.join(output_root, "round_0", "referring.jsonl")),
            "round_0_pointing": os.path.exists(os.path.join(output_root, "round_0", "pointing.jsonl")),
            "round_0_export": os.path.exists(os.path.join(output_root, "round_0", "train_export.jsonl")),
            "round_0_eval": os.path.exists(os.path.join(output_root, "round_0", "eval_report.json")),
            "round_0_hard": os.path.exists(os.path.join(output_root, "round_0", "hard_samples.jsonl")),
        }

        all_pass = all(checks.values())
        report_path = PROJECT_ROOT / "reports" / "flywheel_acceptance.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "checks": checks,
            "all_pass": all_pass,
            "flywheel_response": flywheel_data,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n===== 飞轮验收 {'PASS' if all_pass else 'FAIL'} =====")
        for k, v in checks.items():
            print(f"  {k}: {'✓' if v else '✗'}")
        print(f"\n报告: {report_path}")

        return 0 if all_pass else 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
