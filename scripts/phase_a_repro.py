"""Phase A: Reproduce the official Rex-Omni detection example and generate a report."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REX_REPO = Path(os.getenv("REX_OMNI_REPO", "/path/to/Rex-Omni"))
REPORT_PATH = PROJECT_ROOT / "reports" / "repro_report.md"
RESULT_PATH = PROJECT_ROOT / "reports" / "repro_result.json"


def run(cmd: str, check: bool = False) -> tuple:
    """Run a shell command and return (stdout+stderr, returncode)."""
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if check and p.returncode != 0:
        print(f"[WARN] command returned {p.returncode}: {cmd}")
    return out, p.returncode


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    gpu_info, _ = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    python_info, _ = run(f"{py} -V")
    torch_info, _ = run(f'{py} -c "import torch; print(f\'torch={{torch.__version__}}, cuda={{torch.cuda.is_available()}}, gpus={{torch.cuda.device_count()}}\')"')
    rex_import, rc = run(f'{py} -c "import rex_omni; print(\'rex_omni_import_ok\')"')
    if rc != 0:
        rex_import = f"FAILED (rc={rc}): {rex_import}"

    test_cmd = f'cd "{REX_REPO}" && CUDA_VISIBLE_DEVICES=0 {py} tutorials/detection_example/detection_example.py'
    start = time.time()
    test_out, test_rc = run(test_cmd)
    elapsed = round(time.time() - start, 2)

    output_image = REX_REPO / "tutorials/detection_example/test_images/cafe_visualize.jpg"
    success = output_image.exists() and test_rc == 0

    result = {
        "timestamp": datetime.now().isoformat(),
        "python": python_info,
        "torch": torch_info,
        "gpu": gpu_info,
        "rex_import": rex_import,
        "sample_elapsed_sec": elapsed,
        "sample_return_code": test_rc,
        "sample_success": success,
        "sample_output_image": str(output_image),
        "sample_log_tail": "\n".join(test_out.splitlines()[-40:]),
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Phase A Repro Report

- 时间: {result["timestamp"]}
- Python: `{python_info}`
- Torch: `{torch_info}`
- GPU:
```
{gpu_info}
```
- `rex_omni` 导入状态: `{rex_import}`
- 官方样例命令: `CUDA_VISIBLE_DEVICES=0 python tutorials/detection_example/detection_example.py`
- 运行耗时: `{elapsed}s`
- 返回码: `{test_rc}`
- 样例成功: `{success}`
- 输出图路径: `{output_image}`

## 样例日志尾部

```
{result["sample_log_tail"]}
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[INFO] report: {REPORT_PATH}")
    print(f"[INFO] result: {RESULT_PATH}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
