import os
import sys
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from diskcache import Cache
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "output" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = Cache(str(CACHE_DIR))


def _cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    return f"{prefix}:{sha256(repr(sorted(payload.items())).encode('utf-8')).hexdigest()}"


def _ensure_rex_repo_importable() -> None:
    repo = resolve_rex_omni_repo()
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _ensure_project_root_importable() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@asynccontextmanager
async def lifespan(application: FastAPI):
    _ensure_project_root_importable()
    _ensure_rex_repo_importable()
    yield


app = FastAPI(title="rex-embodied service", version="0.2.0", lifespan=lifespan)

# Deferred imports so sys.path is ready when they execute at request time.

def _lazy_imports():
    from apps.auto_label_engine.engine import run_grounding_batch
    from apps.grasp_optimizer.optimizer import build_grasp_decision
    from services.runtime import get_runtime
    return run_grounding_batch, build_grasp_decision, get_runtime

from services.runtime import resolve_rex_omni_repo
from services.schemas import (
    ApiResponse,
    DetectionRequest,
    FlywheelEvaluateRequest,
    FlywheelRunRequest,
    GraspDecisionRequest,
    GroundingRequest,
    PointingRequest,
    SegmentRequest,
)


@app.get("/health", response_model=ApiResponse)
def health() -> ApiResponse:
    return ApiResponse(data={"status": "ok"})


@app.post("/infer/detection", response_model=ApiResponse)
def infer_detection(req: DetectionRequest) -> ApiResponse:
    payload = req.model_dump()
    key = _cache_key("detection", payload)
    if key in cache:
        return ApiResponse(data={"cached": True, "result": cache[key]})

    run_grounding_batch, build_grasp_decision, get_runtime = _lazy_imports()
    try:
        result = get_runtime().infer(
            image_path=req.image_path,
            task="detection",
            categories=req.categories,
            model_path=req.model_path,
            backend=req.backend,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache[key] = result
    return ApiResponse(data={"cached": False, "result": result})


@app.post("/infer/pointing", response_model=ApiResponse)
def infer_pointing(req: PointingRequest) -> ApiResponse:
    payload = req.model_dump()
    key = _cache_key("pointing", payload)
    if key in cache:
        return ApiResponse(data={"cached": True, "result": cache[key]})

    _, _, get_runtime = _lazy_imports()
    try:
        result = get_runtime().infer(
            image_path=req.image_path,
            task="pointing",
            categories=req.categories,
            model_path=req.model_path,
            backend=req.backend,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache[key] = result
    return ApiResponse(data={"cached": False, "result": result})


@app.post("/infer/segment", response_model=ApiResponse)
def infer_segment(req: SegmentRequest) -> ApiResponse:
    _, _, get_runtime = _lazy_imports()
    try:
        det = get_runtime().infer(
            image_path=req.image_path,
            task="detection",
            categories=req.categories,
            model_path=req.model_path,
            backend=req.backend,
        )
        seg = get_runtime().segment_with_sam(
            image_path=req.image_path,
            predictions=det.get("extracted_predictions", {}),
            sam_checkpoint=req.sam_checkpoint,
            sam_model_type=req.sam_model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ApiResponse(data={"detection": det, "segmentation": seg})


@app.post("/engine/grounding", response_model=ApiResponse)
def engine_grounding(req: GroundingRequest) -> ApiResponse:
    run_grounding_batch, _, _ = _lazy_imports()
    pairs = [(item.image_path, item.caption) for item in req.items]
    try:
        result = run_grounding_batch(
            image_caption_pairs=pairs,
            model_path=req.model_path,
            backend=req.backend,
            output_jsonl=req.output_jsonl,
            output_dir=req.output_dir,
            visualize=req.visualize,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ApiResponse(data=result)


@app.post("/grasp/optimize", response_model=ApiResponse)
def grasp_optimize(req: GraspDecisionRequest) -> ApiResponse:
    _, build_grasp_decision, get_runtime = _lazy_imports()
    categories = list(dict.fromkeys([req.target_object] + req.candidate_objects))
    try:
        det = get_runtime().infer(
            image_path=req.image_path,
            task="detection",
            categories=categories,
            model_path=req.model_path,
            backend=req.backend,
        )
        result = build_grasp_decision(
            target_object=req.target_object,
            predictions=det.get("extracted_predictions", {}),
            candidate_objects=req.candidate_objects,
            robot_exec_mode=req.robot_exec_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ApiResponse(data={"detection": det, **result})


# ---- Flywheel 端点 ----

@app.post("/flywheel/run", response_model=ApiResponse)
def flywheel_run(req: FlywheelRunRequest) -> ApiResponse:
    """触发完整数据飞轮 pipeline。"""
    from apps.data_flywheel.config import FlywheelConfig
    from apps.data_flywheel.pipeline import run_pipeline

    cfg = FlywheelConfig()
    if req.image_dir:
        cfg.image_dirs = [req.image_dir]
    cfg.rounds = req.rounds
    cfg.max_images_per_round = req.max_images
    cfg.augment_per_image = req.augment_per_image
    cfg.filter_adjectives = req.filter_adjectives
    cfg.model_path = req.model_path
    cfg.backend = req.backend
    cfg.sam_checkpoint = req.sam_checkpoint

    try:
        result = run_pipeline(cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ApiResponse(data=result)


@app.post("/flywheel/evaluate", response_model=ApiResponse)
def flywheel_evaluate(req: FlywheelEvaluateRequest) -> ApiResponse:
    """单独触发标注质量评估。"""
    from apps.data_flywheel.evaluator import evaluate_grounding

    try:
        result = evaluate_grounding(
            grounding_jsonl=req.grounding_jsonl,
            model_path=req.model_path,
            backend=req.backend,
            iou_thresh=req.iou_thresh,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ApiResponse(data=result)


@app.get("/flywheel/status", response_model=ApiResponse)
def flywheel_status() -> ApiResponse:
    """查询飞轮输出状态。"""
    import glob
    from apps.data_flywheel.config import FlywheelConfig

    cfg = FlywheelConfig()
    summary_path = os.path.join(cfg.output_root, "flywheel_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return ApiResponse(data={"status": "completed", "summary": summary})

    rounds = sorted(glob.glob(os.path.join(cfg.output_root, "round_*")))
    return ApiResponse(data={
        "status": "idle" if not rounds else "in_progress",
        "rounds_found": len(rounds),
    })


def main() -> None:
    import uvicorn

    _ensure_project_root_importable()
    _ensure_rex_repo_importable()

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8101"))
    uvicorn.run("services.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
