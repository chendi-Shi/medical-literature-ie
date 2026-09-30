from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .cli import safe_child
from .data import prepare, read_json
from .training import Predictor, TrainConfig, allocate_run, compare_runs, update_run


class ImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=1_000_000)
    format: str = "jsonl"
    provenance: str = Field(default="user-provided", max_length=1000)
    seed: int = 42


class TrainRequest(BaseModel):
    dataset: str
    backend: str = "baseline"
    method: str = "full"
    model: str = "BAAI/bge-small-zh-v1.5"
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-5, gt=0, lt=1)
    max_length: int = Field(default=128, ge=8, le=4096)
    accumulation: int = Field(default=1, ge=1, le=128)
    device: str = "auto"
    seed: int = 42


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


def create_app(workspace: Path):
    workspace = workspace.resolve()
    app = FastAPI(title="NLP Training Lab", version="0.1.0")

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Local single-user tool: reject cross-origin browser writes and Host rebinding.
        host = request.headers.get("host", "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "testserver"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "仅接受本地请求"}, status_code=403)
        if request.method != "GET":
            origin = request.headers.get("origin")
            if origin and origin != f"http://{request.headers.get('host')}":
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "不接受跨站写入"}, status_code=403)
        try:
            declared_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared_length = 2_000_001
        if declared_length > 2_000_000:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "请求超过 2MB 限制"}, status_code=413)
        if request.method != "GET" and len(await request.body()) > 2_000_000:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "请求超过 2MB 限制"}, status_code=413)
        return await call_next(request)

    def checked(parent, name, required=True):
        try:
            path = safe_child(workspace / parent, name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if required and not path.exists():
            raise HTTPException(404, "数据集或实验不存在")
        return path

    def spawn(run, mode="train", split="test"):
        log_path = run / ("worker.log" if mode == "train" else "evaluation.log")
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, "-m", "nlp_lab.worker", str(run), mode, split],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/datasets")
    def datasets():
        return [{"id": path.name, **read_json(path / "manifest.json")}
                for path in sorted((workspace / "datasets").glob("*")) if (path / "manifest.json").exists()]

    @app.post("/api/datasets", status_code=201)
    def import_dataset(body: ImportRequest):
        destination = checked("datasets", body.name, required=False)
        if body.format not in ("csv", "jsonl"):
            raise HTTPException(400, "format 必须是 csv / jsonl")
        with tempfile.TemporaryDirectory(prefix="nlp-lab-import-") as temporary:
            path = Path(temporary) / ("upload." + body.format)
            path.write_text(body.content, encoding="utf-8")
            try:
                result = prepare(path, destination, body.seed, body.provenance)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        return {"id": body.name, **result}

    @app.get("/api/runs")
    def runs():
        return [read_json(path / "run.json") for path in sorted((workspace / "runs").glob("*"), reverse=True) if (path / "run.json").exists()]

    @app.post("/api/runs", status_code=202)
    def train(body: TrainRequest):
        if (workspace / ".training.lock").exists():
            raise HTTPException(409, "已有训练任务运行，请等待结束")
        dataset = checked("datasets", body.dataset)
        config = TrainConfig(**body.model_dump(exclude={"dataset"}))
        try:
            run = allocate_run(workspace, dataset, config)
            spawn(run)
        except (ValueError, OSError) as e:
            if "run" in locals():
                update_run(run, status="failed", error=str(e))
            raise HTTPException(400, str(e)) from e
        return read_json(run / "run.json")

    @app.get("/api/runs/{run_id}")
    def detail(run_id: str):
        run = checked("runs", run_id)
        result = read_json(run / "run.json")
        for name in ("history", "validation_metrics", "test_metrics", "validation_errors", "test_errors", "evaluation_status"):
            if (run / f"{name}.json").exists():
                value = read_json(run / f"{name}.json")
                result[name] = value[:100] if name.endswith("errors") else value
        if (run / "worker.log").exists():
            result["log_tail"] = (run / "worker.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        return result

    @app.post("/api/runs/{run_id}/predict")
    def predict(run_id: str, body: PredictRequest):
        try:
            return Predictor(checked("runs", run_id)).predict(body.texts)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/runs/{run_id}/evaluate", status_code=202)
    def evaluate(run_id: str, split: str = "test"):
        run = checked("runs", run_id)
        if split not in ("validation", "test"):
            raise HTTPException(400, "未知评测划分")
        if read_json(run / "run.json")["status"] != "completed":
            raise HTTPException(400, "请等待训练完成")
        status_file = run / "evaluation_status.json"
        if status_file.exists() and read_json(status_file)["status"] == "running":
            raise HTTPException(409, "评测仍在运行")
        from .data import save_json
        save_json(status_file, {"status": "running", "split": split})
        try:
            spawn(run, "evaluate", split)
        except OSError as e:
            save_json(status_file, {"status": "failed", "split": split, "error": str(e)})
            raise HTTPException(500, "无法启动评测进程") from e
        return {"status": "running", "split": split}

    @app.get("/api/compare")
    def compare(run_ids: str, split: str = "validation"):
        if split not in ("validation", "test"):
            raise HTTPException(400, "未知评测划分")
        try:
            return compare_runs([checked("runs", x) for x in run_ids.split(",")], split)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e)) from e

    return app
