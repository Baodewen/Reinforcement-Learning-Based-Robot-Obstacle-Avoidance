import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from training.train import resolve_device
from utils import config
from utils.map_io import MapValidationError, list_maps, load_map, save_map, serialize_map_data
from utils.model_compat import load_ppo_with_env_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = (PROJECT_ROOT / config.MODEL_DIR).resolve()
MAP_DIR = (PROJECT_ROOT / config.MAP_DIR).resolve()
PYTHON_EXE = Path(sys.executable)
TIMESTEP_PATTERN = re.compile(r"total_timesteps\s*\|\s*(\d+)")


class TrainRequest(BaseModel):
    timesteps: int = Field(default=10000, ge=1)
    seed: int = 0
    map_size: int = Field(default=config.MAP_SIZE, ge=16, le=128)
    obstacle_density: float = Field(default=config.OBSTACLE_DENSITY, ge=0.01, le=0.45)
    lidar_rays: int = Field(default=config.LIDAR_RAYS, ge=8, le=128)
    max_steps: int = Field(default=config.MAX_STEPS, ge=50, le=2000)
    device: Literal["auto", "gpu", "cpu"] = "auto"


class MapRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    grid: list[list[int]]
    start: list[int]
    goal: list[int]


class DemoRequest(BaseModel):
    model_path: str
    map_path: Optional[str] = ""
    seed: int = 0
    fps: int = Field(default=20, ge=1, le=120)
    max_episodes: int = Field(default=0, ge=0)


class TrainingTask:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.thread = None
        self.status = "idle"
        self.started_at = None
        self.finished_at = None
        self.logs = []
        self.total_timesteps = 0
        self.error = None
        self.request = None
        self.device_requested = "auto"
        self.device_resolved = "cpu"
        self.cuda_available = torch.cuda.is_available()

    def snapshot(self):
        latest = MODEL_DIR / "latest.zip"
        best = MODEL_DIR / "best_model.zip"
        with self.lock:
            return {
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "total_timesteps": self.total_timesteps,
                "logs": self.logs[-40:],
                "error": self.error,
                "request": self.request,
                "device_requested": self.device_requested,
                "device_resolved": self.device_resolved,
                "cuda_available": self.cuda_available,
                "latest_model_path": str(latest) if latest.exists() else None,
                "best_model_path": str(best) if best.exists() else None,
            }

    def start(self, req: TrainRequest):
        with self.lock:
            if self.status == "running":
                raise HTTPException(status_code=409, detail="A training task is already running.")
            self.status = "queued"
            self.started_at = datetime.utcnow().isoformat() + "Z"
            self.finished_at = None
            self.logs = []
            self.total_timesteps = 0
            self.error = None
            self.request = req.model_dump()
            self.device_requested = req.device
            self.cuda_available = torch.cuda.is_available()
            try:
                self.device_resolved = resolve_device(req.device)
            except RuntimeError as exc:
                self.status = "failed"
                self.error = str(exc)
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        command = [
            str(PYTHON_EXE),
            str(PROJECT_ROOT / "training" / "train.py"),
            "--timesteps",
            str(req.timesteps),
            "--seed",
            str(req.seed),
            "--save_dir",
            str(MODEL_DIR),
            "--device",
            req.device,
            "--map-size",
            str(req.map_size),
            "--obstacle-density",
            str(req.obstacle_density),
            "--lidar-rays",
            str(req.lidar_rays),
            "--max-steps",
            str(req.max_steps),
        ]
        self.thread = threading.Thread(target=self._run, args=(command,), daemon=True)
        self.thread.start()

    def _run(self, command):
        with self.lock:
            self.status = "running"
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        with self.lock:
            self.process = process
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            with self.lock:
                self.logs.append(line)
                match = TIMESTEP_PATTERN.search(line)
                if match:
                    self.total_timesteps = int(match.group(1))
        process.wait()
        with self.lock:
            self.process = None
            self.finished_at = datetime.utcnow().isoformat() + "Z"
            if process.returncode == 0:
                self.status = "completed"
            else:
                self.status = "failed"
                self.error = f"Training process exited with code {process.returncode}."


training_task = TrainingTask()
app = FastAPI(title="EV Simulation Control Panel")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (PROJECT_ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


@app.get("/api/system")
def system_info():
    return {
        "cuda_available": torch.cuda.is_available(),
        "python": sys.version,
        "model_dir": str(MODEL_DIR),
        "map_dir": str(MAP_DIR),
    }


@app.post("/api/train/start")
def start_training(request: TrainRequest):
    training_task.start(request)
    return training_task.snapshot()


@app.get("/api/train/status")
def training_status():
    return training_task.snapshot()


@app.get("/api/models")
def models_status():
    items = []
    for name in ("latest.zip", "best_model.zip"):
        path = MODEL_DIR / name
        items.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "updated_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z" if path.exists() else None,
                "size": path.stat().st_size if path.exists() else None,
            }
        )
    return {"models": items}


@app.get("/api/maps")
def maps_status():
    return {"maps": list_maps(MAP_DIR)}


@app.get("/api/maps/{name}")
def get_map(name: str):
    path = MAP_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Map not found.")
    return serialize_map_data(load_map(path))


@app.post("/api/maps")
def save_map_api(request: MapRequest):
    try:
        path = save_map(MAP_DIR, request.model_dump())
    except MapValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "path": str(path.resolve())}


@app.post("/api/demo/launch")
def launch_demo(request: DemoRequest):
    model_path = Path(request.model_path)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    try:
        _, env_config = load_ppo_with_env_config(model_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}") from exc

    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "demo.py"),
        "--model_path",
        str(model_path),
        "--seed",
        str(request.seed),
        "--fps",
        str(request.fps),
        "--max_episodes",
        str(request.max_episodes),
    ]
    if request.map_path:
        map_path = Path(request.map_path)
        if not map_path.is_absolute():
            map_path = (PROJECT_ROOT / map_path).resolve()
        if not map_path.exists():
            raise HTTPException(status_code=404, detail="Map file not found.")
        command.extend(["--map_path", str(map_path)])

    subprocess.Popen(command, cwd=PROJECT_ROOT)
    return {"ok": True, "command": command, "env_config": env_config}

