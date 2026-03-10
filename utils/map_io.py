import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np


class MapValidationError(ValueError):
    pass


def normalize_map_data(map_data):
    grid = np.array(map_data["grid"], dtype=np.uint8)
    start = tuple(map_data["start"])
    goal = tuple(map_data["goal"])
    name = map_data.get("name", "unnamed")
    created_at = map_data.get("created_at")
    return {
        "name": name,
        "created_at": created_at,
        "grid": grid,
        "start": start,
        "goal": goal,
    }


def serialize_map_data(map_data):
    normalized = normalize_map_data(map_data)
    return {
        "name": normalized["name"],
        "created_at": normalized["created_at"],
        "grid": normalized["grid"].astype(int).tolist(),
        "start": list(normalized["start"]),
        "goal": list(normalized["goal"]),
    }


def _reachable(grid, start, goal):
    height, width = grid.shape
    sx, sy = start
    gx, gy = goal
    queue = deque([(sx, sy)])
    visited = {(sx, sy)}
    while queue:
        x, y = queue.popleft()
        if (x, y) == (gx, gy):
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def validate_map_data(map_data):
    normalized = normalize_map_data(map_data)
    grid = normalized["grid"]
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise MapValidationError("Map grid must be a square 2D array.")
    size = grid.shape[0]
    for point_name in ("start", "goal"):
        x, y = normalized[point_name]
        if not (0 <= x < size and 0 <= y < size):
            raise MapValidationError(f"{point_name} must be inside the map bounds.")
        if grid[y, x] == 1:
            raise MapValidationError(f"{point_name} cannot be placed on an obstacle.")
    if normalized["start"] == normalized["goal"]:
        raise MapValidationError("Start and goal must be different cells.")
    if not _reachable(grid, normalized["start"], normalized["goal"]):
        raise MapValidationError("Start and goal are not connected by a free path.")
    return normalized


def save_map(map_dir, map_data):
    normalized = validate_map_data(map_data)
    payload = serialize_map_data({
        "name": normalized["name"],
        "created_at": normalized["created_at"] or datetime.utcnow().isoformat() + "Z",
        "grid": normalized["grid"],
        "start": normalized["start"],
        "goal": normalized["goal"],
    })
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    file_path = map_dir / f"{payload['name']}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return file_path


def load_map(map_path):
    path = Path(map_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("name", path.stem)
    return validate_map_data(payload)


def list_maps(map_dir):
    map_dir = Path(map_dir)
    if not map_dir.exists():
        return []
    items = []
    for path in sorted(map_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.append(
            {
                "name": payload.get("name", path.stem),
                "path": str(path.resolve()),
                "created_at": payload.get("created_at"),
                "size": len(payload.get("grid", [])),
            }
        )
    return items
