import random
from collections import deque

import numpy as np


def bfs_reachable(grid, start, goal):
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    if grid[sy, sx] == 1 or grid[gy, gx] == 1:
        return False
    q = deque([(sx, sy)])
    visited = {(sx, sy)}
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return False


def generate_map(
    size=64,
    obstacle_density=0.15,
    min_rect=3,
    max_rect=9,
    max_tries=200,
    rng=None,
):
    rng = rng or random.Random()
    grid = np.zeros((size, size), dtype=np.uint8)

    target_obstacles = int(size * size * obstacle_density)
    placed = 0
    tries = 0
    while placed < target_obstacles and tries < max_tries:
        tries += 1
        rw = rng.randint(min_rect, max_rect)
        rh = rng.randint(min_rect, max_rect)
        x0 = rng.randint(0, size - rw - 1)
        y0 = rng.randint(0, size - rh - 1)
        if grid[y0 : y0 + rh, x0 : x0 + rw].sum() > 0:
            continue
        grid[y0 : y0 + rh, x0 : x0 + rw] = 1
        placed = int(grid.sum())

    free_cells = list(zip(*np.where(grid == 0)))
    if len(free_cells) < 2:
        grid[:] = 0
        free_cells = list(zip(*np.where(grid == 0)))

    for _ in range(200):
        sy, sx = free_cells[rng.randrange(len(free_cells))]
        gy, gx = free_cells[rng.randrange(len(free_cells))]
        if (sx, sy) == (gx, gy):
            continue
        if bfs_reachable(grid, (sx, sy), (gx, gy)):
            return grid, (sx, sy), (gx, gy)

    grid[:] = 0
    sy, sx = free_cells[0]
    gy, gx = free_cells[-1]
    return grid, (sx, sy), (gx, gy)
