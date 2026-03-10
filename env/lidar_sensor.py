import math

import numpy as np


def lidar_scan(grid, position, heading, num_rays=24, max_range=15.0, step=0.5):
    h, w = grid.shape
    x, y = position
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)
    distances = []
    for a in angles:
        ang = heading + a
        dist = 0.0
        hit = False
        while dist < max_range:
            dist += step
            rx = x + dist * math.cos(ang)
            ry = y + dist * math.sin(ang)
            if rx < 0 or ry < 0 or rx >= w or ry >= h:
                hit = True
                break
            if grid[int(ry), int(rx)] == 1:
                hit = True
                break
        if not hit:
            dist = max_range
        distances.append(min(dist, max_range) / max_range)
    return np.array(distances, dtype=np.float32)
