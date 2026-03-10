import math
import random

import gymnasium as gym
import numpy as np

from env.lidar_sensor import lidar_scan
from env.map_generator import generate_map
from utils import config


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        seed=None,
        map_size=None,
        obstacle_density=None,
        lidar_rays=None,
        max_steps=None,
        fixed_map=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.map_size = map_size or config.MAP_SIZE
        self.obstacle_density = obstacle_density if obstacle_density is not None else config.OBSTACLE_DENSITY
        self.lidar_rays = lidar_rays or config.LIDAR_RAYS
        self.max_steps = max_steps or config.MAX_STEPS
        self.fixed_map = fixed_map
        if self.fixed_map is not None:
            self.map_size = self.fixed_map["grid"].shape[0]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_len = self.lidar_rays + 4
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        self.grid = None
        self.start = None
        self.goal = None
        self.pos = None
        self.heading = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.steps = 0
        self.prev_dist = None
        self.trail = []
        self.renderer = None
        self.episode_id = 0
        self.last_reward = 0.0
        self.last_info = {}

    def set_fixed_map(self, fixed_map):
        self.fixed_map = fixed_map
        if fixed_map is not None:
            self.map_size = fixed_map["grid"].shape[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and options.get("fixed_map") is not None:
            self.set_fixed_map(options["fixed_map"])

        map_seed = int(self.rng.integers(0, 2**31 - 1))
        if self.fixed_map is not None:
            self.grid = self.fixed_map["grid"].copy()
            self.start = tuple(self.fixed_map["start"])
            self.goal = tuple(self.fixed_map["goal"])
        else:
            grid, start, goal = generate_map(
                size=self.map_size,
                obstacle_density=self.obstacle_density,
                rng=random.Random(map_seed),
            )
            self.grid = grid
            self.start = start
            self.goal = goal

        self.pos = np.array([self.start[0] + 0.5, self.start[1] + 0.5], dtype=np.float32)
        self.heading = float(self.rng.uniform(-math.pi, math.pi))
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.steps = 0
        self.prev_dist = self._dist_to_goal()
        self.trail = [tuple(self.pos.tolist())]
        self.last_reward = 0.0
        self.episode_id += 1
        lidar = self._lidar()
        obs = self._get_obs(lidar)
        info = {
            "distance": self.prev_dist,
            "start": self.start,
            "goal": self.goal,
            "map_seed": map_seed,
            "episode_id": self.episode_id,
            "episode_result": "running",
            "reward": 0.0,
            "min_lidar": float(lidar.min()),
            "linear_vel": self.linear_vel,
            "angular_vel": self.angular_vel,
            "steps": self.steps,
            "map_mode": "fixed" if self.fixed_map is not None else "random",
        }
        self.last_info = info
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.linear_vel = float((action[0] + 1.0) * 0.5 * config.MAX_LINEAR_VEL)
        self.angular_vel = float(action[1] * config.MAX_ANGULAR_VEL)

        self.heading += self.angular_vel * config.DT
        dx = self.linear_vel * math.cos(self.heading) * config.DT
        dy = self.linear_vel * math.sin(self.heading) * config.DT
        step_distance = float(math.hypot(dx, dy))
        self.pos += np.array([dx, dy], dtype=np.float32)
        self.steps += 1
        self.trail.append(tuple(self.pos.tolist()))

        collision = self._check_collision()
        reached = self._dist_to_goal() < 1.0

        dist = self._dist_to_goal()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        lidar = self._lidar()
        min_lidar = float(lidar.min())

        reward = 0.3 * progress
        reward -= 0.01
        reward -= config.PATH_LENGTH_PENALTY * step_distance
        if min_lidar < config.SAFETY_DIST / config.LIDAR_RANGE:
            ratio = (config.SAFETY_DIST / config.LIDAR_RANGE - min_lidar) / (config.SAFETY_DIST / config.LIDAR_RANGE)
            reward -= 0.1 * ratio
        if collision:
            reward -= 1.0
        if reached:
            reward += 2.0
        self.last_reward = reward

        terminated = collision or reached
        truncated = self.steps >= self.max_steps
        episode_result = "running"
        if collision:
            episode_result = "collision"
        elif reached:
            episode_result = "reached"
        elif truncated:
            episode_result = "truncated"

        obs = self._get_obs(lidar)
        info = {
            "distance": dist,
            "collision": collision,
            "reached": reached,
            "steps": self.steps,
            "linear_vel": self.linear_vel,
            "angular_vel": self.angular_vel,
            "reward": reward,
            "min_lidar": min_lidar,
            "step_distance": step_distance,
            "path_penalty": config.PATH_LENGTH_PENALTY * step_distance,
            "episode_result": episode_result,
            "episode_id": self.episode_id,
            "start": self.start,
            "goal": self.goal,
            "map_mode": "fixed" if self.fixed_map is not None else "random",
        }
        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _dist_to_goal(self):
        gx, gy = self.goal
        return float(math.hypot(self.pos[0] - gx, self.pos[1] - gy))

    def _check_collision(self):
        x, y = self.pos
        if x < 0 or y < 0 or x >= self.map_size or y >= self.map_size:
            return True
        if self.grid[int(y), int(x)] == 1:
            return True
        return False

    def _lidar(self):
        return lidar_scan(
            self.grid,
            (float(self.pos[0]), float(self.pos[1])),
            float(self.heading),
            num_rays=self.lidar_rays,
            max_range=config.LIDAR_RANGE,
        )

    def _get_obs(self, lidar=None):
        if lidar is None:
            lidar = self._lidar()
        gx, gy = self.goal
        dx = gx - self.pos[0]
        dy = gy - self.pos[1]
        goal_angle = math.atan2(dy, dx) - self.heading
        obs = np.concatenate(
            [
                lidar,
                np.array(
                    [
                        math.sin(goal_angle),
                        math.cos(goal_angle),
                        self.linear_vel / config.MAX_LINEAR_VEL,
                        self.angular_vel / config.MAX_ANGULAR_VEL,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def render(self, overlay=None):
        from visualization.renderer import Renderer

        if self.renderer is None:
            self.renderer = Renderer(self.map_size)
        lidar = self._lidar()
        angles = np.linspace(-math.pi, math.pi, self.lidar_rays, endpoint=False)
        points = []
        for i, angle in enumerate(angles):
            dist = lidar[i] * config.LIDAR_RANGE
            rx = self.pos[0] + dist * math.cos(self.heading + angle)
            ry = self.pos[1] + dist * math.sin(self.heading + angle)
            points.append((rx, ry))
        info = {
            "step": self.steps,
            "dist": round(self._dist_to_goal(), 2),
            "reward": round(self.last_reward, 3),
            "v": round(self.linear_vel, 2),
            "w": round(self.angular_vel, 2),
            "min_lidar": round(float(lidar.min()) * config.LIDAR_RANGE, 2),
            "status": self.last_info.get("episode_result", "running"),
        }
        if overlay:
            info.update(overlay)
        self.renderer.render(
            self.grid,
            (float(self.pos[0]), float(self.pos[1])),
            float(self.heading),
            self.goal,
            points,
            self.trail,
            info,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
