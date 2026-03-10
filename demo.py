import argparse
import os
import sys
from pathlib import Path

import pygame

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env.car_env import CarEnv
from utils.map_io import load_map
from utils.model_compat import load_ppo_with_env_config


def build_overlay(env, model_path, fps, paused, completed_episodes, last_summary):
    info = dict(env.last_info)
    return {
        "model_name": Path(model_path).name,
        "step": info.get("steps", env.steps),
        "reward": round(float(info.get("reward", 0.0)), 4),
        "distance": round(float(info.get("distance", env._dist_to_goal())), 3),
        "linear_vel": round(float(info.get("linear_vel", env.linear_vel)), 3),
        "angular_vel": round(float(info.get("angular_vel", env.angular_vel)), 3),
        "min_lidar": round(float(info.get("min_lidar", 0.0)), 3),
        "status": info.get("episode_result", "running"),
        "episode_id": info.get("episode_id", env.episode_id),
        "paused": paused,
        "fps": fps,
        "completed_episodes": completed_episodes,
        "controls": "Space pause/resume | R reset | N new map | [ ] speed | Q quit",
        "last_result": last_summary.get("result", "-"),
        "last_steps": last_summary.get("steps", "-"),
        "last_distance": last_summary.get("distance", "-"),
        "map_mode": info.get("map_mode", "random"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--map_path", type=str, default="")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    fixed_map = load_map(args.map_path) if args.map_path else None
    model, env_config = load_ppo_with_env_config(model_path)
    env = CarEnv(
        render_mode=None,
        seed=args.seed,
        fixed_map=fixed_map,
        lidar_rays=env_config["lidar_rays"],
    )

    obs, _ = env.reset()
    paused = False
    running = True
    current_fps = max(1, args.fps)
    completed_episodes = 0
    last_summary = {"result": "-", "steps": "-", "distance": "-"}

    env.render(build_overlay(env, model_path, current_fps, paused, completed_episodes, last_summary))
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                elif event.key == pygame.K_n:
                    if env.fixed_map is None:
                        obs, _ = env.reset()
                elif event.key == pygame.K_LEFTBRACKET:
                    current_fps = max(1, current_fps - 5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    current_fps = min(120, current_fps + 5)

        if not paused and running:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                completed_episodes += 1
                last_summary = {
                    "result": info.get("episode_result", "unknown"),
                    "steps": info.get("steps", env.steps),
                    "distance": round(float(info.get("distance", env._dist_to_goal())), 3),
                }
                if args.max_episodes and completed_episodes >= args.max_episodes:
                    running = False
                else:
                    obs, _ = env.reset()

        env.render(build_overlay(env, model_path, current_fps, paused, completed_episodes, last_summary))
        clock.tick(current_fps)

    env.close()


if __name__ == "__main__":
    main()
