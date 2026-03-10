import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env.car_env import CarEnv
from utils.map_io import load_map


def run_episode(env, model):
    obs, _ = env.reset()
    done = False
    steps = 0
    path_len = 0.0
    prev_pos = env.pos.copy()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        path_len += float(np.linalg.norm(env.pos - prev_pos))
        prev_pos = env.pos.copy()

    return info.get("reached", False), info.get("collision", False), steps, path_len, info.get("episode_result", "running")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map_path", type=str, default="")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    fixed_map = load_map(args.map_path) if args.map_path else None
    env = CarEnv(render_mode=None, seed=args.seed, fixed_map=fixed_map)
    model = PPO.load(str(model_path))

    print("model_path", model_path)
    if args.map_path:
        print("map_path", Path(args.map_path).resolve())
    results = [run_episode(env, model) for _ in range(args.episodes)]
    reached = sum(result[0] for result in results)
    collisions = sum(result[1] for result in results)
    steps = [result[2] for result in results]
    paths = [result[3] for result in results]

    print("episodes", args.episodes)
    print("success_rate", reached / args.episodes)
    print("collision_rate", collisions / args.episodes)
    print("avg_steps", sum(steps) / args.episodes)
    print("avg_path", sum(paths) / args.episodes)


if __name__ == "__main__":
    main()
