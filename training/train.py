import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.car_env import CarEnv
from utils.seed import set_seed


class TrainingLogCallback(BaseCallback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = None
        self.writer = None

    def _on_training_start(self):
        self.csv_file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["timesteps", "episode_reward", "episode_length"])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.writer.writerow([
                self.num_timesteps,
                round(float(episode["r"]), 6),
                int(episode["l"]),
            ])
            self.csv_file.flush()
        return True

    def _on_training_end(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None


class VerboseEvalCallback(EvalCallback):
    def _on_step(self):
        previous_best = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > previous_best:
            best_path = Path(self.best_model_save_path) / "best_model.zip"
            print(f"[eval] new best model: {best_path}")
            print(f"[eval] best mean reward: {self.best_mean_reward:.4f}")
        return result


def resolve_device(device):
    requested = device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested but CUDA is not available.")
        return "cuda"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    return "cpu"


def make_env(map_size, obstacle_density, lidar_rays, max_steps):
    def _init():
        env = CarEnv(
            render_mode=None,
            map_size=map_size,
            obstacle_density=obstacle_density,
            lidar_rays=lidar_rays,
            max_steps=max_steps,
        )
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--obstacle-density", type=float, default=0.15)
    parser.add_argument("--lidar-rays", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(args.device)

    print(f"[train] requested device: {args.device}")
    print(f"[train] cuda available: {torch.cuda.is_available()}")
    print(f"[train] using device: {resolved_device}")

    env = DummyVecEnv([make_env(args.map_size, args.obstacle_density, args.lidar_rays, args.max_steps)])
    eval_env = DummyVecEnv([make_env(args.map_size, args.obstacle_density, args.lidar_rays, args.max_steps)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
        device=resolved_device,
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=5000,
            save_path=str(save_dir),
            name_prefix="ppo_car_checkpoint",
        ),
        VerboseEvalCallback(
            eval_env,
            best_model_save_path=str(save_dir),
            log_path=str(save_dir),
            eval_freq=5000,
            deterministic=True,
            render=False,
        ),
        TrainingLogCallback(save_dir / "training_log.csv"),
    ])

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    latest_model_path = save_dir / "latest.zip"
    model.save(str(latest_model_path.with_suffix("")))

    print(f"[train] latest model saved to: {latest_model_path}")
    print(f"[train] best model path: {save_dir / 'best_model.zip'}")
    print(f"[train] csv log path: {save_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()
