import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from env.car_env import CarEnv
from training.train import resolve_device
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


def make_env(rank, seed, map_size, obstacle_density, lidar_rays, max_steps):
    def _init():
        env = CarEnv(
            render_mode=None,
            seed=seed + rank,
            map_size=map_size,
            obstacle_density=obstacle_density,
            lidar_rays=lidar_rays,
            max_steps=max_steps,
        )
        return Monitor(env)

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="GPU server training entry for EV RL car.")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models_gpu")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--obstacle-density", type=float, default=0.12)
    parser.add_argument("--lidar-rays", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--hidden-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(args.device)
    cuda_available = torch.cuda.is_available()
    print(f"[server-train] requested device: {args.device}")
    print(f"[server-train] cuda available: {cuda_available}")
    print(f"[server-train] using device: {resolved_device}")
    if resolved_device == "cuda":
        print(f"[server-train] gpu name: {torch.cuda.get_device_name(0)}")

    env = DummyVecEnv([
        make_env(i, args.seed, args.map_size, args.obstacle_density, args.lidar_rays, args.max_steps)
        for i in range(args.n_envs)
    ])
    eval_env = DummyVecEnv([
        make_env(10_000 + i, args.seed, args.map_size, args.obstacle_density, args.lidar_rays, args.max_steps)
        for i in range(2)
    ])

    policy_kwargs = {
        "activation_fn": nn.Tanh,
        "net_arch": dict(pi=[args.hidden_size, args.hidden_size], vf=[args.hidden_size, args.hidden_size]),
    }

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=resolved_device,
        tensorboard_log=str(save_dir / "tb"),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(1, args.save_freq // max(1, args.n_envs)),
            save_path=str(save_dir),
            name_prefix="ppo_car_server_checkpoint",
        ),
        VerboseEvalCallback(
            eval_env,
            best_model_save_path=str(save_dir),
            log_path=str(save_dir),
            eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        ),
        TrainingLogCallback(save_dir / "training_log.csv"),
    ])

    print(f"[server-train] timesteps: {args.timesteps}")
    print(f"[server-train] envs: {args.n_envs}")
    print(f"[server-train] map_size: {args.map_size}")
    print(f"[server-train] obstacle_density: {args.obstacle_density}")
    print(f"[server-train] lidar_rays: {args.lidar_rays}")
    print(f"[server-train] max_steps: {args.max_steps}")
    print(f"[server-train] save_dir: {save_dir}")

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    latest_model_path = save_dir / "latest.zip"
    model.save(str(latest_model_path.with_suffix("")))

    print(f"[server-train] latest model saved to: {latest_model_path}")
    print(f"[server-train] best model path: {save_dir / 'best_model.zip'}")
    print(f"[server-train] csv log path: {save_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()
