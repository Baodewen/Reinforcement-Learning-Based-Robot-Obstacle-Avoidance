from stable_baselines3 import PPO


def load_ppo_with_env_config(model_path):
    model = PPO.load(str(model_path))
    obs_shape = getattr(model.observation_space, "shape", None)
    if not obs_shape or len(obs_shape) != 1:
        raise ValueError(f"Unsupported observation space shape: {obs_shape}")

    obs_dim = int(obs_shape[0])
    if obs_dim < 4:
        raise ValueError(f"Observation dimension is too small: {obs_dim}")

    return model, {
        "obs_dim": obs_dim,
        "lidar_rays": obs_dim - 4,
    }
