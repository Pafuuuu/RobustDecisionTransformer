"""
Evaluate a pre-trained IQL policy on Windows using gymnasium.
Loads the actor from IQL_model/{env}/3000.pt and rolls out n_episodes.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

import gymnasium
from algos.IQL import DeterministicPolicy
from utils.dt_functions import get_normalized_score

GYM_ENV_ID = {
    "walker2d":    "Walker2d-v4",
    "hopper":      "Hopper-v4",
    "halfcheetah": "HalfCheetah-v4",
}

ENV_DIMS = {
    "walker2d-medium-replay-v2":    {"state_dim": 17, "action_dim": 6, "max_action": 1.0},
    "hopper-medium-replay-v2":      {"state_dim": 11, "action_dim": 3, "max_action": 1.0},
    "halfcheetah-medium-replay-v2": {"state_dim": 17, "action_dim": 6, "max_action": 1.0},
}


def load_state_stats(dataset_path, env_name, sample_ratio=0.1):
    pt_path = os.path.join(dataset_path, "original", f"{env_name}_ratio_{sample_ratio}.pt")
    dataset = torch.load(pt_path, weights_only=False)
    obs = dataset["observations"]  # flat numpy array (N, state_dim)
    mean = obs.mean(0, keepdims=True)
    std = obs.std(0, keepdims=True) + 1e-3
    return mean, std


def eval_iql(env_name, model_path, dataset_path, n_episodes, seed, device):
    dims = ENV_DIMS[env_name]
    state_dim  = dims["state_dim"]
    action_dim = dims["action_dim"]
    max_action = dims["max_action"]

    # load actor
    policy = DeterministicPolicy(state_dim, action_dim, max_action, n_hidden=2, dropout=None)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["actor"])
    policy = policy.to(device).eval()

    # state normalization from dataset
    state_mean, state_std = load_state_stats(dataset_path, env_name)

    # create gymnasium env
    gym_id = GYM_ENV_ID[env_name.split("-")[0]]
    env = gymnasium.make(gym_id)

    np.random.seed(seed)
    torch.manual_seed(seed)

    episode_returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_norm = (obs - state_mean) / state_std
            action = policy.act(obs_norm.squeeze(), device)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        episode_returns.append(ep_return)
        if (ep + 1) % 10 == 0:
            print(f"  episode {ep+1}/{n_episodes}  return={ep_return:.1f}")

    env.close()
    returns = np.array(episode_returns)
    norm_scores = get_normalized_score(env_name, returns) * 100
    print(f"\n{'='*50}")
    print(f"IQL  |  {env_name}")
    print(f"{'='*50}")
    print(f"  reward mean : {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"  norm score  : {norm_scores.mean():.2f} ± {norm_scores.std():.2f}")
    print(f"{'='*50}\n")
    return returns, norm_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",          default="walker2d-medium-replay-v2")
    parser.add_argument("--iql_model_dir",default="IQL_model")
    parser.add_argument("--dataset_path", default="datasets")
    parser.add_argument("--n_episodes",   type=int, default=100)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = os.path.join(args.iql_model_dir, args.env, "3000.pt")
    eval_iql(args.env, model_path, args.dataset_path, args.n_episodes, args.seed, args.device)
