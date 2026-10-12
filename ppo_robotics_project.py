"""PPO for a Continuous-Control Robotics Task

Environment: CustomContinuousRobotJointEnv
- Gymnasium-compatible API: reset(seed, options) and step(action)
- State: [sin(angle), cos(angle), angular_velocity / max_speed, sin(error), cos(error)]
  where error = angle - target (the target expressed relative to the joint)
- Action: continuous motor torque in [-2, 2]
- Goal: move a single rotary robot joint to a target angle and stabilize it.

The implementation includes:
- Gaussian actor-critic neural network
- PPO clipped surrogate objective
- Generalized Advantage Estimation (GAE) with correct time-limit bootstrapping
- Fixed observation scaling to [-1, 1] (inside the environment) and advantage normalization
- Mini-batch updates over multiple epochs
- Training metrics (reward, losses, entropy, clip fraction, approx. KL) and plots
- Deterministic evaluation against random and zero-torque baselines
- Optional ablation study of clipping and GAE lambda across several seeds

Setup:
    pip install -r requirements.txt     (Python 3.10+)
Run:
    python ppo_robotics_project.py              # main training run + evaluation
    python ppo_robotics_project.py --ablation   # also run the clipping / GAE ablation
All outputs are written to ./outputs
"""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import csv
import json
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    import gymnasium as gym
    from gymnasium import spaces
    GymBase = gym.Env
except ImportError:
    gym = None
    spaces = None
    GymBase = object


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Continuous robotics environment
# -----------------------------
class CustomContinuousRobotJointEnv(GymBase):
    """A lightweight continuous-control robotic joint reaching task.

    The agent controls motor torque applied to a rotary joint. Each episode uses
    a fixed target angle sampled at reset. The agent receives higher reward as
    the joint reaches the target with low angular velocity and modest torque.

    Episodes never fail; they end only when the 200-step time limit is reached
    (truncated=True, terminated=False). This distinction matters for GAE.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 200, dt: float = 0.05):
        if GymBase is not object:
            super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.max_torque = 2.0
        self.max_speed = 8.0
        self.inertia = 1.0
        self.damping = 0.18
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.target = 0.0
        self.steps = 0
        self.rng = np.random.default_rng()

        self.observation_dim = 5
        self.action_dim = 1
        self.action_low = np.array([-self.max_torque], dtype=np.float32)
        self.action_high = np.array([self.max_torque], dtype=np.float32)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.ones(self.observation_dim, dtype=np.float32),
                high=np.ones(self.observation_dim, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=self.action_low, high=self.action_high, dtype=np.float32
            )

    def _get_obs(self) -> np.ndarray:
        # Preprocessing / fixed observation scaling: every feature lies in [-1, 1].
        # Angles are encoded as (sin, cos) to avoid the discontinuity at +/- pi.
        # The target is given RELATIVE to the joint (sin/cos of the angle error),
        # like the fingertip-to-target vector in MuJoCo Reacher. An earlier version
        # passed the raw target / pi, which forced the network to learn wrapped
        # angle subtraction itself; that version barely beat a random policy.
        error = self._angle_difference(self.angle, self.target)
        return np.array(
            [
                math.sin(self.angle),       # absolute angle is still needed for gravity
                math.cos(self.angle),
                self.angular_velocity / self.max_speed,
                math.sin(error),
                math.cos(error),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        return ((a - b + math.pi) % (2 * math.pi)) - math.pi

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.angle = self.rng.uniform(-math.pi, math.pi)
        self.angular_velocity = self.rng.uniform(-0.5, 0.5)
        self.target = self.rng.uniform(-math.pi, math.pi)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        torque = float(np.clip(action[0], -self.max_torque, self.max_torque))

        # Simple rotary-joint dynamics with damping and a small gravity term.
        gravity = 0.35 * math.sin(self.angle)
        acceleration = (torque - self.damping * self.angular_velocity - gravity) / self.inertia
        self.angular_velocity += acceleration * self.dt
        self.angular_velocity = float(np.clip(self.angular_velocity, -self.max_speed, self.max_speed))
        self.angle += self.angular_velocity * self.dt
        self.angle = ((self.angle + math.pi) % (2 * math.pi)) - math.pi
        self.steps += 1

        error = self._angle_difference(self.angle, self.target)
        # Reward is near +1 when accurately stabilized; otherwise penalizes error.
        reward = 1.0 - 1.4 * (error / math.pi) ** 2 - 0.04 * self.angular_velocity**2 - 0.002 * torque**2

        # Small bonus for precise, stable reaching.
        if abs(error) < 0.10 and abs(self.angular_velocity) < 0.20:
            reward += 0.25

        terminated = False
        truncated = self.steps >= self.max_steps
        info = {
            "angle_error": abs(error),
            "target": self.target,
            "angle": self.angle,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# -----------------------------
# Actor-Critic Model
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        # Learnable log standard deviation for the Gaussian policy.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return raw_action, log_prob, entropy, value

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    seed: int = 42
    total_steps: int = 204_800    # 400 updates; exact multiple of rollout_steps
    rollout_steps: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    update_epochs: int = 5
    batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_size: int = 64


# -----------------------------
# GAE
# -----------------------------
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Generalized Advantage Estimation.

    dones[t] = 1 marks the last step of an episode, so the recursion does not
    leak across episode boundaries. Time-limit truncation is handled before this
    function is called by adding gamma * V(s_T) to the final reward.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# -----------------------------
# PPO Training
# -----------------------------
def train(config: PPOConfig, verbose: bool = True):
    """Train one PPO agent. Returns (model, env, episode_rewards, episode_errors, update_rows)."""
    set_seed(config.seed)
    device = torch.device("cpu")

    env = CustomContinuousRobotJointEnv(max_steps=200)
    model = ActorCritic(env.observation_dim, env.action_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    obs, _ = env.reset(seed=config.seed)
    global_step = 0
    episode_reward = 0.0
    episode_rewards = []
    episode_errors = []
    update_rows = []

    assert config.total_steps % config.rollout_steps == 0, \
        "total_steps must be a multiple of rollout_steps (avoids a tiny final batch)"

    while global_step < config.total_steps:
        obs_buf, action_buf, logp_buf = [], [], []
        reward_buf, done_buf, value_buf = [], [], []

        for _ in range(config.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                raw_action, log_prob, _, value = model.act(obs_tensor)

            # Environment receives bounded torque. PPO stores the raw Gaussian sample
            # so the likelihood ratio is computed consistently.
            env_action = torch.clamp(raw_action, -env.max_torque, env.max_torque)
            next_obs, reward, terminated, truncated, info = env.step(env_action.cpu().numpy()[0])
            done = terminated or truncated
            episode_reward += reward  # log the true environment reward

            # Time-limit truncation is not a real terminal state: the joint could keep
            # collecting reward. Bootstrap with V(s_T) so the critic's targets stay
            # consistent instead of dropping to zero at step 200.
            train_reward = reward
            if truncated and not terminated:
                with torch.no_grad():
                    final_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    train_reward += config.gamma * model.value(final_obs_t).item()

            obs_buf.append(obs)
            action_buf.append(raw_action.cpu().numpy()[0])
            logp_buf.append(float(log_prob.item()))
            reward_buf.append(train_reward)
            done_buf.append(float(done))
            value_buf.append(float(value.item()))

            obs = next_obs
            global_step += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_errors.append(info["angle_error"])
                episode_reward = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(model.value(obs_tensor).item())

        rewards = np.asarray(reward_buf, dtype=np.float32)
        values = np.asarray(value_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards, values, dones, last_value, config.gamma, config.gae_lambda
        )
        # Advantage normalization keeps the policy-gradient scale stable.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.asarray(action_buf), dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(np.asarray(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

        n = len(obs_buf)
        policy_losses, value_losses, entropies, clip_fracs, approx_kls = [], [], [], [], []

        for _ in range(config.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, config.batch_size):
                mb_idx = indices[start : start + config.batch_size]

                new_logp, entropy, new_value = model.evaluate(obs_t[mb_idx], actions_t[mb_idx])
                log_ratio = new_logp - old_logp_t[mb_idx]
                ratio = torch.exp(log_ratio)

                # PPO clipped surrogate objective.
                unclipped = ratio * adv_t[mb_idx]
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_range, 1.0 + config.clip_range
                ) * adv_t[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (ret_t[mb_idx] - new_value).pow(2).mean()
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_mean
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_fracs.append(float(((ratio - 1.0).abs() > config.clip_range).float().mean()))
                    # Approximate KL divergence between old and new policy (update size).
                    approx_kls.append(float(((ratio - 1.0) - log_ratio).mean()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))

        recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else float("nan")
        update_rows.append(
            {
                "step": global_step,
                "mean_recent_reward": recent_reward,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "clip_fraction": float(np.mean(clip_fracs)),
                "approx_kl": float(np.mean(approx_kls)),
            }
        )

        if verbose and len(update_rows) % 20 == 0:
            print(
                f"step={global_step:7d} episodes={len(episode_rewards):5d} "
                f"mean_reward_10={recent_reward:8.2f} "
                f"value_loss={update_rows[-1]['value_loss']:.3f} "
                f"entropy={update_rows[-1]['entropy']:.3f} "
                f"clip_frac={update_rows[-1]['clip_fraction']:.3f}"
            )

    return model, env, episode_rewards, episode_errors, update_rows


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_policy(model, policy: str = "ppo", episodes: int = 50, seed_offset: int = 10_000):
    """Evaluate a policy on fixed held-out seeds.

    policy = "ppo"    -> deterministic PPO policy (Gaussian mean, no sampling)
    policy = "random" -> uniform random torque
    policy = "zero"   -> no torque (passive joint)
    """
    env = CustomContinuousRobotJointEnv(max_steps=200)
    rng = np.random.default_rng(seed_offset)
    returns, errors = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total, done = 0.0, False
        while not done:
            if policy == "ppo":
                with torch.no_grad():
                    action = model.actor(torch.tensor(obs, dtype=torch.float32)).numpy()
            elif policy == "random":
                action = rng.uniform(-env.max_torque, env.max_torque, size=1)
            else:
                action = np.zeros(1)
            action = np.clip(action, -env.max_torque, env.max_torque)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        errors.append(info["angle_error"])
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_final_error_rad": float(np.mean(errors)),
        "success_rate": float(np.mean(np.asarray(errors) < 0.10)),
    }


# -----------------------------
# Output helpers
# -----------------------------
def moving_average(x, window):
    x = np.asarray(x, dtype=np.float32)
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid"), window


def save_outputs(out: Path, model, episode_rewards, episode_errors, update_rows):
    torch.save(model.state_dict(), out / "ppo_robot_joint_model.pt")

    with open(out / "training_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=update_rows[0].keys())
        writer.writeheader()
        writer.writerows(update_rows)

    with open(out / "episode_rewards.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "final_angle_error_rad"])
        for i, (r, e) in enumerate(zip(episode_rewards, episode_errors), start=1):
            writer.writerow([i, r, e])

    # Reward curve.
    rewards_np = np.asarray(episode_rewards, dtype=np.float32)
    moving, window = moving_average(rewards_np, 50)
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(rewards_np) + 1), rewards_np, alpha=0.25, label="Episode reward")
    plt.plot(np.arange(window, len(rewards_np) + 1), moving, linewidth=2, label=f"{window}-episode mean")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO Learning Progress - Continuous Robot Joint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "reward_curve.png", dpi=180)
    plt.close()

    # PPO diagnostics: each metric on its own axis so small values stay visible.
    steps = [r["step"] for r in update_rows]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    panels = [
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy", "Policy entropy"),
        ("clip_fraction", "Clip fraction"),
        ("approx_kl", "Approx. KL (old || new)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(steps, [r[key] for r in update_rows], linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
    ax = axes.flat[5]
    errs_ma, w = moving_average(episode_errors, 50)
    ax.plot(np.arange(w, len(episode_errors) + 1), errs_ma, color="tab:red")
    ax.set_title(f"Final angle error ({w}-ep mean, rad)")
    ax.set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(out / "ppo_diagnostics.png", dpi=180)
    plt.close()


# -----------------------------
# Ablation: contribution of clipping and GAE
# -----------------------------
def run_ablation(base: PPOConfig, out: Path, seeds=(1, 2, 3)):
    """Train each variant on several seeds. Each finished run is cached as JSON in
    outputs/ablation_runs, so an interrupted ablation resumes where it stopped."""
    variants = {
        "PPO (clip 0.2, lambda 0.95)": {},
        "No clipping (clip 10)": {"clip_range": 10.0},
        "Monte Carlo adv. (lambda 1.0)": {"gae_lambda": 1.0},
        "One-step TD adv. (lambda 0)": {"gae_lambda": 0.0},
    }
    run_dir = out / "ablation_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for v_idx, (name, overrides) in enumerate(variants.items()):
        curves[name], finals = [], []
        for seed in seeds:
            cache = run_dir / f"variant{v_idx}_seed{seed}.json"
            if cache.exists():
                res = json.loads(cache.read_text())
            else:
                cfg = replace(base, seed=seed, **overrides)
                model, _, _, _, upd = train(cfg, verbose=False)
                res = {"curve": [r["mean_recent_reward"] for r in upd],
                       "clip_fraction": [r["clip_fraction"] for r in upd],
                       "approx_kl": [r["approx_kl"] for r in upd],
                       "eval": evaluate_policy(model, "ppo")}
                cache.write_text(json.dumps(res))
            curves[name].append(res["curve"])
            finals.append(res["eval"])
            print(f"  {name:32s} seed={seed} eval_return={res['eval']['mean_return']:7.2f} "
                  f"eval_error={res['eval']['mean_final_error_rad']:.3f} "
                  f"mean_approx_kl={np.mean(res['approx_kl']):.4f}")
        rows.append({
            "variant": name,
            "eval_return_mean": float(np.mean([f["mean_return"] for f in finals])),
            "eval_return_std_across_seeds": float(np.std([f["mean_return"] for f in finals])),
            "eval_error_mean_rad": float(np.mean([f["mean_final_error_rad"] for f in finals])),
            "eval_success_rate": float(np.mean([f["success_rate"] for f in finals])),
        })

    with open(out / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    steps = np.arange(1, len(next(iter(curves.values()))[0]) + 1) * base.rollout_steps
    plt.figure(figsize=(8, 4.5))
    for name, runs in curves.items():
        arr = np.asarray([moving_average(c, 10)[0] for c in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = steps[len(steps) - len(mean):]
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Environment steps")
    plt.ylabel("Training return (smoothed)")
    plt.title(f"Ablation: clipping and GAE ({len(seeds)} seeds, mean \u00b1 std)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ablation_curve.png", dpi=180)
    plt.close()
    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="also run the clipping/GAE ablation")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    torch.set_num_threads(1)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = PPOConfig()
    print("Training main PPO agent:", config)
    model, env, ep_rewards, ep_errors, update_rows = train(config)
    save_outputs(out, model, ep_rewards, ep_errors, update_rows)

    # Deterministic evaluation vs. baselines on the same held-out seeds.
    evals = {name: evaluate_policy(model, name) for name in ("ppo", "random", "zero")}

    r = np.asarray(ep_rewards)
    summary = {
        "episodes": len(ep_rewards),
        "total_steps": config.total_steps,
        "first_50_mean_train_return": float(r[:50].mean()),
        "last_50_mean_train_return": float(r[-50:].mean()),
        "first_50_mean_final_error_rad": float(np.mean(ep_errors[:50])),
        "last_50_mean_final_error_rad": float(np.mean(ep_errors[-50:])),
    }
    for name, ev in evals.items():
        for k, v in ev.items():
            summary[f"eval_{name}_{k}"] = v

    with open(out / "training_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("\nTraining summary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.ablation:
        print("\nRunning ablation (this trains 12 agents)...")
        run_ablation(config, out)

    print(f"Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
