# PPO for Continuous-Control Robotics

**Robot-joint reaching with Proximal Policy Optimization**
Reinforcement Learning Project, Part 2: PhDAI-830-A02 Applied Machine Intelligence and Reinforcement Learning, University of the Cumberlands
Author: Subash Kambala

A single rotary robot joint must be driven to a randomly chosen target angle and held there, using a continuous motor torque in [-2, 2]. The agent is trained from scratch with PPO (actor-critic, clipped surrogate objective, GAE). It is evaluated on held-out episodes against random-torque and zero-torque baselines, with a 3-seed ablation of clipping and GAE.

## Repository contents

| Path | What it is |
|---|---|
| `ppo_robotics_project.py` | Environment, actor-critic model, PPO training, evaluation, plots, ablation |
| `make_demo_video.py` | Renders the demo video of the trained policy on held-out episodes |
| `requirements.txt` | Python dependencies |
| `outputs/` | Results from the reported run (model, metrics, plots, ablation, demo video) |
| `presentation/` | Slide deck and video presenter guide |

## Setup

Requires Python 3.10 or newer (tested with Python 3.13 on macOS, CPU only).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
# 1. Train the main agent (seed 42, 204,800 steps), evaluate it, and write plots/metrics
python ppo_robotics_project.py

# 2. Optional: also run the clipping / GAE ablation (12 extra training runs, 3 seeds x 4 variants)
python ppo_robotics_project.py --ablation

# 3. Render the demo video from the saved model
python make_demo_video.py
```

Everything is written to `outputs/`. Seeds are fixed, so a rerun on the same software versions reproduces the reported numbers; other PyTorch or NumPy versions may give slightly different values. The ablation caches each finished run in `outputs/ablation_runs/`, so an interrupted ablation picks up where it stopped.

## Method

**Environment** (`CustomContinuousRobotJointEnv`, Gymnasium `reset` / `step` API)

- **State (5-D):** `[sin θ, cos θ, ω/ω_max, sin e, cos e]`, where `e = θ − target`. The target is given *relative* to the joint, and sin/cos avoid the jump at ±π.
- **Action:** one continuous torque, clipped to [-2, 2].
- **Dynamics:** inertia, damping and a small gravity term; dt = 0.05.
- **Reward:** `1 − 1.4(e/π)² − 0.04ω² − 0.002τ²`, plus a bonus of 0.25 when `|e| < 0.1` and `|ω| < 0.2`.
- **Episodes:** random start angle, velocity and target. A 200-step time limit (truncation, not failure).

**Agent**

- Separate actor and critic MLPs (5 → 64 → 64, tanh). The actor outputs a Gaussian mean with a learnable log-std.
- PPO clipped surrogate (clip 0.2), GAE (γ = 0.99, λ = 0.95), and advantage normalization.
- The value of the final state is bootstrapped at the time limit (Pardo et al., 2018), so truncation is not treated as a terminal state.
- 512-step rollouts, 5 epochs of 64-sample mini-batches, Adam (lr 3e-4), gradient-norm clip 0.5, entropy coefficient 0.

## Results

Values below come from the files in `outputs/` (seed 42, PyTorch 2.14.1, NumPy 2.5.4, Python 3.13, macOS, CPU).

**Training** (1,024 episodes, 204,800 steps). See `outputs/reward_curve.png` and `outputs/ppo_diagnostics.png`.

| | First 50 episodes | Last 50 episodes |
|---|---|---|
| Mean return | 104.5 | 227.1 |
| Mean final angle error | 1.54 rad | 0.024 rad |

Value loss falls from a peak of about 57 to under 0.3. Policy entropy drops from 0.92 to 0.56. Approximate KL stays below 0.03 on every update, and the clip fraction is mostly between 0.01 and 0.18.

**Held-out evaluation** (50 episodes, seeds 10000–10049, never used in training; success = final error < 0.1 rad):

| Policy | Mean return | Mean final error | Success |
|---|---|---|---|
| PPO (deterministic) | 231.2 ± 14.7 | 0.023 rad | **100%** |
| Random torque | 116.3 ± 50.8 | 1.44 rad | 4% |
| Zero torque | 113.4 ± 55.8 | 1.40 rad | 2% |

**Ablation** (mean of seeds 1, 2 and 3; `outputs/ablation_results.csv`, `outputs/ablation_curve.png`):

| Variant | Eval return | Final error | Success |
|---|---|---|---|
| PPO (clip 0.2, λ = 0.95) | 229.7 | 0.018 rad | 100% |
| No clipping (clip range 10) | 184.3 | 0.395 rad | 16% |
| Monte Carlo advantages (λ = 1.0) | 215.6 | 0.032 rad | 100% |
| One-step TD advantages (λ = 0) | 187.2 | 0.238 rad | 18% |

Without clipping the mean approximate KL per update is about 18 to 73 times larger and success drops sharply. λ = 0 depends too heavily on the early, inaccurate critic. λ = 1 still succeeds but reaches a lower return and higher variance than λ = 0.95.

**Demo** (`outputs/ppo_robot_joint_demo.mp4`, 23 s): the first four held-out seeds (10000–10003), not hand-picked. All four end within 0.037 rad of the target, including seed 10001, which starts 2.36 rad away.

## Changes from the Part 1 plan

| Early version | Fix | Why |
|---|---|---|
| Target given as a raw angle | Target given relative to the joint (sin/cos of the error) | The network had to learn wrapped angle subtraction itself; the early run barely beat random (p = 0.54) |
| Time limit treated as a terminal state | Bootstrap V(s_T) at truncation | The task never fails; zeroing the value at step 200 corrupted the critic's targets |
| 32-step leftover final batch | `total_steps` is an exact multiple of the rollout length | The small batch caused a misleading value-loss spike |
| 20,000 steps, training curve only | 204,800 steps, held-out evaluation, baselines, 3-seed ablation, demo | To show convergence and measure performance properly |

## Limitations

- Simplified single-joint physics, not a full robot simulator.
- A small steady-state offset (about 0.02 rad): the policy holds a constant torque against gravity.
- The main run uses one seed; the ablation uses three.
- No comparison with SAC or DDPG, and no sensor-noise or disturbance tests.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *IJRR, 32*(11), 1238–1274.
- Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). Time limits in reinforcement learning. *ICML 2018*, 4045–4054.

## AI use disclosure

ChatGPT helped organize the project, review PPO/GAE concepts and improve clarity. Claude (Anthropic) reviewed the code and results, identified the state-representation, truncation and batch issues, helped write the evaluation, ablation and demo scripts, and helped prepare this repository and the slides. The author reviewed all code, reran training, and checked every reported number.
