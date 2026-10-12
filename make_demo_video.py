"""Demo video of the trained PPO policy on the robot-joint task.

Loads outputs/ppo_robot_joint_model.pt (written by ppo_robotics_project.py), runs
the deterministic policy (Gaussian mean, no sampling) on the first four held-out
evaluation seeds (10000-10003, the same seeds used by evaluate_policy), and renders
outputs/ppo_robot_joint_demo.mp4 plus a poster frame outputs/demo_frame.png.

The episodes are not hand-picked: they are simply the first four evaluation seeds.

Run (after training):
    python make_demo_video.py
"""

from pathlib import Path
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch

from ppo_robotics_project import ActorCritic, CustomContinuousRobotJointEnv, PPOConfig

TEAL = "#0f7b7b"
ORANGE = "#e3a008"
RED = "#c0392b"
GREY = "#d0d0d0"
SUCCESS_RAD = 0.10


def configure_ffmpeg() -> None:
    """Use a system ffmpeg if present, otherwise the binary bundled with imageio-ffmpeg."""
    if FFMpegWriter.isAvailable():
        return
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit("ffmpeg not found: install it or run `pip install imageio-ffmpeg`") from exc


def rollout(model, seed: int):
    """Run one deterministic episode and record angle, target, error, torque and reward."""
    env = CustomContinuousRobotJointEnv(max_steps=200)
    obs, _ = env.reset(seed=seed)
    angles, errors, torques, rewards = [env.angle], [], [], []
    done = False
    while not done:
        with torch.no_grad():
            action = model.actor(torch.tensor(obs, dtype=torch.float32)).numpy()
        action = np.clip(action, -env.max_torque, env.max_torque)
        obs, reward, terminated, truncated, info = env.step(action)
        angles.append(info["angle"])
        errors.append(info["angle_error"])
        torques.append(float(action[0]))
        rewards.append(reward)
        done = terminated or truncated
    return {
        "seed": seed,
        "target": env.target,
        "angles": np.asarray(angles[1:]),
        "start_error": abs(env._angle_difference(angles[0], env.target)),
        "errors": np.asarray(errors),
        "torques": np.asarray(torques),
        "returns": np.cumsum(rewards),
    }


def arm_xy(angle: float):
    # Angle 0 is the gravity-stable position, drawn hanging straight down.
    return math.sin(angle), -math.cos(angle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/ppo_robot_joint_model.pt")
    parser.add_argument("--output", default="outputs/ppo_robot_joint_demo.mp4")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=10_000)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--hold-frames", type=int, default=30, help="pause after each episode")
    args = parser.parse_args()

    configure_ffmpeg()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = CustomContinuousRobotJointEnv()
    model = ActorCritic(env.observation_dim, env.action_dim, PPOConfig().hidden_size)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    episodes = [rollout(model, args.seed_offset + i) for i in range(args.episodes)]
    for ep in episodes:
        print(f"seed {ep['seed']}: start error {ep['start_error']:.2f} rad -> "
              f"final error {ep['errors'][-1]:.3f} rad, return {ep['returns'][-1]:.1f}")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax_arm = fig.add_axes([0.03, 0.08, 0.50, 0.80])
    ax_err = fig.add_axes([0.60, 0.56, 0.36, 0.32])
    ax_tq = fig.add_axes([0.60, 0.12, 0.36, 0.32])
    title = fig.text(0.03, 0.93, "", fontsize=14)
    status = fig.text(0.05, 0.085, "", fontsize=12, family="monospace")

    steps = np.arange(1, 201)
    writer = FFMpegWriter(fps=args.fps, metadata={"title": "PPO robot joint demonstration"},
                          extra_args=["-pix_fmt", "yuv420p"])
    poster_saved = False
    with writer.saving(fig, str(out_path), dpi=100):
        for idx, ep in enumerate(episodes):
            # Static parts of the frame for this episode.
            ax_arm.clear()
            ax_arm.set_xlim(-1.35, 1.35)
            ax_arm.set_ylim(-1.35, 1.35)
            ax_arm.set_aspect("equal")
            ax_arm.axis("off")
            ax_arm.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls=":", color="#aaaaaa"))
            tx, ty = arm_xy(ep["target"])
            ax_arm.plot([0, tx], [0, ty], "--", color=ORANGE, lw=2)
            ax_arm.plot(tx, ty, "X", color=ORANGE, ms=16, label="Target")
            (arm,) = ax_arm.plot([], [], color=TEAL, lw=14, solid_capstyle="round", label="Joint (arm)")
            ax_arm.plot(0, 0, "o", color="#1f2937", ms=20, zorder=5)
            ax_arm.legend(loc="upper right", frameon=False, fontsize=11)

            for ax, data, label in ((ax_err, ep["errors"], "Angle error (rad)"),
                                    (ax_tq, ep["torques"], "Applied torque")):
                ax.clear()
                ax.plot(steps, data, color=GREY, lw=1.5)
                ax.set_xlim(0, 200)
                ax.set_title(label)
            ax_err.axhline(SUCCESS_RAD, ls="--", color="#555555", lw=1)
            ax_err.text(198, SUCCESS_RAD + 0.03, f"success < {SUCCESS_RAD} rad", ha="right", fontsize=9)
            ax_tq.set_ylim(-2.2, 2.2)
            ax_tq.set_xlabel("Environment step")
            (err_line,) = ax_err.plot([], [], color=RED, lw=2)
            (tq_line,) = ax_tq.plot([], [], color=TEAL, lw=2)
            title.set_text(f"Episode {idx + 1} of {len(episodes)}  |  held-out seed {ep['seed']}  |  "
                           "deterministic PPO policy")

            n = len(ep["errors"])
            for t in list(range(n)) + [n - 1] * args.hold_frames:
                x, y = arm_xy(ep["angles"][t])
                arm.set_data([0, x], [0, y])
                err_line.set_data(steps[: t + 1], ep["errors"][: t + 1])
                tq_line.set_data(steps[: t + 1], ep["torques"][: t + 1])
                status.set_text(f"step {t + 1}/{n}   angle error {ep['errors'][t]:.3f} rad   "
                                f"return {ep['returns'][t]:.1f}")
                writer.grab_frame()
                if not poster_saved and idx == 1 and t == 34:
                    fig.savefig(out_path.parent / "demo_frame.png", dpi=100)
                    poster_saved = True
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
