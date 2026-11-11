"""Render a deterministic simulation video; requires the optional analysis dependencies."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import torch

from legged_gym.algorithms.cooperative_ppo import PPOSettings
from legged_gym.envs.mini_cheetah.coop_config import CoopSettings
from legged_gym.scripts.train_coop import make_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gait", default="trot")
    parser.add_argument("--terrain", default="irregular")
    parser.add_argument("--terrain-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, weights_only=False, map_location="cpu") if args.checkpoint else None
    options = saved["settings"].copy() if saved else {}
    options.update(
        fixed_start=True,
        gait_names=[args.gait],
        terrain=args.terrain,
        terrain_seed=args.terrain_seed,
        randomize_terrain=False,
    )
    runner = make_runner(
        args.base_checkpoint,
        CoopSettings(**options),
        PPOSettings(**saved["params"]) if saved else PPOSettings(),
        saved["stage"] if saved else "base",
        1,
        args.seed,
        "cpu",
        args.seconds,
        args.speed,
    )
    renderer = writer = None
    try:
        if args.checkpoint:
            runner.load(args.checkpoint, resume=False)
        runner.env.capture_render = True
        model = runner.env.sim.batch.get_model(0)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 2.5, 130, -20
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=25)
        for tick in range(runner.env.max_episode_length):
            _, _, done, info = runner.act_step()
            if tick % 4 == 0 or done[0]:
                qpos, qvel = info["render_state"]
                data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
            if done[0]:
                path.with_suffix(".json").write_text(json.dumps(info["episodes"][0], indent=2) + "\n")
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        runner.env.close()


if __name__ == "__main__":
    main()
