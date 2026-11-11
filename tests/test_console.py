import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from cpg_console.cli import parse_args
from cpg_console.controls import Controls, axis_value, joystick_velocity
from cpg_console.runtime import Runtime, find_checkpoint
from legged_gym.envs import task_registry
from legged_gym.envs.base.cpg import inverse_kinematics
from legged_gym.utils.agent_factory import create_agent

BOUNDS = np.array([[-0.5, 0.8], [-0.3, 0.3], [-0.6, 0.6]])


@pytest.fixture(scope="module")
def checkpoint():
    root = Path(__file__).resolve().parents[1] / "logs"
    try:
        return find_checkpoint(root)
    except FileNotFoundError:
        pytest.skip("Local training checkpoint is not distributed")


def test_gamepad_mapping_and_stop():
    np.testing.assert_allclose(joystick_velocity([1, -1, 0.05], BOUNDS, 0.12), [0.8, -0.3, 0])
    assert axis_value(float("nan"), 0.12) == 0
    controls = Controls(BOUNDS)
    controls.joystick([0.8, 0, 0], True)
    np.testing.assert_allclose(controls.advance(0.01, 1), [0.01, 0, 0])
    controls.joystick([0.8, 0, 0], False)
    np.testing.assert_array_equal(controls.advance(0.01, 1), 0)
    controls.joystick([0.8, 0, 0], True)
    controls.key(" ", joystick=True)
    controls.joystick([0.8, 0, 0], True)
    np.testing.assert_array_equal(controls.advance(0.01, 1), 0)
    controls.joystick([0, 0, 0], True)
    controls.joystick([0.8, 0, 0], True)
    assert controls.advance(0.01, 1)[0] > 0
    controls.key("p")
    controls.joystick([0.8, 0, 0], True)
    np.testing.assert_array_equal(controls.advance(0.01, 1), 0)
    controls.key("p")
    np.testing.assert_array_equal(controls.advance(0.01, 1), 0)


def test_keyboard_limits_and_reset():
    controls = Controls(BOUNDS)
    for _ in range(20):
        controls.key("w")
        controls.key("d")
        controls.key("q")
    np.testing.assert_allclose(controls.target, [0.8, -0.3, 0.6])
    controls.key("r")
    assert controls.reset_requested
    np.testing.assert_array_equal(controls.command, 0)
    controls.key("\x1b")
    assert controls.quit


def test_forward_only_policy_does_not_map_reverse_stick_to_forward():
    bounds = np.array([[0.2, 0.8], [0, 0], [0, 0]])
    np.testing.assert_array_equal(joystick_velocity([-1, 1, -1], bounds, 0.12), 0)


@pytest.mark.parametrize("argv", [
    ["--steps", "0"], ["--deadzone", "1"], ["--velocity", "nan", "0", "0"],
    ["--fast"], ["--joystick", "--velocity", "1", "0", "0"],
    ["--demo", "stand", "--velocity", "1", "0", "0"], ["--axis-signs", "1", "0", "1"],
])
def test_invalid_cli(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_checkpoint_discovery_numeric_order(tmp_path):
    run = tmp_path / "2026-11-09"
    run.mkdir()
    (run / "run_config.json").write_text('{"task": "mini_cheetah_cpg"}')
    for name in ("model_9.pt", "model_1100.pt", "model_final.pt"):
        (run / name).touch()
    assert find_checkpoint(tmp_path).name == "model_1100.pt"
    newer = tmp_path / "2026-11-10"
    newer.mkdir()
    (newer / "run_config.json").write_text('{"task": "mini_cheetah_allgaits"}')
    (newer / "model_1.pt").touch()
    assert find_checkpoint(tmp_path).parent == newer
    assert find_checkpoint(tmp_path, "mini_cheetah_cpg").parent == run


def test_ik_matches_mujoco_foot_centers():
    runtime = Runtime(demo="stand")
    try:
        env = runtime.env
        data = mujoco.MjData(env.sim.model)
        data.qpos[:7] = [0, 0, 0.5, 1, 0, 0, 0]
        rng = np.random.default_rng(8)
        for _ in range(20):
            feet = np.zeros((1, 4, 3))
            feet[0, :, 0] = rng.uniform(-0.05, 0.05, 4)
            feet[0, :, 1] = np.array([-1, 1, -1, 1]) * env.cfg.cpg.link_lengths[0] + rng.uniform(-0.02, 0.02, 4)
            feet[0, :, 2] = rng.uniform(-0.30, -0.25, 4)
            data.qpos[env.sim.qpos_ids] = inverse_kinematics(feet, env.cfg.cpg.link_lengths)[0]
            mujoco.mj_forward(env.sim.model, data)
            origins = data.xanchor[env.sim.joint_ids[::3]]
            np.testing.assert_allclose(data.site_xpos[env.sim.site_ids] - origins, feet[0], atol=1e-8)
    finally:
        runtime.close()


def test_stand_and_fall_freeze():
    runtime = Runtime(demo="stand")
    try:
        for _ in range(200):
            runtime.step()
        assert not runtime.failed
        assert runtime.env.base_height[0] > 0.24
        runtime.env.sim.qpos[0, 3:7] = [np.cos(0.8), np.sin(0.8), 0, 0]
        runtime.step()
        assert runtime.failed
        pose = runtime.env.sim.qpos.copy()
        runtime.step()
        np.testing.assert_array_equal(runtime.env.sim.qpos, pose)
        runtime.reset()
        assert not runtime.failed
        np.testing.assert_array_equal(runtime.command, 0)
    finally:
        runtime.close()


def test_checkpoint_actor_parity_and_command_persistence(checkpoint):
    torch.set_num_threads(1)
    runtime = Runtime(checkpoint)
    try:
        _, cfg = task_registry.get_cfgs(runtime.task)
        from legged_gym.utils.helpers import update_class_from_dict

        update_class_from_dict(cfg, json.loads((checkpoint.parent / "run_config.json").read_text())["train"])
        runner = create_agent(runtime.env, cfg, device="cpu")
        runner.load(str(checkpoint), restore_env=False)
        reference = runner.get_inference_policy(device="cpu")
        for _ in range(600):
            runtime.set_command([0.5, 0.1, -0.1])
            with torch.inference_mode():
                obs = runtime.env.get_observations()
                torch.testing.assert_close(runtime.policy(obs), reference(obs), rtol=0, atol=0)
            runtime.step()
            assert not runtime.failed
            np.testing.assert_array_equal(runtime.env.commands[0], [0.5, 0.1, -0.1])
            np.testing.assert_allclose(runtime.env.get_observations()["policy"][0, :3], [0.5, 0.1, -0.1])
    finally:
        runtime.close()


def test_checkpoint_mismatch(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="task"):
        Runtime(checkpoint, task="mini_cheetah_allgaits")
    config = json.loads((checkpoint.parent / "run_config.json").read_text())
    config["env"]["cpg"]["height"] += 0.01
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    path = tmp_path / checkpoint.name
    path.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match="contract"):
        Runtime(path)
