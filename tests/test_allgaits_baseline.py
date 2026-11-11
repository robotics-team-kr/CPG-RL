"""Baseline ablation isolation, bounded policy loading and terminal diagnostics."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel

from legged_gym.envs.mini_cheetah.allgaits import MiniCheetahAllGaits
from legged_gym.envs.mini_cheetah.config import MiniCheetahAllGaitsCfg, MiniCheetahAllGaitsPPO
from legged_gym.scripts.play import pose_yaw
from legged_gym.utils.agent_factory import policy_contract, runner_config
from legged_gym.utils.allgaits_profiles import apply_allgaits_profile
from legged_gym.utils.coop_io import load_base
from legged_gym.utils.helpers import class_to_dict, get_args


class BaselineTests(unittest.TestCase):
    def test_profiles_change_only_declared_factors(self):
        base_env, base_train = MiniCheetahAllGaitsCfg(), MiniCheetahAllGaitsPPO()
        for profile in ("entropy0", "bounded", "bounded_yaw"):
            env, train = MiniCheetahAllGaitsCfg(), MiniCheetahAllGaitsPPO()
            apply_allgaits_profile(env, train, profile)
            expected_env, expected_train = class_to_dict(base_env), class_to_dict(base_train)
            expected_train["baseline_profile"] = profile
            expected_train["algorithm"]["entropy_coef"] = 0.0
            expected_train["policy"]["action_distribution"] = "gaussian" if profile == "entropy0" else "beta"
            if profile == "bounded_yaw":
                expected_env["rewards"]["scales"]["tracking_yaw"] = 1.0
            self.assertEqual(class_to_dict(env), expected_env)
            self.assertEqual(class_to_dict(train), expected_train)

    def test_profile_cannot_override_resumed_policy(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args(["--task", "mini_cheetah_allgaits", "--allgaits_profile", "bounded", "--resume"])

    def test_bounded_policy_gradient_export_and_base_loading(self):
        torch.set_num_threads(1)
        cfg, train = MiniCheetahAllGaitsCfg(), MiniCheetahAllGaitsPPO()
        cfg.env.num_envs = 2
        train.policy.actor_hidden_dims = [32, 32]
        apply_allgaits_profile(cfg, train, "bounded")
        env = MiniCheetahAllGaits(cfg)
        try:
            options = runner_config(train)["actor"].copy()
            options.pop("class_name")
            with contextlib.redirect_stdout(io.StringIO()):
                actor = MLPModel(env.get_observations(), {"actor": ["policy"]}, "actor", 8, **options)
            obs = env.get_observations()
            action = actor(obs, stochastic_output=True)
            self.assertTrue((action.abs() <= 1).all())
            loss = -actor.get_output_log_prob(action).mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
            actor.eval()
            expected = actor(obs).detach()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                contract = policy_contract(env)
                torch.save(
                    {"actor_state_dict": actor.state_dict(), "infos": {"legged_gym": {"contract": contract}}}, path
                )
                (path.parent / "run_config.json").write_text(
                    json.dumps(
                        dict(
                            task="mini_cheetah_allgaits",
                            env=class_to_dict(cfg),
                            train=class_to_dict(train),
                            contract=contract,
                        )
                    )
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    frozen, _, _ = load_base(path, 2)
                torch.testing.assert_close(frozen(obs["policy"]), expected)
        finally:
            env.close()

    def test_action_metrics_and_terminal_pose_survive_reset(self):
        cfg = MiniCheetahAllGaitsCfg()
        cfg.env.num_envs = 2
        cfg.env.episode_length_s = 0.01
        env = MiniCheetahAllGaits(cfg)
        try:
            _, _, done, info = env.step(np.full((2, 8), 3.0))
            self.assertTrue(done.all())
            self.assertEqual(info["log"]["Diagnostics/action_clipped_fraction"], 1.0)
            self.assertEqual(info["log"]["Diagnostics/raw_action_abs_mean"], 3.0)
            self.assertFalse(np.array_equal(env.last_diagnostic_pose, env.sim.qpos[:, :7]))
            self.assertTrue((env.raw_actions == 0).all())
        finally:
            env.close()

    def test_yaw_wrap_uses_short_angular_difference(self):
        angle = np.deg2rad([179, -179])
        pose = np.zeros((2, 7))
        pose[:, 3], pose[:, 6] = np.cos(angle / 2), np.sin(angle / 2)
        yaw = pose_yaw(pose)
        delta = (yaw[1] - yaw[0] + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(np.degrees(delta), 2.0)


if __name__ == "__main__":
    unittest.main()
