"""Research correctness tests, not evidence of learned locomotion performance."""

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel

from legged_gym.algorithms.cooperative_ppo import CooperativePPO, PPOSettings
from legged_gym.algorithms.multi_rate_rollout import duration_gae
from legged_gym.envs.base.allgaits import GAIT_CONTACT_TIMES
from legged_gym.envs.mini_cheetah.allgaits import MiniCheetahAllGaits
from legged_gym.envs.mini_cheetah.config import MiniCheetahAllGaitsCfg, MiniCheetahAllGaitsPPO
from legged_gym.envs.mini_cheetah.coop_allgaits import MiniCheetahCoopAllGaits
from legged_gym.envs.mini_cheetah.coop_config import CoopSettings
from legged_gym.scripts.analyze_coop import paired_bootstrap
from legged_gym.utils.agent_factory import policy_contract, runner_config
from legged_gym.utils.coop_events import ContactEvents
from legged_gym.utils.coop_io import load_base, module_hash
from legged_gym.utils.helpers import class_to_dict


def config(n=2):
    cfg = MiniCheetahAllGaitsCfg()
    cfg.env.num_envs = n
    cfg.seed = 123
    cfg.gait.name = "trot"
    return cfg


def fixture_base(directory):
    """Untrained synthetic checkpoint used ONLY to exercise loading and algorithms."""
    cfg, train = config(), MiniCheetahAllGaitsPPO()
    train.policy.actor_hidden_dims = [32, 32]
    env = MiniCheetahAllGaits(cfg)
    try:
        actor_cfg = runner_config(train)["actor"].copy()
        actor_cfg.pop("class_name")
        actor = MLPModel(env.get_observations(), {"actor": ["policy"]}, "actor", 8, **actor_cfg)
        contract = policy_contract(env)
        path = Path(directory) / "model.pt"
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "infos": {"legged_gym": {"contract": contract}},
                "iter": 0,
                "untrained_test_fixture": True,
            },
            path,
        )
        (path.parent / "run_config.json").write_text(
            json.dumps(
                {
                    "task": "mini_cheetah_allgaits",
                    "env": class_to_dict(env.cfg),
                    "train": class_to_dict(train),
                    "contract": contract,
                }
            )
        )
        return path
    finally:
        env.close()


class CooperativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.base_path = fixture_base(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def runner(self, stage="cooperative_spatial", settings=None):
        with contextlib.redirect_stdout(io.StringIO()):
            base, cfg, audit = load_base(self.base_path, 2)
        cfg.env.episode_length_s = 0.11
        env = MiniCheetahCoopAllGaits(cfg, settings or CoopSettings())
        return CooperativePPO(
            env, base, audit, PPOSettings(rollout_steps=13, hidden=(32, 32), epochs=2, minibatches=2), stage=stage
        )

    def test_legacy_identity_ten_thousand_ticks(self):
        cfg = config()
        old = MiniCheetahAllGaits(deepcopy(cfg))
        new = MiniCheetahCoopAllGaits(deepcopy(cfg), CoopSettings(enabled=False))
        rng = np.random.default_rng(9)
        try:
            for _ in range(1000):
                action = rng.uniform(-1, 1, (2, 8))
                a, ar, ad, _ = old.step(action)
                b, br, bd, _ = new.step_coop(action)
                np.testing.assert_array_equal(old.sim.qpos, new.sim.qpos)
                np.testing.assert_array_equal(a["policy"], b["base"])
                np.testing.assert_array_equal(ar, br)
                np.testing.assert_array_equal(ad, bd)
        finally:
            old.close()
            new.close()

    def test_all_gaits_and_shapes(self):
        for gait in GAIT_CONTACT_TIMES:
            cfg = config(1)
            cfg.gait.name = gait
            env = MiniCheetahCoopAllGaits(cfg)
            try:
                obs = env.observations_coop()
                self.assertEqual(
                    {k: v.shape[-1] for k, v in obs.items()}, {"base": 68, "slow": 102, "reflex": 121, "critic": 147}
                )
                self.assertEqual(env.cpg.gait_names[env.cpg.gait_ids[0]], gait)
            finally:
                env.close()

    def test_phase_bound_mapping_and_causality(self):
        env = MiniCheetahCoopAllGaits(config(), CoopSettings(phase_enabled=True))
        try:
            env.set_slow(np.ones((2, 8)), [0, 1])
            np.testing.assert_allclose(env.observations_coop()["reflex"][:, 68:76], np.tanh(1), rtol=1e-6)
            env.set_reflex(np.full((2, 20), 100.0), [0, 1])
            self.assertLess(np.abs(env.phase_target).max(), np.pi)
            self.assertTrue(np.all(np.abs(env.foot_target) <= env.settings.foot_scale))
            for _ in range(20):
                env.step_coop(np.zeros((2, 8)))
                self.assertTrue(np.isfinite(env.cpg.theta).all())
                self.assertTrue(((env.cpg.theta >= 0) & (env.cpg.theta < 2 * np.pi)).all())
        finally:
            env.close()

    def test_contact_debounce_guard(self):
        events = ContactEvents(1)
        events.guard[:] = 0
        phase = np.full((1, 4), np.pi / 2)
        for _ in range(2):
            events.update(np.ones((1, 4), bool), phase, 0.001, np.array([True]))
        self.assertFalse(events.contact.any())
        for _ in range(5):
            events.update(np.ones((1, 4), bool), phase, 0.001, np.array([True]))
        self.assertTrue(events.early.all())
        events.reset([0])
        for _ in range(20):
            events.update(np.ones((1, 4), bool), phase, 0.001, np.array([True]))
        self.assertFalse(events.early.any())

    def test_duration_gae_timeout_does_not_leak_reset(self):
        gamma = 0.9
        macro = sum(gamma**i for i in range(5))
        advantages, returns = duration_gae(
            [macro, 100], [0, 0], [2, 0], [5, 1], [False, True], [True, True], gamma, 0.95
        )
        self.assertAlmostEqual(returns[0], macro + gamma**5 * 2)
        self.assertEqual(advantages[1], 100)
        _, returns = duration_gae([1 + gamma], [0], [999], [2], [True], [True], gamma, 0.95)
        self.assertAlmostEqual(returns[0], 1 + gamma)

    def test_paused_simulation_does_not_advance(self):
        env = MiniCheetahCoopAllGaits(config())
        try:
            saved = env.sim.qpos[1].copy(), env.cpg.theta[1].copy()
            env.set_slow(np.zeros((1, 8)), [0])
            env.set_reflex(np.zeros((1, 20)), [0])
            env.step_coop(np.zeros((2, 8)), [True, False])
            np.testing.assert_array_equal(saved[0], env.sim.qpos[1])
            np.testing.assert_array_equal(saved[1], env.cpg.theta[1])
            self.assertEqual(int(env.episode_length_buf[1]), 0)
        finally:
            env.close()

    def test_real_terrain_contact_and_lowered_geometry(self):
        for terrain in ("raised", "lowered", "irregular", "friction"):
            env = MiniCheetahCoopAllGaits(config(1), CoopSettings(terrain=terrain, fixed_start=True))
            try:
                # Drop the feet onto a course surface that is not the floor plane.
                env.sim.qpos[0, 0] = 2.0
                for _ in range(30):
                    env.step_coop(np.zeros((1, 8)))
                    if env.sim.foot_contacts[:, :, 0].any():
                        break
                self.assertTrue(env.sim.foot_contacts[:, :, 0].any(), terrain)
                self.assertLess(env.sim.model.geom_pos[env.sim.model.geom("floor").id, 2], 0)
            finally:
                env.close()

    def test_async_macros_gradients_and_frozen_base(self):
        runner = self.runner()
        try:
            runner.env.episode_length_buf[0] = runner.env.max_episode_length - 2
            before = module_hash(runner.base)
            buffers, episodes, steps = runner.collect()
            self.assertIn(2, runner.last_durations["slow"])
            self.assertEqual(sum(runner.last_durations["slow"]), steps)
            self.assertEqual(sum(runner.last_durations["reflex"]), steps)
            self.assertFalse(runner.env.slow_countdown.any())
            stats = runner.update(buffers)
            self.assertGreater(stats["slow_grad_norm"], 0)
            self.assertGreater(stats["reflex_grad_norm"], 0)
            self.assertEqual(before, module_hash(runner.base))
            self.assertTrue(episodes)
        finally:
            runner.env.close()

    def test_exact_resume(self):
        a = self.runner()
        b = None
        try:
            buffers, _, _ = a.collect()
            a.update(buffers)
            path = Path(self.directory.name) / "resume.pt"
            a.save(path)
            buffers_a, _, _ = a.collect()
            stats_a = a.update(buffers_a)
            qpos = a.env.sim.qpos.copy()
            b = self.runner()
            b.load(path)
            buffers_b, _, _ = b.collect()
            stats_b = b.update(buffers_b)
            np.testing.assert_array_equal(qpos, b.env.sim.qpos)
            self.assertEqual(stats_a, stats_b)
            for role in a.agents:
                self.assertEqual(module_hash(a.agents[role]), module_hash(b.agents[role]))
        finally:
            a.env.close()
            if b:
                b.env.close()

    def test_frozen_branch(self):
        runner = self.runner(stage="sequential")
        try:
            original = module_hash(runner.agents["reflex"])
            buffers, _, _ = runner.collect()
            runner.update(buffers)
            self.assertEqual(original, module_hash(runner.agents["reflex"]))
        finally:
            runner.env.close()

    def test_monolithic_and_same_rate(self):
        for stage, settings in [
            ("monolithic", CoopSettings()),
            ("same_rate_20", CoopSettings(slow_interval=5, reflex_interval=5)),
        ]:
            runner = self.runner(stage, settings)
            try:
                buffers, _, steps = runner.collect()
                stats = runner.update(buffers)
                self.assertTrue(np.isfinite(stats["reflex_loss"]))
                self.assertEqual(sum(runner.last_durations["reflex"]), steps)
            finally:
                runner.env.close()

    def test_bootstrap_units(self):
        rows = [
            dict(method=m, training_seed=str(seed), scenario=str(i), completed=str(value))
            for seed in range(3)
            for i in range(4)
            for m, value in [("base", False), ("coop", True)]
        ]
        result = paired_bootstrap(rows, "base", "coop", "completed", samples=100)
        self.assertEqual(result["delta"], 1)
        self.assertEqual(result["training_seeds"], 3)

    def test_joint_ppo_counts_slow_decisions_once(self):
        runner = self.runner("monolithic")
        try:
            buffers, _, _ = runner.collect()
            data = buffers["reflex"].tensors("cpu")
            self.assertEqual(int(data["slow_due"].sum()), len(runner.last_durations["slow"]))
            with torch.no_grad():
                r = runner.shared.distribution(data["obs"], "reflex")
                s = runner.shared.distribution(data["slow_obs"], "slow")
                logp = r.log_prob(data["action"]).sum(-1) + data["slow_due"] * s.log_prob(data["slow_action"]).sum(-1)
            torch.testing.assert_close(logp, data["logp"])
        finally:
            runner.env.close()

    def test_popart_preserves_values(self):
        runner = self.runner()
        try:
            agent = runner.agents["slow"]
            state = runner.env.observations_coop()["critic"]
            before = agent.value(state).detach().clone()
            agent.normalize_returns(torch.tensor([-10.0, 20.0, 40.0]), runner.optimizers["slow"])
            torch.testing.assert_close(before, agent.value(state), atol=2e-6, rtol=2e-5)
        finally:
            runner.env.close()

    def test_random_terrain_and_dynamics_resume(self):
        settings = CoopSettings(
            terrain="irregular",
            randomize_terrain=True,
            randomize_dynamics=True,
            fixed_start=True,
            contact_delay_steps=1,
            contact_dropout=0.1,
            velocity_noise=0.02,
        )
        a = self.runner(settings=settings)
        b = None
        try:
            buffers, _, _ = a.collect()
            a.update(buffers)
            path = Path(self.directory.name) / "terrain_resume.pt"
            a.save(path)
            buffers, _, _ = a.collect()
            expected = a.update(buffers)
            b = self.runner(settings=settings)
            b.load(path)
            buffers, _, _ = b.collect()
            actual = b.update(buffers)
            self.assertEqual(expected, actual)
            np.testing.assert_array_equal(a.env.sim.qpos, b.env.sim.qpos)
        finally:
            a.env.close()
            if b:
                b.env.close()

    def test_force_energy_and_observation_corruption(self):
        clean = MiniCheetahCoopAllGaits(config(1), CoopSettings())
        pushed = MiniCheetahCoopAllGaits(
            config(1), CoopSettings(push_start=0, push_force=(0, 50, 0), contact_dropout=1.0)
        )
        try:
            for _ in range(5):
                clean.step_coop(np.zeros((1, 8)))
                obs, _, _, _ = pushed.step_coop(np.zeros((1, 8)))
            self.assertGreater(np.linalg.norm(clean.sim.qvel - pushed.sim.qvel), 1e-4)
            self.assertTrue(np.all(pushed.absolute_energy >= pushed.positive_energy))
            self.assertTrue(torch.equal(obs["base"], obs["reflex"][:, :68]))
            self.assertEqual(float(obs["base"][:, 12:16].sum()), 0.0)
            self.assertFalse(pushed.events.contact.any())
        finally:
            clean.close()
            pushed.close()


if __name__ == "__main__":
    unittest.main()
