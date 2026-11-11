"""Opt-in baseline ablations; never modify the legacy default or a resumed run."""


def apply_allgaits_profile(env_cfg, train_cfg, profile):
    if profile not in ("entropy0", "bounded", "bounded_yaw"):
        raise ValueError(f"Unknown AllGaits profile: {profile}")
    train_cfg.baseline_profile = profile
    train_cfg.algorithm.entropy_coef = 0.0
    train_cfg.policy.action_distribution = "gaussian" if profile == "entropy0" else "beta"
    if profile == "bounded_yaw":
        env_cfg.rewards.scales = {**env_cfg.rewards.scales, "tracking_yaw": 1.0}
