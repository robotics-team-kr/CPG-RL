"""Mini Cheetah adaptation of CPG-RL (2022), Sections III and Table I.

Physics/CPG: 1 kHz; policy: 100 Hz. Default: uncoupled oscillators, full feedback,
no observation noise or dynamics randomization. Mini Cheetah geometry, PD gains,
command ranges and periodic phase encoding differ from the paper's Unitree A1.
"""

from legged_gym.envs.base.allgaits import GAIT_CONTACT_TIMES
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

LEGS = ("fr", "fl", "hr", "hl")
FEET = ("front_right", "front_left", "hind_right", "hind_left")


class MiniCheetahCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12
        observation = "full"  # full or medium; both actor and critic use the same feedback.

    class asset(LeggedRobotCfg.asset):
        file = "mini_cheetah/mini_cheetah.xml"
        root_body = mass_body = "body"
        joint_names = [
            name
            for leg in LEGS
            for name in (f"torso_to_abduct_{leg}_j", f"abduct_{leg}_to_thigh_{leg}_j", f"thigh_{leg}_to_knee_{leg}_j")
        ]
        feet = [f"lower_leg_{leg}" for leg in FEET]
        foot_sites = [f"foot_{leg}" for leg in FEET]
        foot_geoms = [f"lower_leg_{leg}_foot_collision" for leg in FEET]
        foot_solimp = [0.9, 0.95, 0.001, 0.5, 2]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0, 0, 0.2975]
        # Updated from the model-specific IK in the environment constructor.
        default_joint_angles = {".*": 0}

    class control(LeggedRobotCfg.control):
        mode = "cpg"
        stiffness = {".*": 35.0}
        damping = {".*": 1.0}
        effort_limits = {".*": 25.0}
        armature = {".*": 0.01}
        action_scale = 0.25
        clip_actions = 1.0

    class cpg:
        amplitude_range = [1.0, 2.0]
        frequency_range = [0.0, 4.5]  # Hz
        direction_rate_range = [-1.5, 1.5]  # Hz
        convergence = 150.0
        coupling_strength = 0.0  # Paper's learned coordination; >0 enables the trot coupling ablation.
        link_lengths = [0.062, 0.209, 0.18]
        step_length = 0.08
        height = 0.28
        clearance = 0.05
        penetration = 0.005
        randomize_geometry = False
        height_range = [0.26, 0.30]
        clearance_range = [0.03, 0.07]

    class commands(LeggedRobotCfg.commands):
        ranges = [[-0.5, 0.8], [-0.3, 0.3], [-0.6, 0.6]]
        limit_ranges = ranges
        resampling_time_range = [5.0, 5.0]
        heading_command = False
        standing_fraction = 0.1
        curriculum = False

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_com = False
        randomize_encoder_bias = False
        push_robots = False

    class termination(LeggedRobotCfg.termination):
        illegal_contact_threshold = 1.0
        max_tilt = 1.05

    class rewards(LeggedRobotCfg.rewards):
        # Section III-C, including the paper's printed work proxy |tau dot delta(qdot)|.
        # Scales are multiplied by policy dt exactly once in LeggedRobot.compute_reward.
        scales = {
            "tracking_x": 0.75,
            "tracking_y": 0.75,
            "tracking_yaw": 0.5,
            "vertical_velocity": -2.0,
            "body_ang_vel": -0.05,
            "work": -0.001,
        }
        pose_std_standing = {".*": 0.3}
        pose_std_walking = {".*": 0.3}
        pose_std_running = {".*": 0.3}


class MiniCheetahJointCfg(MiniCheetahCfg):
    class control(MiniCheetahCfg.control):
        mode = "joint"


class MiniCheetahPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "mini_cheetah_cpg"
        max_iterations = 3000


class MiniCheetahJointPPO(MiniCheetahPPO):
    class runner(MiniCheetahPPO.runner):
        experiment_name = "mini_cheetah_joint"


class MiniCheetahAllGaitsCfg(MiniCheetahCfg):
    """AllGaits-inspired forward locomotion; Mini Cheetah, not the paper's Go1.

    Uses the paper's coupling, 8-D action, implicit gait feedback, transition
    interval and reward structure. Keeps this project's PD/dynamics settings
    and narrower speed/geometry ranges; this is not a full reproduction.
    """

    class env(MiniCheetahCfg.env):
        num_actions = 8

    class cpg(MiniCheetahCfg.cpg):
        coupling_strength = 10.0  # W_ij for i != j; W_ii = 0.
        gait_contact_times = GAIT_CONTACT_TIMES
        frequency_range = [0.0, 8.0]
        randomize_geometry = True
        foot_offset = 0.0
        foot_offset_range = [-0.02, 0.02]
        penetration_range = [0.0, 0.01]

    class gait:
        # This is an external command, like velocity; it is not a policy input.
        name = "random"
        resampling_time = 3.0

    class commands(MiniCheetahCfg.commands):
        ranges = [[0.2, 0.8], [0.0, 0.0], [0.0, 0.0]]
        limit_ranges = ranges
        standing_fraction = 0.0

    class rewards(MiniCheetahCfg.rewards):
        # AllGaits Table I; the base class applies dt once.
        scales = {"tracking_x": 3.0, "lateral_vertical_velocity": -2.0, "all_body_ang_vel": -0.1, "power": -0.001}


class MiniCheetahAllGaitsPPO(MiniCheetahPPO):
    baseline_profile = "legacy"

    class runner(MiniCheetahPPO.runner):
        experiment_name = "mini_cheetah_allgaits"

    class evaluation(MiniCheetahPPO.evaluation):
        commands = {
            "forward_slow": ([0.2, 0, 0], [0.2, 0, 0]),
            "forward_medium": ([0.2, 0, 0], [0.5, 0, 0]),
            "forward_fast": ([0.2, 0, 0], [0.8, 0, 0]),
        }
