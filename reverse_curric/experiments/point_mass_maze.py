from multiprocessing import cpu_count
from dps.utils import Config
from dps.hyper import run_experiment

from reverse_curric.base import BrownianMotionUpdater, BrownianMotion_RenderHook, build_env


debug = False
n_parallel = 1 if debug else cpu_count()

config = Config(
    seed=0,
    get_updater=BrownianMotionUpdater,
    build_env=build_env,
    stopping_criteria="uniform_dist_reward,max",

    # env params
    maze_id=11,
    start_size=2,
    goal_size=2,
    terminal_eps=0.3,
    only_feasible=True,
    distance_metric='L2',
    extend_dist_rew=False,

    # alg params
    seed_with='only_goods',
    brownian_variance=1,
    brownian_horizon=100,

    min_reward=0.1,
    max_reward=0.9,
    distance_threshold=0,
    n_new_starts=200,
    n_old_starts=100,

    horizon=500,
    n_inner_iters=5,

    # policy params
    output_gain=0.1,
    policy_init_std=1,

    # rendering params
    n_traj=3,
    sampling_res=1,
    render_hook=BrownianMotion_RenderHook(),
    render_step=1,

    n_workers=2,
)

# more env params
maze_id, start_size = config.maze_id, config.start_size
start_center = None
if maze_id == 0 and start_size == 2:
    start_center = (2, 2)
elif maze_id == 0 and start_size == 4:
    start_center = (2, 2, 0, 0)
elif start_size == 2:
    start_center = (0, 0)
else:
    start_center = (0, 0, 0, 0)

config.start_center = start_center
config.start_range = 7
config.ultimate_goal = (4, 4)
config.goal_range = 7
config.goal_center = (0, 0)


config["TRPO"] = Config(
    network="mlp",
    network_kwargs=None,
    timesteps_per_batch=1024,  # original alg used 20000,
    max_kl=0.001,  # original al used 0.01
    cg_iters=10,
    gamma=0.995,
    lam=1.0,
    ent_coef=0.0,
    cg_damping=1e-2,
    vf_stepsize=3e-4,
    vf_iters=3,
    callback=None
)

run_experiment("point_mass_brownian", config, "Point mass")
