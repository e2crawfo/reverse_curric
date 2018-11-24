import argparse
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator

from curriculum.experiments.starts.maze.maze_gym_brownian_algo import run_task

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    args = parser.parse_args()

    mode = 'local'
    n_parallel = cpu_count() if not args.debug else 1

    exp_prefix = 'start-brownian-maze11-run1'

    vg = VariantGenerator()
    vg.add('maze_id', [11])  # default is 0
    vg.add('start_size', [2])  # this is the ultimate start we care about: getting the pendulum upright
    vg.add('start_range',
           lambda maze_id: [4] if maze_id == 0 else [7])  # this will be used also as bound of the state_space
    # vg.add('start_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, 0)])
    vg.add('start_center', lambda maze_id, start_size: [(2, 2)] if maze_id == 0 and start_size == 2
                                                else [(2, 2, 0, 0)] if maze_id == 0 and start_size == 4
                                                else [(0, 0)] if start_size == 2
                                                else [(0, 0, 0, 0)])
    vg.add('ultimate_goal', lambda maze_id: [(0, 4)] if maze_id == 0 else [(2, 4), (0, 0)] if maze_id == 12 else [(4, 4)])
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('terminal_eps', [0.3])
    vg.add('only_feasible', [True])
    vg.add('goal_range',
           lambda maze_id: [4] if maze_id == 0 else [7])  # this will be used also as bound of the state_space
    vg.add('goal_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, 0)])
    # brownian params
    vg.add('seed_with', ['only_goods'])  # good from brown, onPolicy, previousBrown (ie no good)
    vg.add('brownian_variance', [1])
    vg.add('brownian_horizon', [100])

    # goal-algo params
    vg.add('min_reward', [0.1])
    vg.add('max_reward', [0.9])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [False])  # !!!!
    vg.add('persistence', [1])
    vg.add('n_traj', [3])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    vg.add('sampling_res', [1])
    vg.add('with_replacement', [True])
    vg.add('use_trpo_paths', [True])

    # replay buffer
    vg.add('replay_buffer', [True])
    vg.add('coll_eps', [0.3])
    vg.add('num_new_starts', [200])
    vg.add('num_old_starts', [100])

    # sampling params
    vg.add('horizon', lambda maze_id: [200] if maze_id == 0 else [500])
    vg.add('outer_iters', lambda maze_id: [200] if maze_id == 0 else [1000])
    vg.add('inner_iters', [5])  # again we will have to divide/adjust the
    vg.add('pg_batch_size', [20000])

    # policy initialization
    vg.add('output_gain', [0.1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [False])
    vg.add('adaptive_std', [False])
    vg.add('discount', [0.995])
    vg.add('constant_baseline', [False])

    vg.add('seed', [0])

    variants = vg.variants()
    assert len(variants) == 1
    variant = variants[0]

    run_task(variant)

    run_experiment_lite(
        # use_cloudpickle=False,
        stub_method_call=run_task,
        variant=variant,
        mode='local',
        n_parallel=n_parallel,
        snapshot_mode="last",
        seed=variant['seed'],
        exp_prefix=exp_prefix,
    )
