import matplotlib
matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np

from rllab.misc import logger
from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.logging.logger import ExperimentLogger
from curriculum.logging.visualization import plot_labeled_samples, plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from curriculum.state.evaluator import convert_label, label_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection

from curriculum.envs.start_env import generate_starts
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy, plot_policy_means
from curriculum.envs.maze.point_maze_env import PointMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = v.get('sampling_res', 2)

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=10)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    inner_env = normalize(PointMazeEnv(maze_id=v['maze_id']))

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    uniform_start_generator = UniformStateGenerator(state_size=v['start_size'], bounds=v['start_range'],
                                                    center=v['start_center'])

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[:v['goal_size']],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
        terminate_env=True,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        learn_std=v['learn_std'],
        adaptive_std=v['adaptive_std'],
        std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
        output_gain=v['output_gain'],
        init_std=v['policy_init_std'],
    )

    if v['constant_baseline']:
        from curriculum.state.constant_baseline import ConstantBaseline
        logger.log("Using constant baseline")
        baseline = ConstantBaseline(env_spec=env.spec, value=1.0)
    else:
        logger.log("Using linear baseline")
        baseline = LinearFeatureBaseline(env_spec=env.spec)

    # initialize all logging arrays on itr0
    outer_iter = 0
    all_starts = StateCollection(distance_threshold=v['coll_eps'])

    # seed_starts: from which we will be performing brownian motion exploration
    seed_starts = generate_starts(env, starts=[v['ultimate_goal']], subsample=v['num_new_starts'])

    def plot_states(states, summary_string, report, itr, **kwargs):
        states = np.array(states)
        if states.size == 0:
            states = np.zeros((1, 2))
        img = plot_labeled_samples(
            states, np.zeros(len(states), dtype='uint8'), markers={0: 'o'}, text_labels={0: "all"}, **kwargs)
        report.add_image(img, 'itr: {}\n{}'.format(itr, summary_string), width=500)

    plot_kwargs = dict(limit=v['goal_range'], center=v['goal_center'], report=report, maze_id=v['maze_id'])

    for outer_iter in range(1, v['outer_iters']):
        logger.log('\n\n' + '*' * 40 + " Start outer iter: {} ".format(outer_iter) + '*' * 40 + '\n')

        report.new_row()
        plot_kwargs['itr'] = outer_iter

        logger.log('Generating Heatmap...')
        plot_policy_means(
            policy, env, sampling_res=sampling_res, report=report, limit=v['goal_range'], center=v['goal_center'])
        test_and_plot_policy(
            policy, env, as_goals=False, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
            itr=outer_iter, report=report, center=v['goal_center'], limit=v['goal_range'])

        logger.log("Outer itr # %i" % outer_iter)
        logger.log("Sampling starts")

        plot_states(seed_starts, "seed starts", **plot_kwargs)

        logger.log("{} seed starts".format(len(seed_starts)))

        starts = generate_starts(env, starts=seed_starts, subsample=v['num_new_starts'],
                                 horizon=v['brownian_horizon'], variance=v['brownian_variance'])

        logger.log("{} starts sampled via brownian motion".format(len(starts)))

        plot_states(starts, "brownian starts", **plot_kwargs)

        sampled_from_buffer = []
        if v['replay_buffer'] and outer_iter > 0 and all_starts.size > 0:
            sampled_from_buffer = all_starts.sample(v['num_old_starts'])
            logger.log("{} starts sampled from replay buffer".format(len(sampled_from_buffer)))
            starts = np.vstack([starts, sampled_from_buffer])

        plot_states(sampled_from_buffer, "starts sampled from buffer", **plot_kwargs)

        logger.log("{} starts total".format(len(starts)))

        labels = label_states(
            starts, env, policy, v['horizon'], as_goals=False, min_reward=v['min_reward'], max_reward=v['max_reward'],
            n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(starts, labels, summary_string_base='all starts before update', **plot_kwargs)

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment start generator")
            env.update_start_generator(
                UniformListStateGenerator(
                    starts.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                )
            )

            logger.log("Training the algorithm")
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=v['pg_batch_size'],
                max_path_length=v['horizon'],
                n_itr=v['inner_iters'],
                step_size=0.01,
                discount=v['discount'],
                plot=False,
            )

            trpo_paths = algo.train()

        logger.log("Started with {} start states.".format(len(starts)))
        logger.log("Number of trajectories for update steps: {} .".format([len(paths) for paths in trpo_paths]))
        logger.log("Total number of trajectories: {} .".format(sum([len(paths) for paths in trpo_paths])))

        # Note that here we are getting rid of starts for which we do not have at
        # least 2 trajectories from the preceding round of TRPO updates.
        starts, path_labels, state_dict = label_states_from_paths(
            trpo_paths, min_reward=v['min_reward'], max_reward=v['max_reward'],
            min_traj=2, key='goal_reached', as_goal=False, env=env)

        logger.log("{} start states had at least 1 sample".format(len(state_dict)))
        logger.log("{} start states had at least 2 samples".format(len(starts)))

        plot_labeled_states(starts, path_labels, summary_string_base='labels based on trpo paths', **plot_kwargs)

        after_labels = label_states(
            starts, env, policy, v['horizon'], as_goals=False, min_reward=v['min_reward'], max_reward=v['max_reward'],
            n_traj=v['n_traj'], key='goal_reached')
        plot_labeled_states(starts, after_labels, summary_string_base='all starts after update', **plot_kwargs)

        with logger.tabular_prefix("OnStarts_"):
            paths = [path for paths in trpo_paths for path in paths]
            env.log_diagnostics(paths)

        start_classes, _ = convert_label(path_labels)
        path_labels = np.logical_and(path_labels[:, 0], path_labels[:, 1]).astype(int).reshape((-1, 1))

        logger.dump_tabular(with_prefix=False)

        filtered_raw_starts = [start for start, label in zip(starts, path_labels) if label[0] == 1]
        all_starts.append(starts)
        # all_starts.append(filtered_raw_starts)

        if v['seed_with'] == 'only_goods':
            if len(filtered_raw_starts) > 0:
                logger.log("Only goods A")
                seed_starts = filtered_raw_starts

            elif np.sum(start_classes == 0) > np.sum(start_classes == 1):  # if more low reward than high reward
                logger.log("Only goods B")
                seed_starts = all_starts.sample(300)  # sample them from the replay

            else:
                logger.log("Only goods C")
                # add a ton of noise if all the states I had ended up being high_reward
                seed_starts = generate_starts(
                    env, starts=starts, horizon=int(v['horizon'] * 10),
                    subsample=v['num_new_starts'], variance=v['brownian_variance'] * 10)

        elif v['seed_with'] == 'all_previous':
            seed_starts = starts

        elif v['seed_with'] == 'on_policy':
            seed_starts = generate_starts(env, policy, starts=starts, horizon=v['horizon'], subsample=v['num_new_starts'])

        report.save()
