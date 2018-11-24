import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os.path as osp
import random
import numpy as np

import rllab
from rllab.misc import logger
from rllab.envs.normalized_env import normalize

import gym
from gym import Env
from baselines.trpo_mpi.incremental import build_trpo_context, trpo_take_n_steps

from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.logging.logger import ExperimentLogger
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection
from curriculum.envs.start_env import generate_starts
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.point_maze_env import PointMazeEnv

from curriculum.logging.visualization import plot_labeled_samples, plot_labeled_states, save_image
from curriculum.state.evaluator import compute_labels, label_states_from_paths, evaluate_state
from curriculum.envs.maze.maze_evaluate import find_empty_spaces, plot_heatmap, tile_space

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def plot_policy_means(policy, env, sampling_res=2, report=None, center=None, limit=None):  # only for start envs!
    states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
    goal = env.current_goal
    observations = [
        np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal])
        for state in states]

    mean, log_std = policy._evaluate([policy.pd.mean, policy.pd.logstd], observations)
    vars = [np.exp(log_std) * 0.25 for log_std in log_std]
    ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for e in ells:
        ax.add_artist(e)
        e.set_alpha(0.2)
    plt.scatter(*goal, color='r', s=100)
    Q = plt.quiver(states[:, 0], states[:, 1], mean[:, 0], mean[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)
    plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
    vec_img = save_image()
    if report is not None:
        report.add_image(vec_img, 'policy mean')


def evaluate_states(states, env, policy, horizon, n_traj=1, n_processes=-1, full_path=False, key='rewards',
                    as_goals=True, aggregator=(np.sum, np.mean)):
    result = []
    for state in states:
        r = evaluate_state(
            state=state, env=env, policy=policy, horizon=horizon, n_traj=n_traj, full_path=full_path,
            key=key, as_goals=as_goals, aggregator=aggregator)
        result.append(r)

    if full_path:
        return np.array([state[0] for state in result]), [path for state in result for path in state[1]]

    return np.array(result)


def label_states(
        states, env, policy, horizon, as_goals, min_reward, max_reward, key='rewards', n_traj=1,
        n_processes=-1, full_path=False, return_rew=False):

    logger.log("Labelling starts")
    result = evaluate_states(
        states, env, policy, horizon, as_goals=as_goals,
        n_traj=n_traj, n_processes=n_processes, key=key, full_path=full_path
    )
    if full_path:
        mean_rewards, paths = result
    else:
        mean_rewards = result
    logger.log("Evaluated states.")

    mean_rewards = mean_rewards.reshape(-1, 1)
    labels = compute_labels(mean_rewards, min_reward=min_reward, max_reward=max_reward)

    logger.log("Starts labelled")

    if full_path:
        return labels, paths
    elif return_rew:
        return labels, mean_rewards
    return labels


def test_policy(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1, center=None, bounds=None):
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None
    gen_state_size = np.size(old_start_generator)

    max_path_length = 400

    if bounds is not None:
        if np.array(bounds).size == 1:
            bounds = [-1 * bounds * np.ones(gen_state_size), bounds * np.ones(gen_state_size)]
        states, spacing = tile_space(bounds, sampling_res)
    else:
        states, spacing = find_empty_spaces(train_env, sampling_res=sampling_res)

    states = [np.pad(s, (0, gen_state_size - np.size(s)), 'constant') for s in states]

    avg_totRewards = []
    avg_success = []
    avg_time = []
    logger.log("Evaluating {} states in a grid".format(np.shape(states)[0]))
    rewards, paths = evaluate_states(
        states, train_env, policy, max_path_length, as_goals=as_goals, n_traj=n_traj, full_path=True)
    logger.log("States evaluated")

    path_index = 0
    for _ in states:
        state_paths = paths[path_index:path_index + n_traj]
        avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in state_paths]))

        avg_success.append(
            np.mean([int(np.min(path['env_infos']['distance']) <= train_env.terminal_eps)
                     for path in state_paths]
            )
        )

        avg_time.append(np.mean([path['rewards'].shape[0] for path in state_paths]))
        path_index += n_traj

    return avg_totRewards, avg_success, states, spacing, avg_time


def test_and_plot_policy(policy, env, as_goals=True, visualize=True, sampling_res=1, n_traj=1,
                         max_reward=1, itr=0, report=None, center=None, limit=None, bounds=None):

    avg_totRewards, avg_success, states, spacing, avg_time = test_policy(
        policy, env, as_goals, visualize, center=center, sampling_res=sampling_res, n_traj=n_traj, bounds=bounds)

    obj = env
    while not hasattr(obj, '_maze_id') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    maze_id = obj._maze_id if hasattr(obj, '_maze_id') else None
    plot_heatmap(avg_success, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
                 center=center, limit=limit)
    reward_img = save_image()

    plot_heatmap(avg_time, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
                 center=center, limit=limit, adaptive_range=True)
    time_img = save_image()

    mean_rewards = np.mean(avg_totRewards)
    success = np.mean(avg_success)

    with logger.tabular_prefix('Outer_'):
        logger.record_tabular('iter', itr)
        logger.record_tabular('MeanRewards', mean_rewards)
        logger.record_tabular('Success', success)

    if report is not None:
        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
                itr, mean_rewards, success
            )
        )
        report.add_image(
            time_img,
            'policy time\n itr: {} \n'.format(
                itr
            )
        )

    return mean_rewards, success


def rllab_space_2_gym_space(rllab_space):
    if isinstance(rllab_space, rllab.spaces.discrete.Discrete):
        return gym.spaces.discrete.Discrete(rllab_space.n)
    elif isinstance(rllab_space, rllab.spaces.box.Box):
        return gym.spaces.box.Box(rllab_space.low, rllab_space.high, dtype=np.float32)
    else:
        raise Exception("Can't convert rllab space {} to a gym space.".format(rllab_space))


class RLLabWrapperEnv(Env):
    """ A wrapper that allows rllab environments to interface with algorithms built for gym. """

    def __init__(self, env):
        self.wrapped_env = env

    def step(self, action):
        """ obs, reward, done, info """
        return self.wrapped_env.step(action)

    def reset(self, *args, **kwargs):
        return self.wrapped_env.reset(*args, **kwargs)

    @property
    def action_space(self):
        return rllab_space_2_gym_space(self.wrapped_env.action_space)

    @property
    def observation_space(self):
        return rllab_space_2_gym_space(self.wrapped_env.observation_space)

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self):
        pass


trpo_kwargs = dict(
    timesteps_per_batch=1024,
    max_kl=0.001,
    cg_iters=10,
    gamma=0.99,
    lam=1.0,  # advantage estimation
    ent_coef=0.0,
    cg_damping=1e-2,
    vf_stepsize=3e-4,
    vf_iters=3,
    callback=None,
    network='mlp',
    # **network_kwargs
)

# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     hidden_sizes=(64, 64),
#     # Fix the variance since different goals will require different variances, making this parameter hard to learn.
#     learn_std=v['learn_std'],
#     adaptive_std=v['adaptive_std'],
#     std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
#     output_gain=v['output_gain'],
#     init_std=v['policy_init_std'],
# )

def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = v.get('sampling_res', 2)

    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
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

    gym_env = RLLabWrapperEnv(env)

    context = build_trpo_context(env=gym_env, **trpo_kwargs)
    policy = context['pi']
    outer_iter = 0
    all_starts = StateCollection(distance_threshold=v['coll_eps'])

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

            # algo = TRPO(
            #     env=env,
            #     policy=policy,
            #     batch_size=v['pg_batch_size'],
            #     max_path_length=v['horizon'],
            #     n_itr=v['inner_iters'],
            #     step_size=0.01,
            #     discount=v['discount'],
            #     plot=False,
            # )
            # trpo_paths = algo.train()

            trpo_paths = trpo_take_n_steps(v['innter_iters'], context)

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
        logger.dump_tabular(with_prefix=False)

        all_starts.append(starts)

        new_seed_starts = [start for start, pl in zip(starts, path_labels) if pl[0] and pl[1]]

        if v['seed_with'] == 'only_goods':
            if len(new_seed_starts) > 0:
                logger.log("Only goods A")
                seed_starts = new_seed_starts

            elif np.sum(1-path_labels[:, 0]) > np.sum(1-path_labels[:, 1]):  # if more low reward than high reward
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
