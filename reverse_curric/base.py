import numpy as np
import gym
import rllab

from dps import cfg
from dps.updater import Updater
from dps.utils.tf import RenderHook
from dps.utils import Param
from dps.utils.html_report import HTMLReport

from gym import Env as GymEnv

from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from curriculum.state.utils import StateCollection
from curriculum.envs.start_env import generate_starts
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.maze.point_maze_env import PointMazeEnv
from curriculum.logging.visualization import plot_labeled_states

from rllab.envs.normalized_env import normalize

from reverse_curric import trpo
from reverse_curric.evaluate import (
    label_states_from_paths, plot_states, plot_policy_means, label_states, test_and_plot_policy)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def build_env():
    inner_env = normalize(PointMazeEnv(maze_id=cfg.maze_id))

    fixed_goal_generator = FixedStateGenerator(state=cfg.ultimate_goal)
    uniform_start_generator = UniformStateGenerator(
        state_size=cfg.start_size, bounds=cfg.start_range, center=cfg.start_center)

    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=uniform_start_generator,
        obs2start_transform=lambda x: x[:cfg.start_size],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[:cfg.goal_size],
        terminal_eps=cfg.terminal_eps,
        distance_metric=cfg.distance_metric,
        extend_dist_rew=cfg.extend_dist_rew,
        only_feasible=cfg.only_feasible,
        terminate_env=True,
    )
    return env


def rllab_space_2_gym_space(rllab_space):
    if isinstance(rllab_space, rllab.spaces.discrete.Discrete):
        return gym.spaces.discrete.Discrete(rllab_space.n)
    elif isinstance(rllab_space, rllab.spaces.box.Box):
        return gym.spaces.box.Box(rllab_space.low, rllab_space.high, dtype=np.float32)
    else:
        raise Exception("Can't convert rllab space {} to a gym space.".format(rllab_space))


class RLLabWrapperEnv(GymEnv):
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

    def render(self, mode='human'):
        self.wrapped_env.render(mode=mode)

    def close(self):
        pass

    def seed(self):
        pass


class BrownianMotionUpdater(Updater):
    n_old_starts = Param()
    n_new_starts = Param()
    brownian_horizon = Param()
    brownian_variance = Param()
    ultimate_goal = Param()
    distance_threshold = Param()
    seed_with = Param()

    min_reward = Param()
    max_reward = Param()

    horizon = Param()
    n_inner_iters = Param()

    def trainable_variables(self, for_opt):
        return self.trpo.trainable_variables()

    def _build_graph(self):
        self.gym_env = RLLabWrapperEnv(self.env)

        self.trpo = trpo.TRPO(self.gym_env, self.mpi_context)

        self.all_starts = StateCollection(distance_threshold=self.distance_threshold)
        self.seed_starts = None

    def _update(self, _):
        if self.seed_starts is None:
            self.seed_starts = generate_starts(
                self.env, starts=[self.ultimate_goal], subsample=self.n_new_starts,
                animated=True)

        policy = self.trpo.pi

        print("{} seed starts".format(len(self.seed_starts)))
        old_seed_starts = self.seed_starts

        brownian_starts = generate_starts(
            self.env, starts=self.seed_starts, subsample=self.n_new_starts,
            horizon=self.brownian_horizon, variance=self.brownian_variance,
            animated=True)

        print("{} starts sampled via brownian motion".format(len(brownian_starts)))

        buffer_starts = []
        if self.all_starts.size > 0:
            buffer_starts = self.all_starts.sample(self.n_old_starts)
            print("{} starts sampled from replay buffer".format(len(buffer_starts)))
            sampled_starts = np.vstack([brownian_starts, buffer_starts])
        else:
            sampled_starts = brownian_starts

        print("{} starts total".format(len(sampled_starts)))

        print("Updating the environment start generator")

        self.env.update_start_generator(
            UniformListStateGenerator(list(sampled_starts), persistence=1, with_replacement=True)
        )

        print("Running {} steps of TRPO.".format(self.n_inner_iters))

        trpo_paths = self.trpo.take_n_steps(self.n_inner_iters)

        print("Number of steps for update steps: {} .".format([len(path["ob"]) for path in trpo_paths]))
        print("Total number of steps: {} .".format(sum([len(path["ob"]) for path in trpo_paths])))

        # Note that here we are getting rid of starts for which we do not have at
        # least 2 trajectories from the preceding round of TRPO updates.

        evaluated_starts, path_labels, mean_rewards, state_dict = label_states_from_paths(
            trpo_paths, min_reward=self.min_reward, max_reward=self.max_reward,
            min_traj=2, key='goal_reached', env=self.env)

        print("{} start states had at least 1 sample".format(len(state_dict)))
        print("{} start states had at least 2 samples".format(len(evaluated_starts)))

        self.all_starts.append(sampled_starts)
        # self.all_starts.append(evaluated_starts)

        new_seed_starts = [start for start, pl in zip(evaluated_starts, path_labels) if pl[0] and pl[1]]

        if self.seed_with == 'only_goods':
            if len(new_seed_starts) > 0:
                print("Only goods A")
                seed_starts = new_seed_starts

            elif np.sum(1-path_labels[:, 0]) > np.sum(1-path_labels[:, 1]):  # if more low reward than high reward
                print("Only goods B")
                seed_starts = self.all_starts.sample(300)  # sample them from the replay

            else:
                print("Only goods C")
                # add a ton of noise if all the states I had ended up being high_reward
                seed_starts = generate_starts(
                    self.env, starts=evaluated_starts, horizon=int(self.horizon * 10),
                    subsample=self.n_new_starts, variance=self.brownian_variance * 10)

        elif self.seed_with == 'all_previous':
            seed_starts = evaluated_starts

        elif self.seed_with == 'on_policy':
            seed_starts = generate_starts(
                self.env, policy, starts=evaluated_starts,
                horizon=self.horizon, subsample=self.n_new_starts)

        self.old_seed_starts = old_seed_starts
        self.brownian_starts = brownian_starts
        self.buffer_starts = buffer_starts
        self.sampled_starts = sampled_starts
        self.evaluated_starts = evaluated_starts
        self.path_labels = path_labels
        self.seed_starts = seed_starts

        return {}

    def _evaluate(self, _batch_size, mode):
        return {"uniform_dist_reward": 0}

    def worker_code(self):
        print("In worker_code", flush=True)

        while True:
            op = self.comm.bcast(None, root=0)

            if op == "trpo":
                pass
            elif op == "eval_states":
                pass
            elif op == "set_start_states":
                pass

            pass

    def eval_states(self, states):


render_hook_params = dict(
    sampling_res=2
)


class BrownianMotion_RenderHook(RenderHook):
    # sampling_res = Param()
    # maze_id = Param()
    # goal_center = Param()
    # goal_range = Param()

    # max_reward = Param()
    # min_reward = Param()
    # horizon = Param()
    # n_traj = Param()

    def __init__(self, **kwargs):
        self.report = None
        super().__init__(**kwargs)

    def __call__(self, updater):
        if self.report is None:
            self.report = HTMLReport(updater.exp_dir.path_for('plots', 'report.html'), images_per_row=10)
            # self.report.add_text(format_dict(v))

        report = self.report
        env = updater.env
        policy = updater.trpo.pi

        with report:
            report.new_row()

            plot_kwargs = dict(limit=cfg.goal_range, center=cfg.goal_center, report=report, maze_id=cfg.maze_id, itr=0)

            plot_policy_means(
                policy, env, sampling_res=cfg.sampling_res,
                report=report, limit=cfg.goal_range, center=cfg.goal_center)

            test_and_plot_policy(
                policy, env, as_goals=False, max_reward=cfg.max_reward, sampling_res=cfg.sampling_res,
                n_traj=cfg.n_traj, itr=0, report=report, center=cfg.goal_center, limit=cfg.goal_range)

            plot_states(updater.old_seed_starts, "seed starts", **plot_kwargs)
            plot_states(updater.brownian_starts, "brownian starts", **plot_kwargs)
            plot_states(updater.buffer_starts, "starts sampled from buffer", **plot_kwargs)

            labels = label_states(
                updater.sampled_started, env, policy, cfg.horizon, as_goals=False,
                min_reward=cfg.min_reward, max_reward=cfg.max_reward, n_traj=cfg.n_traj, key='goal_reached')
            plot_labeled_states(
                updater.sampled_starts, labels, summary_string_base='all starts before update', **plot_kwargs)

            after_labels = label_states(
                updater.sampled_starts, env, policy, cfg.horizon, as_goals=False,
                min_reward=cfg.min_reward, max_reward=cfg.max_reward, n_traj=cfg.n_traj, key='goal_reached')
            plot_labeled_states(
                updater.sampled_started, after_labels, summary_string_base='all starts after update', **plot_kwargs)

            plot_labeled_states(
                updater.evaluated_starts, updater.path_labels,
                summary_string_base='labels based on trpo paths', **plot_kwargs)
