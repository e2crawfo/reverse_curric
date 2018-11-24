import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from curriculum.logging.visualization import plot_labeled_samples, save_image
from curriculum.state.evaluator import compute_labels
from curriculum.envs.maze.maze_evaluate import find_empty_spaces, plot_heatmap, tile_space
from curriculum.envs.base import FixedStateGenerator


def traj_segment_generator(pi, env, stochastic, horizon, n_timesteps=None, n_traj=None, animate=False):
    assert (n_timesteps is None) != (n_traj is None)
    if n_timesteps is None:
        n_timesteps = horizon * n_traj

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    if animate:
        env.render()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(n_timesteps)])
    rews = np.zeros(n_timesteps, 'float32')
    vpreds = np.zeros(n_timesteps, 'float32')
    news = np.zeros(n_timesteps, 'int32')
    acs = np.array([ac for _ in range(n_timesteps)])
    prevacs = acs.copy()
    infos = None

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if n_timesteps is not None and t > 0 and t % n_timesteps == 0:

            # Note: ep_rets and ep_lens do not get updated here before yielding
            # (because we never actually finished the episode, and so we don't know what
            # values they take on). Just something be aware of.

            paths = {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                     "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                     "ep_rets": ep_rets, "ep_lens": ep_lens}
            paths["info"] = {k: np.array(v) for k, v in infos.items()}
            yield paths

            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % n_timesteps
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)

        if animate:
            env.render()
            time.sleep(0.05)

        if infos is None:
            infos = {k: [None for _ in range(n_timesteps)] for k in info.keys()}

        for k, v in info.items():
            infos[k][i] = info[k]

        rews[i] = rew
        cur_ep_ret += rew
        cur_ep_len += 1

        if horizon and cur_ep_len >= horizon:
            new = True

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def evaluate_state(state, env, policy, horizon, n_traj=1, full_path=False, key='rewards'):
    env.update_start_generator(FixedStateGenerator(state))

    seg_gen = traj_segment_generator(policy, env, True, horizon, n_traj=n_traj)
    paths = split_path(list(seg_gen))

    agg_values = []
    for path in paths:
        values = path[key] if key in path else path["info"][key]
        agg_values.append(sum(values))

    return np.mean(agg_values), paths


def split_path(path):
    offset = 0
    paths = []
    for ep_ret, ep_len in zip(path["ep_rets"], path["ep_lens"]):
        new_path = dict()

        for key, value in path.items():
            if key not in "ep_lens ep_rets info".split():
                new_path[key] = value[offset:offset+ep_len]

        new_path["info"] = dict()
        for key, value in path["info"].items():
            new_path["info"][key] = value[offset:offset+ep_len]

        new_path["ep_lens"] = [ep_len]
        new_path["ep_rets"] = [ep_ret]

        paths.append(new_path)
        offset += ep_len

    return paths


def label_states_from_paths(paths, min_reward, max_reward, key='rew', min_traj=1, env=None):
    state_dict = defaultdict(list)

    for path in paths:
        if not len(path['ob']):
            print("degenerate path")
            print(path)
            continue

        print(path['ob'][0])
        state = env.transform_to_start_space(path['ob'][0])
        values = path[key] if key in path else path["info"][key]
        value = sum(values)

        state_dict[tuple(state)].append(value)

    states = []
    mean_rewards = []
    for state, rewards in state_dict.items():
        if len(rewards) >= min_traj:
            states.append(list(state))
            mean_rewards.append(np.mean(rewards))

    mean_rewards = np.array(mean_rewards).reshape(-1, 1)
    labels = compute_labels(mean_rewards, min_reward=min_reward, max_reward=max_reward)
    states = np.array(states)
    return states, labels, mean_rewards, state_dict


def plot_states(states, summary_string, report, itr, **kwargs):
    states = np.array(states)
    if states.size == 0:
        states = np.zeros((1, 2))
    img = plot_labeled_samples(
        states, np.zeros(len(states), dtype='uint8'), markers={0: 'o'}, text_labels={0: "all"}, **kwargs)
    report.add_image(img, 'itr: {}\n{}'.format(itr, summary_string), width=500)


def plot_policy_means(policy, env, sampling_res=2, report=None, center=None, limit=None):
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


def evaluate_states(states, env, policy, horizon, n_traj=1, n_processes=-1, key='rewards'):
    result = []
    paths = []
    for state in states:
        r, p = evaluate_state(
            state=state, env=env, policy=policy, horizon=horizon, n_traj=n_traj, key=key)
        result.append(r)
        paths.append(p)
    return np.array(result), p


def label_states(
        states, env, policy, horizon, as_goals, min_reward, max_reward, key='rewards', n_traj=1,
        n_processes=-1, return_rew=False):

    print("Labelling starts")

    result, _ = evaluate_states(states, env, policy, horizon, n_traj=n_traj, n_processes=n_processes, key=key)

    print("Evaluated states.")

    labels = compute_labels(result, min_reward=min_reward, max_reward=max_reward)

    print("Starts labelled")

    return labels


def test_policy(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1, center=None, bounds=None):
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None
    gen_state_size = np.size(old_start_generator.state)

    max_path_length = 400

    if bounds is not None:
        if np.array(bounds).size == 1:
            bounds = [-1 * bounds * np.ones(gen_state_size), bounds * np.ones(gen_state_size)]
        states, spacing = tile_space(bounds, sampling_res)
    else:
        states, spacing = find_empty_spaces(train_env, sampling_res=sampling_res)

    states = [np.pad(s, (0, gen_state_size - np.size(s)), 'constant') for s in states]

    print("Evaluating {} states in a grid".format(np.shape(states)[0]))

    rewards, paths = evaluate_states(states, train_env, policy, max_path_length, n_traj=n_traj)

    print("States evaluated")

    avg_totRewards = []
    avg_success = []
    avg_time = []

    for i, p in enumerate(paths):
        success = [int(np.min(path['info']['distance']) <= train_env.terminal_eps) for path in p]
        avg_success.append(np.mean(success))
        avg_time.append(np.mean(path['rewards'].shape[0] for path in p))

    return avg_totRewards, avg_success, avg_time, states, spacing


def test_and_plot_policy(policy, env, as_goals=True, visualize=True, sampling_res=1, n_traj=1,
                         max_reward=1, itr=0, report=None, center=None, limit=None, bounds=None):

    avg_totRewards, avg_success, avg_time, states, spacing = test_policy(
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
