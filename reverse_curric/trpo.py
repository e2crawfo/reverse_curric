from collections import deque
from contextlib import contextmanager
import time

import tensorflow as tf
import numpy as np

from baselines.common import explained_variance, zipsame, dataset
import baselines.common.tf_util as U
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy

from dps.utils import Param, Parameterized

from reverse_curric.evaluate import traj_segment_generator, split_path

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


class TRPO(Parameterized):
    network = Param()
    network_kwargs = Param()
    horizon = Param()
    timesteps_per_batch = Param()
    max_kl = Param()
    cg_iters = Param()
    gamma = Param()
    lam = Param()
    ent_coef = Param()
    cg_damping = Param()
    vf_stepsize = Param()
    vf_iters = Param()
    callback = Param()

    def trainable_variables(self):
        return get_trainable_variables("pi")

    def __init__(self, env, mpi_context=None, **kwargs):
        network_kwargs = self.network_kwargs or {}
        policy = build_policy(env, self.network, value_network='copy', **network_kwargs)

        if MPI is not None and mpi_context is not None:
            self.comm = mpi_context.merged_comm
            n_workers = self.comm.Get_size()
            rank = self.comm.Get_rank()
        else:
            self.comm = None
            n_workers = 1
            rank = 0

        cpus_per_worker = 1
        U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
        ))

        np.set_printoptions(precision=3)
        ob_space = env.observation_space
        ac_space = env.action_space

        ob = observation_placeholder(ob_space)
        with tf.variable_scope("pi"):
            pi = policy(observ_placeholder=ob)
        with tf.variable_scope("oldpi"):
            oldpi = policy(observ_placeholder=ob)

        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.ent_coef * meanent

        vferr = tf.reduce_mean(tf.square(pi.vf - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

        dist = meankl

        all_var_list = get_trainable_variables("pi")
        var_list = get_pi_trainable_variables("pi")
        vf_var_list = get_vf_trainable_variables("pi")

        vfadam = MpiAdam(vf_var_list)

        get_flat = U.GetFlat(var_list)
        set_from_flat = U.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz

        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)

        updates = [tf.assign(oldv, newv) for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))]
        assign_old_eq_new = U.function([], [], updates=updates)

        compute_losses = U.function([ob, ac, atarg], losses)
        compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

        U.initialize()

        th_init = get_flat()
        if self.comm is not None:
            print("bcasting theta init")
            self.comm.Bcast(th_init, root=0)
            print("done bcasting theta init")

        set_from_flat(th_init)
        vfadam.sync()
        print("Init param sum", th_init.sum(), flush=True)

        seg_gen = traj_segment_generator(pi, env, True, self.horizon, self.timesteps_per_batch)

        print("done making seg_gen", flush=True)

        _locals = locals()
        del _locals['self']
        self.__dict__.update(_locals)

        print("done trpo init", flush=True)

    def take_n_steps(self, n):
        print("in trpo take_n_steps")

        locals().update(self.__dict__)

        episodes_so_far = 0
        timesteps_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

        @contextmanager
        def timed(msg):
            if self.rank == 0:
                print(colorize(msg, color='magenta'))
                tstart = time.time()
                yield
                print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
            else:
                yield

        def allmean(x):
            assert isinstance(x, np.ndarray)
            if self.comm is not None:
                out = np.empty_like(x)
                self.comm.Allreduce(x, out, op=MPI.SUM)
                out /= self.n_workers
            else:
                out = np.copy(x)

            return out

        segs = []
        for i in range(n):
            if self.callback:
                self.callback(locals(), globals())

            print("********** TRPO Iteration {} ************".format(i))

            with timed("sampling"):
                seg = self.seg_gen.__next__()
                segs.extend(split_path(seg))

            add_vtarg_and_adv(seg, self.gamma, self.lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(self.pi, "ret_rms"):
                self.pi.ret_rms.update(tdlamret)
            if hasattr(self.pi, "ob_rms"):
                self.pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            def fisher_vector_product(p):
                return allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p

            self.assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = self.compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                print("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=self.rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / self.max_kl)
                # print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = self.get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    self.set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(self.compute_losses(*args)))
                    improve = surr - surrbefore
                    print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        print("Got non-finite value of losses -- bad!")
                    elif kl > self.max_kl * 1.5:
                        print("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        print("surrogate didn't improve. shrinking step.")
                    else:
                        print("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    print("couldn't compute a good step")
                    self.set_from_flat(thbefore)

                if self.n_workers > 1 and i % 20 == 0:
                    paramsums = self.comm.allgather((thnew.sum(), self.vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            for (lossname, lossval) in zip(self.loss_names, meanlosses):
                print(lossname, lossval)

            with timed("vf"):
                for _ in range(self.vf_iters):
                    batches = dataset.iterbatches(
                        (seg["ob"], seg["tdlamret"]), include_final_partial_batch=False, batch_size=64)

                    for (mbob, mbret) in batches:
                        g = allmean(self.compute_vflossandgrad(mbob, mbret))
                        self.vfadam.update(g, self.vf_stepsize)

            print("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

            lrlocal = (seg["ep_lens"], seg["ep_rets"])
            if self.comm is not None:
                listoflrpairs = self.comm.allgather(lrlocal)  # list of tuples
            else:
                listoflrpairs = [lrlocal]

            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)

            print("EpLenMean", np.mean(lenbuffer))
            print("EpRewMean", np.mean(rewbuffer))
            print("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)

            print("EpisodesSoFar", episodes_so_far)
            print("TimestepsSoFar", timesteps_so_far)
            print("TimeElapsed", time.time() - tstart)

        return segs


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]


def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]
