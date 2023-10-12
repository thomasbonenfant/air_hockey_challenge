from multiprocessing import Process, Queue, Event
import os
import time
import numpy as np
from optimistic_exploration import get_optimistic_exploration_action
from utils.env_utils import NormalizedBoxEnv, env_producer
import copy

def traj_segment_function(
        env,
        policy,
        max_path_length,
        num_steps,
        discard_incomplete_paths,
        optimistic_exploration=False,
        optimistic_exploration_kwargs={},
        deterministic_pol=False
):
    paths = []
    returns = []
    num_steps_collected = 0
    while num_steps_collected < num_steps:
        max_path_length_this_loop = min(  # Do not go over num_steps
            max_path_length,
            num_steps - num_steps_collected,
        )
        path = rollout(
            env,
            policy,
            max_path_length=max_path_length_this_loop,
            optimistic_exploration=optimistic_exploration,
            deterministic_pol=deterministic_pol,
            optimistic_exploration_kwargs=optimistic_exploration_kwargs
        )
        path_len = len(path['actions'])
        if (
                # incomplete path
                path_len != max_path_length and

                # that did not end in a terminal state
                not path['terminals'][-1] and

                # and we should discard such path
                discard_incomplete_paths
        ):
            break
        num_steps_collected += path_len
        paths.append(path)
        rewards = path['rewards']
        ret = np.sum(rewards)
        returns.append(ret)

    # self._num_paths_total += len(paths)
    # self._num_steps_total += num_steps_collected
    # self._epoch_paths.extend(paths)
    return paths, returns


def f(env,
      policy,
      max_path_length,
      num_steps,
      discard_incomplete_paths,
      optimistic_exploration=False,
      optimistic_exploration_kwargs={},
      deterministic_pol=False):
    return traj_segment_function(env,
                                 policy,
                                 max_path_length,
                                 num_steps,
                                 discard_incomplete_paths,
                                 optimistic_exploration=optimistic_exploration,
                                 optimistic_exploration_kwargs=optimistic_exploration_kwargs,
                                 deterministic_pol=deterministic_pol)


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        optimistic_exploration=False,
        optimistic_exploration_kwargs={},
        deterministic_pol=False
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:

        if not optimistic_exploration:
            a, agent_info = agent.get_action(o, deterministic=deterministic_pol)
        else:
            a, agent_info = get_optimistic_exploration_action(
                o, **optimistic_exploration_kwargs)

        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    '''

    def __init__(self, output, input, event, env, policy, traj_segment_generator, seed, index, parall_params=None,
                 f_params=None):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.env = env
        self.policy = policy
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed
        self.index = index
        self.parall_params = parall_params
        self.f_params = f_params

    def close_env(self):
        self.env.close()

    def run(self):
        # env = self.make_env(self.index)

        self.env.reset()
        workerseed = self.seed + 10000 * self.index
        np.random.seed(workerseed)
        self.env.seed(workerseed)
        while True:
            self.event.wait()
            self.event.clear()
            command, args = self.input.get()
            if command == 'collect':
                weights = args["actor_weights"]
                max_path_length = args['max_path_length']
                num_steps = args['num_steps']
                discard_incomplete_paths = args['discard_incomplete_paths']
                optimistic_exploration = args['optimistic_exploration']
                optimistic_exploration_kwargs = args['optimistic_exploration_kwargs']
                deterministic_pol = args['deterministic_pol']
                self.policy.load_state_dict(weights)
                samples = self.traj_segment_generator(env=self.env,
                                                      policy=self.policy,
                                                      max_path_length=max_path_length,
                                                      num_steps=num_steps,
                                                      discard_incomplete_paths=discard_incomplete_paths,
                                                      optimistic_exploration=optimistic_exploration,
                                                      optimistic_exploration_kwargs=optimistic_exploration_kwargs,
                                                      deterministic_pol=deterministic_pol
                                                      )
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                self.env.close()
                break


class ParallelSampler(object):
    def __init__(self,
                 envs,
                 policies,
                 n_workers,
                 episodes_per_worker=1,
                 seed=0,
                 parall_params=None):

        self.n_workers = n_workers

        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()
        # num_steps_per_worker = num_steps // self.n_workers + 1
        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]
        self.seed = seed
        self.envs = envs
        self.policies = policies
        n_episodes_per_process = episodes_per_worker
        self.parall_params = parall_params
        print("%s episodes per worker" % n_episodes_per_process)
        self.f_params = dict(
            max_path_length=None,
            num_steps=None,
            discard_incomplete_paths=None,
            num_episodes=episodes_per_worker,
            optimistic_exploration=None,
            optimistic_exploration_kwargs=None,
            deterministic_pol=None
        )

        self.fun = [f] * self.n_workers
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], envs[i], policies[i],
                               self.fun[i], seed + i, i, self.parall_params, self.f_params) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()

    def collect(self, actor_weights, max_path_length, num_steps, discard_incomplete_paths, optimistic_exploration,
                optimistic_exploration_kwargs, deterministic_pol):
        args = {'actor_weights': actor_weights,
                'max_path_length': max_path_length,
                'num_steps': num_steps,
                'discard_incomplete_paths': discard_incomplete_paths,
                'optimistic_exploration': optimistic_exploration,
                'optimistic_exploration_kwargs': optimistic_exploration_kwargs,
                'deterministic_pol': deterministic_pol}
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', args))

        for e in self.events:
            e.set()

        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.append(samples)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):

        path = []
        returns = []
        for batch in sample_batches:
            path.extend(batch[0])
            returns.extend(batch[1])
        #     rets += batch["ret"]
        #     disc_rets += batch["disc_ret"]
        #     lens += batch["len"]
        return path, returns

    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()
        for w in self.workers:
            w.join()

    def restart(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()

        for w in self.workers:
            w.terminate()
            del w
        self.workers = [
            Worker(self.output_queue, self.input_queues[i], self.events[i], self.envs[i], self.policies[i],
                   self.fun[i], self.seed + i, i, self.parall_params, self.f_params) for i in range(self.n_workers)]
        for w in self.workers:
            w.start()
