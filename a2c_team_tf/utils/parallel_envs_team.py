# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from

from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_agents, seed=None):
    while True:
        cmd, data, dfa = conn.recv() # removed task step count
        dfa: List[CrossProductDFA]
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(data)
            # Compute the DFA progress
            [d.next(env) for d in dfa]
            # Compute the task rewards from the xDFA
            task_rewards = [d.rewards(one_off_reward) for d in dfa]
            agent_reward = [0.0 if d.done() else -1.0 for d in dfa]
            if all(d.done() for d in dfa) or env.step_count >= env.max_steps:
                # include a DFA reset
                done = True
                if seed:
                    env.seed(seed)
                [d.reset() for d in dfa]
                obs = env.reset()
            else:
                done = False
            reward_ = np.array([np.array([agent_reward[i]] + task_rewards[i]) for i in range(num_agents)])
            obs_ = np.array([np.append(obs[k], dfa[k].progress) for k in range(num_agents)])
            conn.send((obs_, reward_, done, dfa))  # removed task step count from return tuple
        elif cmd == "reset":  # Worker reset command from pipe
            # Reset the environment attached to the worker
            if seed:
                env.seed(seed)
            obs = env.reset()
            # include a DFA reset
            [d.reset() for d in dfa]
            conn.send((obs, dfa))
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes"""

    def __init__(
            self,
            envs: List[gym.Env],
            dfas: List[List[CrossProductDFA]],
            one_off_reward,
            num_agents,
            seed=None):  # removed max steps from signature
        self.envs = envs
        self.seed = seed
        self.num_agents = num_agents
        if self.envs:
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
        self.dfas: List[List[CrossProductDFA]] = dfas  # A copy of the cross product DFA for each proc
        self.one_off_reward = one_off_reward  # One off reward given on task completion
        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, one_off_reward, num_agents, seed))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        """Multiprocessing environment reset method"""
        for local, dfa in zip(self.locals, self.dfas[1:]):
            local.send(('reset', None, dfa))
        [d.reset() for d in self.dfas[0]]
        if self.seed:
            self.envs[0].seed(self.seed)
        results = list(zip(*[(self.envs[0].reset(), self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[1])
        reset_obs = [np.array(
            [np.append(result[k], self.dfas[i][k].progress) for k in range(self.num_agents)]
        ) for i, result in enumerate(results[0])]
        return np.array(reset_obs, dtype=np.float32)

    def step(self, actions):
        """Multiprocessing environment step method, also computes the cross product DFA progress"""
        for local, action, dfa in zip(self.locals, actions[1:], self.dfas[1:]):
            local.send(("step", action, dfa))
        obs, reward, done, _ = self.envs[0].step(actions[0])
        [d.next(self.envs[0]) for d in self.dfas[0]]
        # Compute the task rewards from the xDFA
        agent_rewards = [0.0 if d.done() else -1.0 for d in self.dfas[0]]
        task_rewards = [d_.rewards(self.one_off_reward) for d_ in self.dfas[0]]

        if all(d.done() for d in self.dfas[0]) or self.envs[0].step_count >= self.envs[0].max_steps:
            # include a DFA reset
            done = True
            if self.seed:
                self.envs[0].seed(self.seed)
            [d.reset() for d in self.dfas[0]]
            obs = self.envs[0].reset()
        else:
            done = False
        # Concatenate the agent reward and tasks rewards
        reward_ = np.array([np.array([agent_rewards[i]] + task_rewards[i]) for i in range(self.num_agents)])
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.array([np.append(obs[k], self.dfas[0][k].progress) for k in range(self.num_agents)])
        results = list(zip(*[(obs_, reward_, done, self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[3])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.int32)

    def render(self):
        raise NotImplementedError