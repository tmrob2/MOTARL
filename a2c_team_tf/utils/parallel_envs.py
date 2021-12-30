# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from
import multiprocessing.connection
from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_agents, seed=None):
    while True:
        cmd, data, dfa = conn.recv() # removed task step count
        dfa: CrossProductDFA
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(data)
            # Compute the DFA progress
            dfa.next(env)
            # Compute the task rewards from the xDFA
            task_rewards = dfa.rewards(one_off_reward)
            agent_reward = 0.0 if dfa.done() else -1.0
            if dfa.done or env.step_count >= env.max_steps:
                # include a DFA reset
                done = True
                if seed:
                    env.seed(seed)
                dfa.reset()
                obs = env.reset()
            else:
                done = False
            reward_ = np.array([agent_reward] + task_rewards)
            obs_ = np.append(obs, np.array(dfa.progress))
            conn.send((obs_, reward_, done, dfa))  # removed task step count from return tuple
        elif cmd == "reset":  # Worker reset command from pipe
            # Reset the environment attached to the worker
            if seed:
                env.seed(seed)
            obs = env.reset()
            # include a DFA reset
            dfa.reset()
            conn.send((obs, dfa))
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes"""

    def __init__(
            self,
            envs: List[List[gym.Env]],
            dfas: List[List[CrossProductDFA]],
            observation_space: int,
            action_space: int,
            one_off_reward,
            num_agents,
            seed=None):  # removed max steps from signature
        self.envs = envs
        self.seed = seed
        self.num_agents = num_agents
        if self.envs:
            self.observation_space = observation_space
            self.action_space = action_space
        self.dfas: List[List[CrossProductDFA]] = dfas  # A copy of the cross product DFA for each proc
        self.one_off_reward = one_off_reward  # One off reward given on task completion
        self.locals: List[List[multiprocessing.connection.Connection]] = []
        for agent in range(self.num_agents):
            agent_locals = []
            for env in self.envs[agent][1:]:
                local, remote = Pipe()
                agent_locals.append(local)
                p = Process(target=worker, args=(remote, env, one_off_reward, seed))
                p.daemon = True
                p.start()
                remote.close()
            self.locals.append(agent_locals)

    def reset(self) -> List[np.ndarray]:
        """Multiprocessing environment reset method"""
        reset_states = []
        for agent in range(self.num_agents):
            for local, dfa in zip(self.locals[agent], self.dfas[agent][1:]):
                local.send(('reset', None, dfa))
            self.dfas[agent][0].reset()
            if self.seed:
                self.envs[agent][0].seed(self.seed)
            results = list(zip(*[(self.envs[agent][0].reset(), self.dfas[agent][0])] + [local.recv() for local in self.locals[agent]]))
            self.dfas[agent] = list(results[1])
            reset_obs = np.array([np.append(result, self.dfas[agent][i].progress) for i, result in enumerate(results[0])])
            # reset obs is a list of ndarrays which is essentially all of he procs reset states for each agent
            reset_states.append(reset_obs)
        return reset_states

    def step(self, actions: List[int], agent: int):
        """Multiprocessing environment step method, also computes the cross product DFA progress"""
        for local, action, dfa in zip(self.locals[agent], actions[1:], self.dfas[agent][1:]):
            local.send(("step", action, dfa))
        obs, reward, done, _ = self.envs[agent][0].step(actions[0])
        self.dfas[agent][0].next(self.envs[agent][0])
        # Compute the task rewards from the xDFA
        agent_rewards = 0.0 if self.dfas[agent][0].done() else -1.0
        task_rewards = self.dfas[agent][0].rewards(self.one_off_reward)

        if all(d.done() for d in self.dfas[agent]) or self.envs[agent][0].step_count >= self.envs[agent][0].max_steps:
            # include a DFA reset
            done = True
            if self.seed:
                self.envs[agent][0].seed(self.seed)
            [d.reset() for d in self.dfas[agent]]
            obs = self.envs[agent][0].reset()
        else:
            done = False
        # Concatenate the agent reward and tasks rewards
        reward_ = np.array([agent_rewards] + task_rewards)
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.append(obs, np.array(self.dfas[agent][0].progress))
        results = list(zip(*[(obs_, reward_, done, self.dfas[agent][0])] + [local.recv() for local in self.locals[agent]]))
        self.dfas = list(results[3])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.int32)

    def render(self):
        raise NotImplementedError