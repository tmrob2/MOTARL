# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from

from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_tasks):
    while True:
        cmd, data, dfa = conn.recv()
        dfa: CrossProductDFA
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(data)
            # Compute the DFA progress
            dfa.next(env)
            # Compute the task rewards from the xDFA
            task_rewards = dfa.rewards(one_off_reward)
            agent_reward = 0
            # Calculate the rewards for the agent based on the DFA progress
            for d in dfa.dfas:
                if d.progress_flag == DFA.Progress.JUST_FINISHED:
                    agent_reward += 1 / num_tasks * (1 - 0.9 * env.step_count / env.max_steps)
            if dfa.done() or env.step_count >= env.max_steps:
                # include a DFA reset
                done = True
                dfa.reset()
                obs = env.reset()
            else:
                done = False
            reward_ = np.array([max(0.0, agent_reward)] + task_rewards)
            obs_ = np.append(obs, np.array(dfa.progress))
            conn.send((obs_, reward_, done, dfa))
        elif cmd == "reset":  # Worker reset command from pipe
            # Reset the environment attached to the worker
            obs = env.reset()
            # include a DFA reset
            dfa.reset()
            conn.send((obs, dfa))
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes"""

    def __init__(self, envs: List[gym.Env], dfas: List[CrossProductDFA], one_off_reward, num_tasks):
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.env_steps = [0] * len(envs)
        self.dfas: List[CrossProductDFA] = dfas  # A copy of the cross product DFA for each proc
        self.one_off_reward = one_off_reward  # One off reward given on task completion
        self.num_tasks = num_tasks  # The number of tasks to be allocated to the agent

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, one_off_reward, num_tasks))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        """Multiprocessing environment reset method"""
        for local, dfa in zip(self.locals, self.dfas[1:]):
            local.send(('reset', None, dfa))
        self.dfas[0].reset()
        results = list(zip(*[(self.envs[0].reset(), self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[1])
        reset_obs = [np.append(result, self.dfas[i].progress) for i, result in enumerate(results[0])]
        return np.array(reset_obs, dtype=np.float32)

    def step(self, actions):
        """Multiprocessing environment step method, also computes the cross product DFA progress"""
        for local, action, dfa in zip(self.locals, actions[1:], self.dfas[1:]):
            local.send(("step", action, dfa))
        obs, reward, done, _ = self.envs[0].step(actions[0])
        self.dfas[0].next(self.envs[0])
        # Compute the task rewards from the xDFA
        task_rewards = self.dfas[0].rewards(self.one_off_reward)
        agent_reward = 0
        # Calculate the rewards for the agent based on the DFA progress
        for d in self.dfas[0].dfas:
            # Condition to force one off reward for the completion of a task
            if d.progress_flag == DFA.Progress.JUST_FINISHED:
                # break up the task rewards into sequence rewards. Reduces the sparsity of the reward
                # so that the agent gets rewarded per task completion instead of on the completion of all tasks
                agent_reward += 1 / self.num_tasks * (1 - 0.9 * self.envs[0].step_count / self.envs[0].max_steps)
        if self.dfas[0].done() or self.envs[0].step_count >= self.envs[0].max_steps:
            # include a DFA reset
            done = True
            self.dfas[0].reset()
            obs = self.envs[0].reset()
        else:
            done = False
        # Concatenate the agent reward and tasks rewards
        reward_ = np.array([max(0, agent_reward)] + task_rewards)
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.append(obs, np.array(self.dfas[0].progress))
        results = list(zip(*[(obs_, reward_, done, self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[3])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.bool)

    def render(self):
        raise NotImplementedError