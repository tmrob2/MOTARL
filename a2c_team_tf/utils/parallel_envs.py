# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from

from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_tasks, max_steps, seed=None):
    while True:
        cmd, data, dfa, task_step_count = conn.recv()
        dfa: CrossProductDFA
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(data)
            # Compute the DFA progress
            dfa.next(env)
            # Compute the task rewards from the xDFA
            task_rewards = dfa.rewards(one_off_reward)
            agent_reward = 0
            task_step_count = [task_step_count[i] + 1.0
                                    if d.progress_flag <= 1 else task_step_count[i]
                                    for (i, d) in enumerate(dfa.dfas)]
            # print([d.progress_flag for d in dfa.dfas])
            # Calculate the rewards for the agent based on the DFA progress
            # for d in dfa.dfas:
            #     if d.progress_flag == DFA.Progress.JUST_FINISHED:
            #         agent_reward = 1.0 - 0.9 * env.step_count / env.max_steps
            for (i, d) in enumerate(dfa.dfas):
                #    # Condition to force one off reward for the completion of a task
                if d.progress_flag == DFA.Progress.JUST_FINISHED:
                    # break up the task rewards into sequence rewards. Reduces the sparsity of the reward
                    # so that the agent gets rewarded per task completion instead of on the completion of all tasks
                    agent_reward += 1.0 / num_tasks * (1.0 - 0.9 * task_step_count[i] / max_steps[i])
                    x = task_step_count
                    [x[j] - x[i]
                     if j != i and dfa.dfas[j].progress_flag <= DFA.Progress.JUST_FINISHED
                     else x[j]
                     for j in range(num_tasks)]
                    task_step_count = x
            # if dfa.done():
            #     agent_reward = 1.0 - 0.9 * env.step_count / env.max_steps
            if dfa.done() or env.step_count >= env.max_steps:
                # include a DFA reset
                done = True
                if seed:
                    env.seed(seed)
                dfa.reset()
                obs = env.reset()
                task_step_count = [0] * num_tasks
            else:
                done = False
            reward_ = np.array([max(0.0, agent_reward)] + task_rewards)
            obs_ = np.append(obs, np.array(dfa.progress))
            conn.send((obs_, reward_, done, dfa, task_step_count))
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

    def __init__(self, envs: List[gym.Env], dfas: List[CrossProductDFA], one_off_reward, num_tasks, max_steps:List[float]=[], seed=None):
        self.envs = envs
        self.seed = seed
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.dfas: List[CrossProductDFA] = dfas  # A copy of the cross product DFA for each proc
        self.one_off_reward = one_off_reward  # One off reward given on task completion
        self.num_tasks = num_tasks  # The number of tasks to be allocated to the agent
        if max_steps:
            self.max_steps = max_steps
        else:
            self.max_steps = [envs[0].max_steps/num_tasks]
        self.task_step_counts = [[0] * self.num_tasks for _ in range(len(envs))]

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, one_off_reward, num_tasks, self.max_steps, seed))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        """Multiprocessing environment reset method"""
        for local, dfa in zip(self.locals, self.dfas[1:]):
            local.send(('reset', None, dfa, None))
        self.dfas[0].reset()
        if self.seed:
            self.envs[0].seed(self.seed)
        results = list(zip(*[(self.envs[0].reset(), self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[1])
        self.task_step_counts = [[0] * self.num_tasks for _ in range(len(self.envs))]
        reset_obs = [np.append(result, self.dfas[i].progress) for i, result in enumerate(results[0])]
        return np.array(reset_obs, dtype=np.float32)

    def step(self, actions):
        """Multiprocessing environment step method, also computes the cross product DFA progress"""
        for local, action, dfa, task_step_count in zip(self.locals, actions[1:], self.dfas[1:], self.task_step_counts[1:]):
            local.send(("step", action, dfa, task_step_count))
        obs, reward, done, _ = self.envs[0].step(actions[0])
        self.dfas[0].next(self.envs[0])
        # Compute the task rewards from the xDFA
        task_rewards = self.dfas[0].rewards(self.one_off_reward)
        agent_reward = 0
        x = self.task_step_counts[0]
        self.task_step_counts[0] = [x[i] + 1.0
                                if d.progress_flag <= DFA.Progress.JUST_FINISHED else x[i]
                                for (i, d) in enumerate(self.dfas[0].dfas)]
        # Calculate the rewards for the agent based on the DFA progress
        for (i, d) in enumerate(self.dfas[0].dfas):
        # Condition to force one off reward for the completion of a task
            if d.progress_flag == DFA.Progress.JUST_FINISHED:
                # break up the task rewards into sequence rewards. Reduces the sparsity of the reward
                # so that the agent gets rewarded per task completion instead of on the completion of all tasks
                agent_reward += 1.0 / self.num_tasks * (1.0 - 0.9 * self.task_step_counts[0][i] / self.max_steps[i])
                # subtract the task steps from other tasks
                x = self.task_step_counts[0]
                self.task_step_counts[0] = [x[j] - x[i]
                                            if j != i and self.dfas[0].dfas[j].progress_flag <= DFA.Progress.JUST_FINISHED
                                            else x[j]
                                            for j in range(self.num_tasks)]

        if self.dfas[0].done() or self.envs[0].step_count >= self.envs[0].max_steps:
            # include a DFA reset
            done = True
            if self.seed:
                self.envs[0].seed(self.seed)
            self.dfas[0].reset()
            obs = self.envs[0].reset()
            self.task_step_counts[0] = [0] * self.num_tasks
        else:
            done = False
        # Concatenate the agent reward and tasks rewards
        reward_ = np.array([max(0, agent_reward)] + task_rewards)
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.append(obs, np.array(self.dfas[0].progress))
        results = list(zip(*[(obs_, reward_, done, self.dfas[0], self.task_step_counts[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[3])
        self.task_step_counts = list(results[4])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.bool)

    def render(self):
        raise NotImplementedError