# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from

from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_tasks, num_agents, max_steps, seed=None):
    while True:
        cmd, data, dfa = conn.recv() # removed task step count
        dfa: List[CrossProductDFA]
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(data)
            # Compute the DFA progress
            [d.next(env) for d in dfa]
            # Compute the task rewards from the xDFA
            task_rewards = [d.rewards(one_off_reward) for d in dfa]
            agent_reward = [0] * num_agents
            # task_step_count = [task_step_count[i + num_tasks * j] + 1.0
            #                         if d.progress_flag <= 1 else task_step_count[i]
            #                         for (j, d_) in enumerate(dfa) for (i, d) in enumerate(d_.dfas)]
            # print([d.progress_flag for d in dfa.dfas])
            # Calculate the rewards for the agent based on the DFA progress
            # for d in dfa.dfas:
            #     if d.progress_flag == DFA.Progress.JUST_FINISHED:
            #         agent_reward = 1.0 - 0.9 * env.step_count / env.max_steps
            for (i, d_) in enumerate(dfa):
                for j, d in enumerate(d_.dfas):
                    #    # Condition to force one off reward for the completion of a task
                    if d.progress_flag == DFA.Progress.JUST_FINISHED:
                        # break up the task rewards into sequence rewards. Reduces the sparsity of the reward
                        # so that the agent gets rewarded per task completion instead of on the completion of all tasks
                        agent_reward[i] -= 1.0 # 1.0 / num_tasks * (1.0 - 0.9 * task_step_count[i + num_tasks * j] / max_steps[i + num_tasks * j])
                        # x = task_step_count
                        # [x[l + num_tasks * k] - x[j + num_tasks * k]
                        #  if l != i and dfa[k].dfas[l].progress_flag <= DFA.Progress.JUST_FINISHED
                        #  else x[l + num_tasks * k]
                        #  for l in range(num_tasks) for k in range(num_agents)]
                        # task_step_count = x
            # if dfa.done():
            #     agent_reward = 1.0 - 0.9 * env.step_count / env.max_steps
            if all(d.done() for d in dfa) or env.step_count >= env.max_steps:
                # include a DFA reset
                done = True
                if seed:
                    env.seed(seed)
                [d.reset() for d in dfa]
                obs = env.reset()
                # task_step_count = [0] * num_tasks * num_agents
            else:
                done = False
            reward_ = np.array([np.array([max(0.0, agent_reward[i])] + task_rewards[i]) for i in range(num_agents)])
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
            num_tasks,
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
        self.num_tasks = num_tasks  # The number of tasks to be allocated to the agent
        # if max_steps:
        #     self.max_steps = max_steps
        # else:
        #     self.max_steps = [envs[0].max_steps/num_tasks for _ in num_tasks * num_agents]
        self.task_step_counts = [[0] * self.num_tasks * self.num_agents for _ in range(len(envs))]

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, one_off_reward, num_tasks, num_agents, self.max_steps, seed))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        """Multiprocessing environment reset method"""
        for local, dfa in zip(self.locals, self.dfas[1:]):
            local.send(('reset', None, dfa, None))
        [d.reset() for d in self.dfas[0]]
        if self.seed:
            self.envs[0].seed(self.seed)
        results = list(zip(*[(self.envs[0].reset(), self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[1])
        self.task_step_counts = [[0] * self.num_tasks * self.num_agents for _ in range(len(self.envs))]
        reset_obs = [np.array(
            [np.append(result[k], self.dfas[i][k].progress) for k in range(self.num_agents)]
        ) for i, result in enumerate(results[0])]
        return np.array(reset_obs, dtype=np.float32)

    def step(self, actions):
        """Multiprocessing environment step method, also computes the cross product DFA progress"""
        for local, action, dfa, task_step_count in zip(self.locals, actions[1:], self.dfas[1:], self.task_step_counts[1:]):
            local.send(("step", action, dfa, task_step_count))
        obs, reward, done, _ = self.envs[0].step(actions[0])
        [d.next(self.envs[0]) for d in self.dfas[0]]
        # Compute the task rewards from the xDFA
        task_rewards = []
        agent_rewards = [0] * self.num_agents
        for d_ in self.dfas[0]:
            task_rewards.append(d_.rewards(self.one_off_reward))
        x = self.task_step_counts[0]
        self.task_step_counts[0] = [x[i + self.num_tasks * j] + 1.0
                                if d.progress_flag <= DFA.Progress.JUST_FINISHED else x[i]
                                for (j, d_) in enumerate(self.dfas[0]) for (i, d) in enumerate(d_.dfas)]
        # Calculate the rewards for the agent based on the DFA progress
        for i, d_ in enumerate(self.dfas[0]):
            for j, d in enumerate(d_.dfas):
            # Condition to force one off reward for the completion of a task
                if d.progress_flag == DFA.Progress.JUST_FINISHED:
                    # break up the task rewards into sequence rewards. Reduces the sparsity of the reward
                    # so that the agent gets rewarded per task completion instead of on the completion of all tasks
                    agent_rewards[i] += 1.0 / self.num_tasks * (1.0 - 0.9 * self.task_step_counts[0][i + self.num_tasks * j] / self.max_steps[i + self.num_tasks * j])
                    # subtract the task steps from other tasks
                    x = self.task_step_counts[0]
                    self.task_step_counts[0] = [x[j_ + self.num_tasks * k] - x[j + self.num_tasks * k]
                                                if j_ != i and self.dfas[0][k].dfas[j_].progress_flag <= DFA.Progress.JUST_FINISHED
                                                else x[j_ + self.num_tasks * k]
                                                for j_ in range(self.num_tasks) for k in range(self.num_agents)]

        if all(d.done() for d in self.dfas[0]) or self.envs[0].step_count >= self.envs[0].max_steps:
            # include a DFA reset
            done = True
            if self.seed:
                self.envs[0].seed(self.seed)
            [d.reset() for d in self.dfas[0]]
            obs = self.envs[0].reset()
            self.task_step_counts[0] = [0] * self.num_tasks * self.num_agents
        else:
            done = False
        # Concatenate the agent reward and tasks rewards
        reward_ = np.array([np.array([max(0, agent_rewards[i])] + task_rewards[i]) for i in range(self.num_agents)])
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.array([np.append(obs[k], self.dfas[0][k].progress) for k in range(self.num_agents)])
        results = list(zip(*[(obs_, reward_, done, self.dfas[0], self.task_step_counts[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[3])
        self.task_step_counts = list(results[4])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.int32)

    def render(self):
        raise NotImplementedError