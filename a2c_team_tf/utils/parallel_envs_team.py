# Implements multiprocessing of environment step and DFA progress
# The purpose of this is to generate significantly more data for the NN model to learn from

from multiprocessing import Process, Pipe
from typing import List
import gym
import numpy as np
from a2c_team_tf.utils.dfa import CrossProductDFA, DFA


def worker(conn, env: gym.Env, one_off_reward, num_agents, n_coeff=1.0, n_coeff2=1.0,
           seed=None, gamma=0.9, reward_machine=False, shaped_rewards=False):
    while True:
        cmd, action, dfa = conn.recv() # removed task step count
        dfa: List[CrossProductDFA]
        if cmd == "step":  # Worker step command from Pipe
            obs, reward, done, info = env.step(action)
            # Compute the DFA progress
            if reward_machine:
                Phi = [d.Phi[d.statespace_mapping[d.product_state]] for d in dfa]
                [d.next({'env': env, 'word': None, 'action': action}) for d in dfa]
                Phi_prime = [d.Phi[d.statespace_mapping[d.product_state]] for d in dfa]
            else:
                [d.next({'env': env, 'word': None, 'action': action}) for d in dfa]
            # Compute the task rewards from the xDFA
            # to do this we require:
            #   * the current state
            #   * next state
            #   * reward
            #   * reward machine reward
            r = [d.rewards(one_off_reward) for d in dfa]
            if reward_machine:
                task_rewards = (np.array(r) + gamma * np.array(Phi_prime) - np.array(Phi)).tolist()
            else:
                if shaped_rewards:
                    distance_rewards = [d_.distance[d.product_state[i]] / n_coeff2 for d in dfa for
                                        (i, d_) in enumerate(d.dfas)]
                    task_rewards = [[r_[0] + distance_rewards[i]] for i, r_ in enumerate(r)]
                else:
                    task_rewards = r
            agent_reward = [0.0 if d.done() else -1.0 / n_coeff for d in dfa]
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
            normalisation_coef=1.0,
            normalisation_coef2=1.0,
            seed=None,
            gamma=0.9,
            reward_machine=False,
            shaped_rewards=False):  # removed max steps from signature
        self.envs = envs
        self.seed = seed
        self.num_agents = num_agents
        self.gamma = gamma
        if self.envs:
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
        self.dfas: List[List[CrossProductDFA]] = dfas  # A copy of the cross product DFA for each proc
        self.one_off_reward = one_off_reward  # One off reward given on task completion
        self.n_coeff = normalisation_coef
        self.n2_coeff = normalisation_coef2
        self.reward_machine = reward_machine
        self.shaped_rewards = shaped_rewards
        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, one_off_reward, num_agents,
                                             self.n_coeff,self.n2_coeff,seed, gamma, reward_machine, shaped_rewards))
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
        if self.reward_machine:
            Phi = [d.Phi[d.statespace_mapping[d.product_state]] for d in self.dfas[0]]
            [d.next({'env': self.envs[0], 'word': None, 'action': actions[0]}) for d in self.dfas[0]]
            Phi_prime = [d.Phi[d.statespace_mapping[d.product_state]] for d in self.dfas[0]]
        else:
            [d.next({'env': self.envs[0], 'word': None, 'action': actions[0]}) for d in self.dfas[0]]
        # Compute the task rewards from the xDFA
        # print()
        agent_rewards = [0.0 if d.done() else -1.0 / self.n_coeff for d in self.dfas[0]]
        r = [d_.rewards(self.one_off_reward) for d_ in self.dfas[0]]
        if self.reward_machine:
            task_rewards = (np.array(r) + self.gamma * np.array(Phi_prime) - np.array(Phi)).tolist()
        else:
            if self.shaped_rewards:
                distance_rewards = [d_.distance[d.product_state[i]] / self.n2_coeff for d in self.dfas[0] for (i, d_) in enumerate(d.dfas)]
                task_rewards = [[r_[0] + distance_rewards[i]] for i,r_ in enumerate(r)]
            else:
                task_rewards = r
            #task_rewards = r + [[d.distance[q]] for d in self.dfas[0] for q in d.product_state]
        #print("states \n", [d.product_state for d in self.dfas[0]], "\nshaped task rewards \n", task_rewards)
        #print()
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
        #print("reward ", reward_)
        # Concatenate the environment state and the DFA progress states for each task
        obs_ = np.array([np.append(obs[k], self.dfas[0][k].progress) for k in range(self.num_agents)])
        results = list(zip(*[(obs_, reward_, done, self.dfas[0])] + [local.recv() for local in self.locals]))
        self.dfas = list(results[3])
        return np.array(results[0], dtype=np.float32), np.array(results[1], dtype=np.float32), np.array(results[2], np.int32)

    def render(self):
        raise NotImplementedError