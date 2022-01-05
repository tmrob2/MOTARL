import itertools
from abc import abstractmethod
from typing import List
from enum import IntEnum
import numpy as np


class DFAStates:
    """Define a set of DFA States"""
    @abstractmethod
    def __init__(self):
        ...


class DFA:
    class Progress(IntEnum):
        FAILED = -1
        IN_PROGRESS = 0
        JUST_FINISHED = 1
        FINISHED = 2

    def __init__(self, start_state, acc, rej, words=None):
        self.handlers = {}
        self.start_state = start_state
        self.current_state = None
        self.acc = acc
        self.rej = rej
        self.states = []
        self.progress_flag = self.Progress.IN_PROGRESS
        if words is None:
            words = []
        self.words = words

    def add_state(self, name, f):
        self.states.append(name)
        self.handlers[name] = f

    def next(self, state, data, agent):
        if state is not None:
            f = self.handlers[state.upper()]
            new_state = f(data, agent)
            self.update_progress(new_state)
            return new_state
        else:
            return self.current_state

    def reset(self):
        self.current_state = self.start_state
        self.progress_flag = self.Progress.IN_PROGRESS

    def update_progress(self, state):
        if state in self.acc:
            if self.progress_flag < 1:
                self.progress_flag = self.Progress.JUST_FINISHED
            else:
                self.progress_flag = self.Progress.FINISHED
        elif state in self.rej:
            self.progress_flag = self.Progress.FAILED

    def assign_reward(self, one_off_reward):
        if self.progress_flag == self.Progress.JUST_FINISHED:
            return one_off_reward
        else:
            return 0.0


class RewardMachine(DFA):
    def __init__(self, start_state, acc, rej, words):
        super(RewardMachine, self).__init__(
            start_state=start_state, acc=acc, rej=rej, words=words)

    def next(self, state, data, agent):
        f = self.handlers[state.upper()]
        new_state = f(data, agent)
        progress = self.update_progress(state, new_state)
        return new_state, progress

    def update_progress(self, q, q_prime):
        if q not in self.acc and q_prime in self.acc:
            return self.Progress.JUST_FINISHED
        elif q in self.acc and q_prime in self.acc:
            return self.Progress.FINISHED
        elif q in self.rej or q_prime in self.rej:
            return self.Progress.FAILED
        else:
            return self.Progress.IN_PROGRESS

    def assign_reward(self, one_off_reward, progress):
        if progress == self.Progress.JUST_FINISHED:
            return one_off_reward
        else:
            return 0.0


class CrossProductDFA:
    def __init__(self, num_tasks, dfas: List[DFA], agent: int):
        self.dfas = dfas
        self.agent = agent
        self.product_state = self.start()
        self.num_tasks = num_tasks
        self.progress = []
        self.state_space = []
        self.state_numbering = []
        self.statespace_mapping = {}
        self.Phi = []  # a list of shaped rewards

    def start(self):
        return tuple([dfa.start_state for dfa in self.dfas])

    def next(self, data):
        self.product_state = tuple([dfa.next(self.product_state[i], data, self.agent) for (i, dfa) in enumerate(self.dfas)])
        self.progress = [dfa.progress_flag for dfa in self.dfas]

    def rewards(self, one_off_reward):
        """
        :param ii: is the agent index
        """
        rewards = [dfa.assign_reward(one_off_reward) for dfa in self.dfas]
        return rewards

    def assign_shaped_rewards(self, v):
        self.Phi = v

    def assign_reward_machine_mappings(self, state_space, statespace_mapping):
        self.state_space = state_space
        self.statespace_mapping = statespace_mapping

    def reset(self):
        for dfa in self.dfas:
            dfa.reset()
        self.progress = [dfa.progress_flag for dfa in self.dfas]
        self.product_state = self.start()

    def done(self):
        return all([dfa.progress_flag == 2 or dfa.progress_flag == -1 for dfa in self.dfas])


class RewardMachines:
    def __init__(self, dfas, one_off_reward, num_tasks):
        self.rms: List[RewardMachine] = dfas
        self.product_states = self.start()
        self.state_space = []
        self.state_numbering = []
        self.statespace_mapping = {}
        self.product_words = []
        self.concat_product_words()
        self.one_off_reward = one_off_reward
        self.num_tasks = num_tasks

    def compute_state_space(self):
        states = [rm.states for rm in self.rms]
        self.state_space = list(itertools.product(*states))
        self.statespace_mapping = {v: k for k, v in enumerate(self.state_space)}
        self.state_numbering = list(range(len(self.state_space)))

    def concat_product_words(self):
        for rm in self.rms:
            self.product_words.extend(rm.words)

    def start(self):
        return tuple([rm.start_state for rm in self.rms])

    def rewards(self, progress: List[RewardMachine.Progress]):
        rewards = [rm.assign_reward(self.one_off_reward, progress[i]) for (i, rm) in enumerate(self.rms)]
        return rewards

    def value_iteration(self, gamma):
        zero = np.finfo(np.float32).eps.item()
        v = np.full([len(self.state_space), self.num_tasks], 0.0)
        eps = self.one_off_reward
        count = 0
        while eps > zero:
            eps_q = np.full([len(self.state_space), self.num_tasks], 0.0)
            for i, qbar in enumerate(self.state_space):
                v_primes = []
                for w in self.product_words:
                    xtransition = [rm.next(qbar[i], {'env': None, 'word': w}, None) for i, rm in enumerate(self.rms)]
                    result = list(zip(*xtransition))
                    q_prime, progress = result[0], result[1]
                    rewards = self.rewards(progress)
                    #print(f"({qbar}, {w}) -> {q_prime}: {self.statespace_mapping[tuple(q_prime)]}")
                    q_prime_index = self.statespace_mapping[tuple(q_prime)]
                    v_prime = rewards + gamma * v[q_prime_index]
                    v_primes.append(v_prime)
                #print("v's ", v_primes)
                #print("max v'", max(v_primes, key=tuple))
                v_prime = max(v_primes, key=tuple)
                #print("v' ", v_prime)
                #print("max ", np.maximum(eps_q[i], np.abs(v[i] - v_prime)))
                eps_q[i] = np.maximum(eps_q[i], np.abs(v[i] - v_prime))
                #print("eps: \n", eps_q[i])
                v[i] = v_prime
                ##print("v \n", v)
            #print("eps_q \n", eps_q)
            #print("eps ", np.amax(eps_q))
            eps = np.amax(eps_q)
            count += 1
            #print()
        #for qv in list(zip(self.state_space, v)):
        #    print("q, value", qv)
        return v













