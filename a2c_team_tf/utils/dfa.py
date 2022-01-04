import itertools
from abc import abstractmethod
from typing import List
from enum import IntEnum


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

    def __init__(self, start_state, acc, rej):
        self.handlers = {}
        self.start_state = start_state
        self.current_state = None
        self.acc = acc
        self.rej = rej
        self.states = []
        self.progress_flag = self.Progress.IN_PROGRESS
        self.words = {}

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

    def reset(self):
        for dfa in self.dfas:
            dfa.reset()
        self.progress = [dfa.progress_flag for dfa in self.dfas]
        self.product_state = self.start()

    def done(self):
        return all([dfa.progress_flag == 2 or dfa.progress_flag == -1 for dfa in self.dfas])


class RewardMachine(DFA):
    def __init__(self, start_state, acc, rej):
        super(RewardMachine, self).__init__(start_state=start_state, acc=acc, rej=rej)


class ProductRewardMachine:
    def __init__(self, RMs, one_off_reward):
        self.RMs = RMs
        self.product_states = self.start()
        self.state_space = []
        self.state_numbering = []
        self.statespace_mapping = {}
        self.product_words = []
        [self.product_words.extend(rm.words) for rm in RMs]
        self.one_off_reward = one_off_reward

    def compute_state_space(self):
        states = [dfa.states for dfa in self.RMs]
        self.state_space = list(itertools.product(*states))
        self.statespace_mapping = {k: v for k, v in enumerate(self.state_space)}
        self.state_numbering = list(range(len(self.state_space)))

    def next_(self, data):
        "A next function without side effects"
        product_state = tuple([rm.next(self.product_states[i], data, None) for (i, rm) in enumerate(self.RMs)])
        return product_state

    def start(self):
        return tuple([rm.start_state for rm in self.RMs])

    def rewards(self):
        rewards = [rm.assign_rewards(self.one_off_reward) for rm in self.RMs]
        return rewards

    def value_iteration(self):
        v = [0.] * len(self.state_space)
        eps = 1.0
        while eps > 0.:
            for q in self.state_space:
                pass









