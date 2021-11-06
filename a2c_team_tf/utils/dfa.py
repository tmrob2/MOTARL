from abc import abstractmethod
from typing import List


class DFAStates:
    """Define a set of DFA States"""
    @abstractmethod
    def __init__(self):
        ...


class DFA:
    def __init__(self, start_state, acc, rej):
        self.handlers = {}
        self.start_state = start_state
        self.current_state = None
        self.acc = acc
        self.rej = rej
        self.dead = False
        self.complete = False

    def add_state(self, name, f):
        self.handlers[name] = f

    def set_start(self, name):
        self.start_state = name.upper()

    def next(self, state, data):
        if state is not None:
            f = self.handlers[state.upper()]
            new_state = f(data)
            return new_state
        else:
            return self.current_state

    def reset(self):
        self.current_state = self.start_state
        self.complete = False
        self.dead = False

    def accepting(self, state):
        if any(x == state for x in self.acc):
            return True
        else:
            return False

    def assign_reward(self, state, one_off_reward):
        if not self.complete and self.accepting(state):
            self.complete = True
            return one_off_reward
        else:
            return 0.0

    def start(self):
        self.current_state = self.next(self.start_state, None)
        return self.current_state


class CrossProductDFA:
    def __init__(self, num_tasks, dfas: List[DFA]):
        self.dfas = dfas
        self.product_state = self.start()
        self.num_tasks = num_tasks
        self.completed = [False] * self.num_tasks

    def start(self):
        return [dfa.start_state for dfa in self.dfas]

    def next(self, env):
        self.product_state = [dfa.next(self.product_state[i], env) for (i, dfa) in enumerate(self.dfas)]

    def rewards(self, one_off_reward):
        """
        :param ii: is the agent index
        """
        rewards = [dfa.assign_reward(self.product_state[i], one_off_reward) for (i, dfa) in enumerate(self.dfas)]
        for (i, r) in enumerate(rewards):
            if r == one_off_reward:
                self.completed[i] = True
        return rewards

    def reset(self):
        for dfa in self.dfas:
            dfa.reset()
        self.product_state = self.start()
        self.completed = [False] * self.num_tasks


