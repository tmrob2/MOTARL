from abc import abstractmethod


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

    def accepting(self, state):
        if any(x == state for x in self.acc):
            return True
        else:
            return False

    def reward_assigned(self):
        assigned = True if self.complete else False
        return assigned

    def start(self):
        self.current_state = self.next(self.start_state, None)
        return self.current_state

    def non_reachable(self, state):
        if state in self.rej:
            self.dead = True

