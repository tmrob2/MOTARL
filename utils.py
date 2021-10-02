from enum import Enum

class Coord:
    def __init__(self):
        self.x, self.y = 0, 0

    def set(self, x, y):
        self.x, self.y = x, y


class Action1(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4
    PICK = 5
    PLACE = 6

