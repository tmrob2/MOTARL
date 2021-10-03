from enum import IntEnum

class Coord:
    def __init__(self, x=0, y=0):
        self.x, self.y = x, y

    def set(self, x, y):
        self.x, self.y = x, y


class Action1(IntEnum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4
    PICK = 5
    PLACE = 6

