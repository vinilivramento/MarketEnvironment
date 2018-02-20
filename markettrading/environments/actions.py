from enum import Enum, unique

@unique
class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
   
    @classmethod
    def size(cls):
        return 3

@unique
class Position(Enum):
    IDLE = 0
    LONG = 1
    SHORT = 2

