from enum import Enum, auto
from dataclasses import dataclass


class Action(Enum):
    """This class defines the different Actions in the game."""

    START = auto()
    STOP = auto()
    RESET = auto()
    CELL_TOGGLE = auto()
    NO_EVENT = auto()


@dataclass
class Event:
    """This class defines an Event (and also the row, column if a cell is toggled)"""

    action: Action
    row: int = 0
    column: int = 0
