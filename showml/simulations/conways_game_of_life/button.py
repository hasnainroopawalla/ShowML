from abc import ABC, abstractmethod
from typing import Tuple
from showml.simulations.conways_game_of_life.config import Color
from showml.simulations.conways_game_of_life.event import Action, Event

import pygame


class Button(ABC):
    def __init__(
        self,
        text: str,
        text_x: int,
        text_y: int,
        width: int,
        height: int,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        screen: pygame.surface.Surface,
    ):
        self.text = text
        self.text_x = text_x
        self.text_y = text_y
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.color = color
        self.button = pygame.draw.rect(screen, color, (x, y, width, height))

    @abstractmethod
    def on_click(self) -> Event:
        pass


class StartButton(Button):
    def __init__(self, screen: pygame.surface.Surface, color: Tuple[int, int, int]):
        screen_info = screen.get_rect()
        w: int = screen_info[2]
        super().__init__(
            text="Start",
            text_x=w - 65,
            text_y=13,
            width=60,
            height=20,
            x=w - 80,
            y=10,
            color=color,
            screen=screen,
        )

    def on_click(self) -> Event:
        return Event(action=Action.START)


class StopButton(Button):
    def __init__(self, screen: pygame.surface.Surface, color: Tuple[int, int, int]):
        screen_info = screen.get_rect()
        w: int = screen_info[2]
        super().__init__(
            text="Stop",
            text_x=w - 65,
            text_y=42,
            width=60,
            height=20,
            x=w - 80,
            y=40,
            color=color,
            screen=screen,
        )

    def on_click(self) -> Event:
        return Event(action=Action.STOP)


class ResetButton(Button):
    def __init__(self, screen: pygame.surface.Surface, color: Tuple[int, int, int]):
        screen_info = screen.get_rect()
        w: int = screen_info[2]
        super().__init__(
            text="Reset",
            text_x=w - 68,
            text_y=72,
            width=60,
            height=20,
            x=w - 80,
            y=70,
            color=color,
            screen=screen,
        )

    def on_click(self) -> Event:
        return Event(action=Action.RESET)
