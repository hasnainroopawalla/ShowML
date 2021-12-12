from abc import ABC, abstractmethod
from showml.simulations.conways_game_of_life.event import Action, Event

import pygame


class Button(ABC):
    def __init__(self, text, text_x, text_y, width, height, x, y, color, screen):
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
    def on_click():
        pass


class StartButton(Button):
    def __init__(self, screen, color):
        _, _, w, _ = screen.get_rect()
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

    def on_click(self):
        return Event(action=Action.START)


class StopButton(Button):
    def __init__(self, screen, color):
        _ , _, w, _ = screen.get_rect()
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

    def on_click(self):
        return Event(action=Action.STOP)


class StopButton(Button):
    def __init__(self, screen: pygame.Surface, color):
        _, _, w, _ = screen.get_rect()
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

    def on_click(self):
        return Event(action=Action.STOP)


class ResetButton(Button):
    def __init__(self, screen, color):
        _, _, w, _ = screen.get_rect()
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

    def on_click(self):
        return Event(action=Action.RESET)
