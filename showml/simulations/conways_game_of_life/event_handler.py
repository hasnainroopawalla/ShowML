import pygame
from showml.simulations.conways_game_of_life.game_window import GameWindow
from showml.simulations.conways_game_of_life.event import Action, Event


class EventHandler:
    """The EventHandler class responsible for observing events taking place in the window as well as initializing and managing the Grid."""

    def __init__(self, window: GameWindow) -> None:
        """Constructor for the EventHandler class.

        Args:
            window (GameWindow): A game window object where the grid and buttons/text live.
        """
        self.window = window

    def get_event(self) -> Event:
        """This method returns an Event object based on the Action taken by the user.
        It checks the collide point of the user's mouse click with the different entities in the window.

        Returns:
            Event: An Event object containing the Action performed by the user (and also the row, column if a cell is toggled).
        """
        for event in pygame.event.get():
            x, y = pygame.mouse.get_pos()

            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if x < self.window.SCREEN_WIDTH - 100 and y:
                    row = y // (self.window.CELL_HEIGHT + self.window.CELL_MARGIN)
                    column = x // (self.window.CELL_WIDTH + self.window.CELL_MARGIN)
                    return Event(action=Action.CELL_TOGGLE, row=row, column=column)
                else:
                    # Check if a button is pressed
                    for button in self.window.buttons:
                        if button.button.collidepoint(x, y):
                            return button.on_click()

        return Event(action=Action.NO_EVENT)
