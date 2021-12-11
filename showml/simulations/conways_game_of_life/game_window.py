import pygame
from showml.simulations.conways_game_of_life.button import (
    StartButton,
    StopButton,
    ResetButton,
)
from showml.simulations.conways_game_of_life.config import GameWindowSettings, Colors
from showml.simulations.conways_game_of_life.grid import Grid


class GameWindow:
    """The GameWindow class responsible for observing events taking place in the window as well as initializing and managing the Grid.
    """

    def __init__(self, grid: Grid) -> None:
        """Constructor for the GameWindow class.

        Args:
            grid (Grid): A 2D grid containing cells where the simulation will take place.
        """
        self.grid = grid

        self.window_settings = GameWindowSettings()
        self.colors = Colors()

        self.CELL_HEIGHT = self.window_settings.CELL_HEIGHT
        self.CELL_WIDTH = self.window_settings.CELL_WIDTH
        self.CELL_MARGIN = self.window_settings.CELL_MARGIN

        self.SCREEN_WIDTH = grid.num_columns * self.CELL_WIDTH + grid.num_columns + 100
        self.SCREEN_HEIGHT = grid.num_rows * self.CELL_HEIGHT + grid.num_rows

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        pygame.display.set_caption(self.window_settings.CAPTION)

        self.buttons = [
            StartButton(self.screen, self.SCREEN_WIDTH, self.colors.WHITE),
            StopButton(self.screen, self.SCREEN_WIDTH, self.colors.WHITE),
            ResetButton(self.screen, self.SCREEN_WIDTH, self.colors.WHITE),
        ]

        self.clock = pygame.time.Clock()

    def display_window_and_grid(self, delay: int) -> None:
        """This method is repsonsible for displaying the entire Game window with the grid, buttons and textual entities.

        Args:
            delay (int): The delay in milliseconds between each iteration.
        """
        pygame.time.wait(delay)
        self._display_buttons_and_text()
        self._display_grid()
        pygame.display.flip()
        self.clock.tick(60)

    def _display_buttons_and_text(self):
        """This private method displays the buttons and the text objects in the window.
        """
        self.screen.fill(self.colors.BLACK)
        for button in self.buttons:
            # Button
            pygame.draw.rect(
                self.screen,
                button.color,
                (button.x, button.y, button.width, button.height),
            )

            # Button Text
            self.screen.blit(
                pygame.font.SysFont(self.window_settings.FONT, 15).render(
                    button.text, True, self.colors.BLACK
                ),
                (button.text_x, button.text_y),
            )

    def _display_grid(self):
        """This private method displays the entire grid in the window.
        """
        for row in range(self.grid.num_rows):
            for column in range(self.grid.num_columns):
                if self.grid.grid[row][column] == 1:
                    color = self.colors.WHITE
                else:
                    color = self.colors.GRAY
                pygame.draw.rect(
                    self.screen,
                    color,
                    [
                        self.window_settings.CELL_MARGIN
                        + (
                            self.window_settings.CELL_MARGIN
                            + self.window_settings.CELL_WIDTH
                        )
                        * column,
                        self.window_settings.CELL_MARGIN
                        + (
                            self.window_settings.CELL_MARGIN
                            + self.window_settings.CELL_HEIGHT
                        )
                        * row,
                        self.window_settings.CELL_WIDTH,
                        self.window_settings.CELL_HEIGHT,
                    ],
                )
