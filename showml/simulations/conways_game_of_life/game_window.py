from typing import List, Tuple
import pygame
from showml.simulations.conways_game_of_life.button import (
    Button,
    StartButton,
    StopButton,
    ResetButton,
)
from showml.simulations.conways_game_of_life.config import GameWindowSettings, Color
from showml.simulations.conways_game_of_life.grid import Grid


class GameWindow:
    """The GameWindow class responsible for observing events taking place in the window as well as initializing and managing the Grid."""

    def __init__(self, grid: Grid) -> None:
        """Constructor for the GameWindow class.

        Args:
            grid (Grid): A 2D grid containing cells where the simulation will take place.
        """
        self.window_settings = GameWindowSettings()

        pygame.init()
        pygame.display.set_caption(self.window_settings.CAPTION)

        self.grid = grid

        self.CELL_HEIGHT = self.window_settings.CELL_HEIGHT
        self.CELL_WIDTH = self.window_settings.CELL_WIDTH
        self.CELL_MARGIN = self.window_settings.CELL_MARGIN

        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self._compute_screen_width_and_height()
        self.SCREEN = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.buttons: List[Button] = [
            StartButton(self.SCREEN, Color.WHITE),
            StopButton(self.SCREEN, Color.WHITE),
            ResetButton(self.SCREEN, Color.WHITE),
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

    def _display_buttons_and_text(self) -> None:
        """This private method displays the buttons and the text objects in the window."""
        self.SCREEN.fill(Color.BLACK)
        for button in self.buttons:
            # Button
            pygame.draw.rect(
                self.SCREEN,
                button.color,
                (button.x, button.y, button.width, button.height),
            )

            # Button Text
            self.SCREEN.blit(
                pygame.font.SysFont(
                    self.window_settings.FONT, self.window_settings.FONT_SIZE
                ).render(button.text, True, Color.BLACK),
                (button.text_x, button.text_y),
            )

    def _display_grid(self) -> None:
        """This private method displays the entire grid in the window."""
        for row in range(self.grid.num_rows):
            for column in range(self.grid.num_columns):
                if self.grid.grid[row][column] == 1:
                    # Alive cell
                    color = Color.WHITE
                else:
                    # Dead cell
                    color = Color.GRAY
                pygame.draw.rect(
                    self.SCREEN,
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

    def _compute_screen_width_and_height(self) -> Tuple[int, int]:
        """This method computes the width and height of the screen based on the defined cell size and number of rows and columns.

        Returns:
            int: The screen width
            int: The screen height
        """
        screen_width = (
            self.grid.num_columns * self.CELL_WIDTH + self.grid.num_columns + 100
        )
        screen_height = self.grid.num_rows * self.CELL_HEIGHT + self.grid.num_rows
        return screen_width, screen_height
