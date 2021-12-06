import pygame
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

        self.BLACK = (0, 0, 0)
        self.GRAY = (50, 50, 50)
        self.WHITE = (255, 255, 255)

        self.CELL_WIDTH = 9
        self.CELL_HEIGHT = 9
        self.CELL_MARGIN = 1

        self.SCREEN_WIDTH = grid.num_columns * self.CELL_WIDTH + grid.num_columns + 100
        self.SCREEN_HEIGHT = grid.num_rows * self.CELL_HEIGHT + grid.num_rows

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Conway's Game of Life - Cellular Automaton")

        self.START_BUTTON = pygame.draw.rect(
            self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 10, 60, 20)
        )
        self.STOP_BUTTON = pygame.draw.rect(
            self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 40, 60, 20)
        )
        self.RESET_BUTTON = pygame.draw.rect(
            self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 70, 60, 20)
        )
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
        self.screen.fill(self.BLACK)
        pygame.draw.rect(self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 10, 60, 20))
        pygame.draw.rect(self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 40, 60, 20))
        pygame.draw.rect(self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 70, 60, 20))

        self.screen.blit(
            pygame.font.SysFont("Arial", 15).render("Start", True, self.BLACK),
            (self.SCREEN_WIDTH - 65, 14),
        )
        self.screen.blit(
            pygame.font.SysFont("Arial", 15).render("Stop", True, self.BLACK),
            (self.SCREEN_WIDTH - 65, 38),
        )
        self.screen.blit(
            pygame.font.SysFont("Arial", 15).render("Reset", True, self.BLACK),
            (self.SCREEN_WIDTH - 65, 68),
        )

    def _display_grid(self):
        """This private method displays the entire grid in the window.
        """
        for row in range(self.grid.num_rows):
            for column in range(self.grid.num_columns):
                if self.grid.grid[row][column] == 1:
                    color = self.WHITE
                else:
                    color = self.GRAY
                pygame.draw.rect(
                    self.screen,
                    color,
                    [
                        self.CELL_MARGIN
                        + (self.CELL_MARGIN + self.CELL_WIDTH) * column,
                        self.CELL_MARGIN + (self.CELL_MARGIN + self.CELL_HEIGHT) * row,
                        self.CELL_WIDTH,
                        self.CELL_HEIGHT,
                    ],
                )
