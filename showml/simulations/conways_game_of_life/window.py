import pygame
from showml.simulations.conways_game_of_life.grid import Grid


class Window:
    def __init__(self, grid: Grid):
        self.BLACK = (0, 0, 0)
        self.GRAY = (50, 50, 50)
        self.WHITE = (255, 255, 255)

        self.CELL_WIDTH = 9
        self.CELL_HEIGHT = 9
        self.CELL_MARGIN = 1

        self.grid = grid

        self.SCREEN_WIDTH = (
            self.grid.num_cols * self.CELL_WIDTH + self.grid.num_cols + 100
        )
        self.SCREEN_HEIGHT = self.grid.num_rows * self.CELL_HEIGHT + self.grid.num_rows

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

    def get_event(self):
        for event in pygame.event.get():
            x, y = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                return {"state": "QUIT", "row": None, "column": None}

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.START_BUTTON.collidepoint(x, y):
                    return {"state": "START", "row": None, "column": None}

                if self.STOP_BUTTON.collidepoint(x, y):
                    return {"state": "STOP", "row": None, "column": None}

                if self.RESET_BUTTON.collidepoint(x, y):
                    return {"state": "RESET", "row": None, "column": None}

                elif x < self.SCREEN_WIDTH - 100 and y:
                    column = x // (self.CELL_WIDTH + self.CELL_MARGIN)
                    row = y // (self.CELL_HEIGHT + self.CELL_MARGIN)
                    return {"state": "CELL_TOGGLE", "row": row, "column": column}

        return {"state": None, "row": None, "column": None}

    def display_board(self):
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

    def display_grid(self):
        for row in range(self.grid.num_rows):
            for column in range(self.grid.num_cols):
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
