import pygame


class GameOfLife:
    def __init__(self, num_cols=100, num_rows=50):
        self.BLACK = (0, 0, 0)
        self.GRAY = (50, 50, 50)
        self.WHITE = (255, 255, 255)

        self.CELL_WIDTH = 9
        self.CELL_HEIGHT = 9
        self.CELL_MARGIN = 1

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.SCREEN_WIDTH = num_cols * self.CELL_WIDTH + num_cols + 100
        self.SCREEN_HEIGHT = num_rows * self.CELL_HEIGHT + num_rows

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

        self.grid = [[0 for x in range(num_cols)] for y in range(num_rows)]

        # Set initial live cells
        self.grid[2][3] = 1
        self.grid[3][3] = 1
        self.grid[3][4] = 1
        self.grid[2][4] = 1

        self.grid[4][5] = 1
        self.grid[5][5] = 1
        self.grid[4][6] = 1
        self.grid[5][6] = 1

    def calculate_neighbor_sum(self, row, column):
        if row == 0 and column == 0:
            neighbor_sum = (
                self.grid[row + 1][column]
                + self.grid[row][column + 1]
                + self.grid[row + 1][column + 1]
            )
        elif row == 0 and column == self.num_cols - 1:
            neighbor_sum = (
                self.grid[row][column - 1]
                + self.grid[row + 1][column]
                + self.grid[row + 1][column - 1]
            )
        elif row == 0:
            neighbor_sum = (
                self.grid[row + 1][column]
                + self.grid[row][column + 1]
                + self.grid[row][column - 1]
                + self.grid[row + 1][column - 1]
                + self.grid[row + 1][column + 1]
            )

        elif column == 0 and row == self.num_rows - 1:
            neighbor_sum = (
                self.grid[row - 1][column]
                + self.grid[row][column + 1]
                + self.grid[row - 1][column + 1]
            )
        elif column == 0:
            neighbor_sum = (
                self.grid[row - 1][column]
                + self.grid[row][column + 1]
                + self.grid[row + 1][column]
                + self.grid[row - 1][column + 1]
                + self.grid[row + 1][column + 1]
            )

        elif column == self.num_cols - 1 and row == self.num_rows - 1:
            neighbor_sum = (
                self.grid[row - 1][column]
                + self.grid[row][column - 1]
                + self.grid[row - 1][column - 1]
            )

        elif row == self.num_rows - 1:
            neighbor_sum = (
                self.grid[row - 1][column]
                + self.grid[row][column + 1]
                + self.grid[row][column - 1]
                + self.grid[row - 1][column - 1]
                + self.grid[row - 1][column + 1]
            )

        elif column == self.num_cols - 1:
            neighbor_sum = (
                self.grid[row][column - 1]
                + self.grid[row - 1][column]
                + self.grid[row + 1][column]
                + self.grid[row - 1][column - 1]
                + self.grid[row + 1][column - 1]
            )

        else:
            neighbor_sum = (
                self.grid[row - 1][column - 1]
                + self.grid[row - 1][column]
                + self.grid[row - 1][column + 1]
                + self.grid[row][column - 1]
                + self.grid[row][column + 1]
                + self.grid[row + 1][column - 1]
                + self.grid[row + 1][column]
                + self.grid[row + 1][column + 1]
            )

        return neighbor_sum

    def update_grid(self):
        new_grid = [[0 for x in range(self.num_cols)] for y in range(self.num_rows)]
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                neighbor_sum = self.calculate_neighbor_sum(row, column)
                if self.grid[row][column] == 1:
                    if neighbor_sum == 2 or neighbor_sum == 3:
                        new_grid[row][column] = 1
                    else:
                        new_grid[row][column] = 0
                elif self.grid[row][column] == 0:
                    if neighbor_sum == 3:
                        new_grid[row][column] = 1
                    else:
                        new_grid[row][column] = 0
        self.grid = new_grid

    def start(self):
        done = False
        clock = pygame.time.Clock()
        game_running = False
        while not done:
            old_grid = self.grid
            if game_running:
                self.update_grid()
            if old_grid == self.grid:
                game_running = False
                delay = 0
            for event in pygame.event.get():
                x, y = pygame.mouse.get_pos()
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.START_BUTTON.collidepoint(x, y):
                        game_running = True
                        delay = 150
                        print("START")
                    if self.STOP_BUTTON.collidepoint(x, y):
                        game_running = False
                        delay = 0
                        print("STOP")
                    if self.RESET_BUTTON.collidepoint(x, y):
                        self.grid = [
                            [0 for x in range(self.num_cols)]
                            for y in range(self.num_rows)
                        ]
                        game_running = False
                        delay = 0
                        print("RESET")
                    elif x < self.SCREEN_WIDTH - 100 and y:
                        column = x // (self.CELL_WIDTH + self.CELL_MARGIN)
                        row = y // (self.CELL_HEIGHT + self.CELL_MARGIN)
                        self.grid[row][column] = 1 if self.grid[row][column] == 0 else 0

            pygame.time.wait(delay)

            self.screen.fill(self.BLACK)
            pygame.draw.rect(
                self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 10, 60, 20)
            )
            pygame.draw.rect(
                self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 40, 60, 20)
            )
            pygame.draw.rect(
                self.screen, self.WHITE, (self.SCREEN_WIDTH - 80, 70, 60, 20)
            )

            self.screen.blit(
                pygame.font.SysFont("Arial", 15).render("Start", True, self.BLACK),
                (self.SCREEN_WIDTH - 65, 14),
            )
            self.screen.blit(
                pygame.font.SysFont("Arial", 15).render("Stop", True, self.BLACK),
                (self.SCREEN_WIDTH - 65, 40),
            )
            self.screen.blit(
                pygame.font.SysFont("Arial", 15).render("Reset", True, self.BLACK),
                (self.SCREEN_WIDTH - 65, 70),
            )

            for row in range(self.num_rows):
                for column in range(self.num_cols):
                    if self.grid[row][column] == 1:
                        color = self.WHITE
                    else:
                        color = self.GRAY
                    pygame.draw.rect(
                        self.screen,
                        color,
                        [
                            self.CELL_MARGIN
                            + (self.CELL_MARGIN + self.CELL_WIDTH) * column,
                            self.CELL_MARGIN
                            + (self.CELL_MARGIN + self.CELL_HEIGHT) * row,
                            self.CELL_WIDTH,
                            self.CELL_HEIGHT,
                        ],
                    )

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
