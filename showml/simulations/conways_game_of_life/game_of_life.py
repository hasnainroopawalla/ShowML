import pygame

from showml.simulations.conways_game_of_life.grid import Grid
from showml.simulations.conways_game_of_life.window import Window


class GameOfLife:
    def __init__(self, window: Window):
        self.window = window
        self.delay = 0
        self.game_running = False
        self.clock = pygame.time.Clock()

    def start_event(self):
        self.game_running = True
        self.delay = 150

    def stop_event(self):
        self.game_running = False
        self.delay = 0

    def reset_event(self):
        self.game_running = False
        self.window.grid.reset_grid()
        self.delay = 0

    def quit_event(self):
        pygame.quit()

    def cell_toggle_event(self, row, column):
        self.window.grid.toggle_cell_value(row, column)

    def simulate(self):
        while True:
            if self.game_running:
                self.window.grid.update_grid()

            event = self.window.get_event()

            if event["state"] == "START":
                self.start_event()

            elif event["state"] == "STOP":
                self.stop_event()

            elif event["state"] == "RESET":
                self.reset_event()

            elif event["state"] == "QUIT":
                self.quit_event()

            elif event["state"] == "CELL_TOGGLE":
                self.cell_toggle_event(event["row"], event["column"])

            pygame.time.wait(self.delay)

            self.window.display_board()
            self.window.display_grid()

            pygame.display.flip()
            self.clock.tick(60)


G = Grid(num_rows=50, num_cols=100)
W = Window(G)
GameOfLife(W).simulate()
