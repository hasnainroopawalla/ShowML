from showml.simulations.conways_game_of_life import GameOfLife
from showml.simulations.conways_game_of_life.grid import Grid
from showml.simulations.conways_game_of_life.controller import Controller

grid = Grid(num_rows=50, num_cols=100)
controller = Controller(grid)
GameOfLife(controller).run()
