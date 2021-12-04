from showml.simulations.conways_game_of_life.event import Action
from showml.simulations.conways_game_of_life.grid import Grid
from showml.simulations.conways_game_of_life.controller import Controller


class GameOfLife:
    """A simulation of Conway's Game of Life (Cellular Automaton): https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
    """

    def __init__(self, controller: Controller) -> None:
        """Constructor for the GameOfLife class.

        Args:
            controller (Controller): The game controller responsible for reporting any events as well as handling the Grid.
        """
        self.controller = controller
        self.delay: int = 0
        self.game_running: bool = False

    def start_event(self) -> None:
        """This method runs when the game starts.
        """
        self.game_running = True
        self.delay = 150

    def stop_event(self) -> None:
        """This method runs when the game is stopped.
        """
        self.game_running = False
        self.delay = 0

    def reset_event(self) -> None:
        """This method runs when the RESET button is pressed.
        """
        self.game_running = False
        self.controller.grid.reset_grid()
        self.delay = 0

    def cell_toggle_event(self, row: int, column: int) -> None:
        """This method runs when a cell value is toggled by clicking on the cell in the grid.

        Args:
            row (int): The row index of the cell which has been toggled.
            column (int): The column index of the cell which has been toggled.
        """
        self.controller.grid.toggle_cell_value(row, column)

    def run(self) -> None:
        """This method runs the game loop by communicating with the Game Controller to receive event information.
        """
        while True:
            if self.game_running:
                self.controller.grid.update_grid()

            event = self.controller.get_event()

            if event.action == Action.START:
                self.start_event()

            elif event.action == Action.STOP:
                self.stop_event()

            elif event.action == Action.RESET:
                self.reset_event()

            elif event.action == Action.CELL_TOGGLE:
                self.cell_toggle_event(event.row, event.column)

            self.controller.display_window_and_grid(self.delay)


grid = Grid(num_rows=50, num_cols=100)
controller = Controller(grid)
GameOfLife(controller).run()
