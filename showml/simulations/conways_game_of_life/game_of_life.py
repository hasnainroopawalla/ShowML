from showml.simulations.conways_game_of_life.event import Action
from showml.simulations.conways_game_of_life.game_window import GameWindow
from showml.simulations.conways_game_of_life.grid import Grid
from showml.simulations.conways_game_of_life.event_handler import EventHandler


class GameOfLife:
    """A simulation of Conway's Game of Life (Cellular Automaton): https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"""

    def __init__(self, num_rows: int = 50, num_columns: int = 100) -> None:
        """Constructor for the GameOfLife class.

        Args:
            num_rows (int, optional): Number of rows in the grid. Defaults to 50.
            num_columns (int, optional): Number of columns in the grid. Defaults to 100.
        """
        self.window = GameWindow(Grid(num_rows, num_columns))
        self.event_handler = EventHandler(self.window)
        self.delay: int = 0
        self.game_running: bool = False
        self.event_dict = {
            Action.START: self.start_event,
            Action.STOP: self.stop_event,
            Action.RESET: self.reset_event,
            Action.CELL_TOGGLE: self.cell_toggle_event,
            Action.NO_EVENT: self.no_event,
        }

    def start_event(self, row: int, column: int) -> None:
        """This method runs when the game starts."""
        self.game_running = True
        self.delay = 150

    def stop_event(self, row: int, column: int) -> None:
        """This method runs when the game is stopped."""
        self.game_running = False
        self.delay = 0

    def reset_event(self, row: int, column: int) -> None:
        """This method runs when the RESET button is pressed."""
        self.game_running = False
        self.window.grid.reset_grid()
        self.delay = 0

    def cell_toggle_event(self, row: int, column: int) -> None:
        """This method runs when a cell value is toggled by clicking on the cell in the grid.

        Args:
            row (int): The row index of the cell which has been toggled.
            column (int): The column index of the cell which has been toggled.
        """
        self.window.grid.toggle_cell_value(row, column)

    def no_event(self, row: int, column: int) -> None:
        pass

    def run(self) -> None:
        """This method runs the game loop by communicating with the EventHandler to receive event information."""
        while True:
            if self.game_running:
                self.window.grid.update_grid()

            event = self.event_handler.get_event()

            # Call the event function
            self.event_dict[event.action](event.row, event.column)

            self.window.display_window_and_grid(self.delay)
