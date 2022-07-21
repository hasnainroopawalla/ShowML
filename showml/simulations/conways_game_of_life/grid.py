import numpy as np


class Grid:
    """A 2D Grid class to display the cells."""

    def __init__(self, num_rows: int, num_columns: int) -> None:
        """[summary]

        Args:
            num_rows (int): The number of rows in the grid.
            num_columns (int): The number of columns in the grid.
        """
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.grid: np.ndarray = np.zeros((num_rows, num_columns))

        # Set initial live cells
        self.grid[9][11] = 1
        self.grid[10][11] = 1
        self.grid[11][11] = 1
        self.grid[10][12] = 1
        self.grid[10][10] = 1

    def calculate_neighbor_sum(self, row: int, column: int) -> int:
        """This method computes the sum of the 8-neighbors of the cell at [row, column].
        Reference: https://stackoverflow.com/a/37026344

        Args:
            row (int): The row index.
            column (int): The column index.

        Returns:
            int: Sum of the 8 neighbors of the cell.
        """
        # Extract the region (8-neighbors and self) of element at [row, column]
        region = self.grid[max(0, row - 1) : row + 2, max(0, column - 1) : column + 2]
        # Sum the region and subtract center
        return np.sum(region) - self.grid[row, column]

    def update_grid(self) -> None:
        """This method updates the grid using the rules of Conway's Game of Life.
        Reference: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
        """
        new_grid = np.zeros((self.num_rows, self.num_columns))
        for row in range(self.num_rows):
            for column in range(self.num_columns):
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

    def reset_grid(self) -> None:
        """This method resets the grid i.e., kills all the cells."""
        self.grid = np.zeros((self.num_rows, self.num_columns))

    def toggle_cell_value(self, row: int, column: int) -> None:
        """This method toggles the value of a cell (sets to 1 if 0 else 0)

        Args:
            row (int): The row index.
            column (int): The column index.
        """
        self.grid[row][column] = 1 if self.grid[row][column] == 0 else 0
