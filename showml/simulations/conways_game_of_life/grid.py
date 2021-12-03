class Grid:
    def __init__(self, num_rows, num_cols) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

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
        new_grid = [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
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

    def reset_grid(self):
        self.grid = [[0 for x in range(self.num_cols)] for y in range(self.num_rows)]

    def toggle_cell_value(self, row, column):
        self.grid[row][column] = 1 if self.grid[row][column] == 0 else 0
