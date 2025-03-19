import numpy as np
import pickle
from utils import EMOJI_ICONS

class Tile:
    def __init__(self, label, row, col):
        self.label = label
        self.row = row
        self.col = col
        
        self.up = None
        self.down = None
        self.left = None
        self.right = None

    def get_neighbour(self, direction):
        return getattr(self, direction)

    def disconnect(self):
        if self.up:
            self.up.down = None
        if self.down:
            self.down.up = None
        if self.left:
            self.left.right = None
        if self.right:
            self.right.left = None
        self.up = self.down = self.left = self.right = None

class Blank(Tile):
    def __init__(self, row, col):
        super().__init__(0, row, col)
        self.blank = True

class Boundary:
    def __init__(self, direction):
        self.label= f"{direction}_bound"


class Board:
    def __init__(self, board):
        self.m, self.n = board.shape
        self.grid = []
        for i in range(self.m):
            cur_row = []
            for j in range(self.n):
                if board[i, j] == 0:
                    cur_row.append(Blank(i, j))
                else:
                    cur_row.append(Tile(board[i, j], i, j))
            self.grid.append(cur_row)
        self._link_tiles()

    def _link_tiles(self):
        for i in range(self.m):
            for j in range(self.n):
                tile = self.grid[i][j]
                tile.up = self.grid[i-1][j] if i > 0 else Boundary("up")
                tile.down = self.grid[i+1][j] if i < self.m - 1 else Boundary("down")
                tile.left = self.grid[i][j-1] if j > 0 else Boundary("left")
                tile.right = self.grid[i][j+1] if j < self.n - 1 else Boundary("right")

    def _update_two_way_pointers(self, tile_coord, to_connect): 

        row, col = tile_coord 

        up = self.grid[row-1][col] if row > 0 else Boundary("up")
        down = self.grid[row+1][col] if row < self.m - 1 else Boundary("down")
        left = self.grid[row][col-1] if col > 0 else Boundary("left")
        right = self.grid[row][col+1] if col < self.n - 1 else Boundary("right")

        to_connect.up = up
        to_connect.down = down
        to_connect.left = left
        to_connect.right = right

        if type(up) is not Boundary:
            up.down = to_connect
        if type(down) is not Boundary:
            down.up = to_connect
        if type(left) is not Boundary:
            left.right = to_connect
        if type(right) is not Boundary:
            right.left = to_connect
    
    def __getitem__(self, pos):
        i, j = pos
        return self.grid[i][j]

    def show(self):
        # Print column headers.
        print("   ", end="")
        for col in range(self.n):
            print(f"{col:2}", end="")
        print()

        # Print each row with its row number.
        for i, row in enumerate(self.grid):
            print(f" {i:2} ", end="")
            for tile in row:
                # If the tile's value is 0, display a blank space; otherwise display the icon.
                if tile.label== 0:
                    print("  ", end="")
                else:
                    icon = EMOJI_ICONS[tile.label]
                    print(f"{icon}", end="")
            print()

    def remove_tile(self, tile):
        row, col = tile.row, tile.col
        tile.disconnect()
        self.grid[row][col] = Blank(tile.row, tile.col)
        self._update_two_way_pointers((row, col), self.grid[row][col])

    
    def all_cleared(self):
        try:
            for i in range(self.m):
                for j in range(self.n):
                    assert type(self.grid[i][j]) is Blank
            return True
        except AssertionError:
            return False

    def copy(self):
        new_board_labels = np.array([[tile.label for tile in row] for row in self.grid])
        return Board(new_board_labels)

    def move_tile(self, tile, new_coord):
        new_row, new_col = new_coord

        tile.disconnect()
        self.remove_tile(tile)
        
        tile.row, tile.col = new_row, new_col
        self.grid[new_row][new_col] = tile

        self._update_two_way_pointers((new_row, new_col), tile)


    # Add additional methods for moving tiles, removing tiles, etc.
if __name__ == "__main__":
    with open(".cache/test1/board.pkl", "rb") as f:
        board = pickle.load(f)

    print(board)
    board = Board(board)
    board.show()
    print(board.all_cleared())
    print(board[*np.array([0, 0])].label)
