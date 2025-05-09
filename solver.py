import numpy as np
import pickle
import os
import time
from components import Board, Tile, Boundary, Blank
from bricksnbricks import BricksNBricks


class BricksNBricksSolver:
    """
    A solver for the BricksNBricks game using depth-first search to find a valid solution.
    """
    def __init__(self, board_array):
        """
        Initialize the solver with a board array.
        
        Args:
            board_array: numpy array representing the initial board state
        """
        self.initial_board_array = board_array
        self.game = BricksNBricks(board_array)
        self.directions = ["up", "down", "left", "right"]
        self.max_steps = {
            "up": board_array.shape[0],
            "down": board_array.shape[0],
            "left": board_array.shape[1],
            "right": board_array.shape[1]
        }
        
        # Solution path and visited states
        self.solution_path = []
        self.visited_states = set()
        
        # Statistics
        self.moves_explored = 0
        self.start_time = None
        self.end_time = None
    
    def find_valid_moves(self, board):
        """
        Find all valid moves on the current board.
        
        Returns:
            List of tuples (src, action) that represent valid moves
        """
        valid_moves = []
        
        # Iterate over all tiles on the board
        for i in range(board.m):
            for j in range(board.n):
                tile = board[i, j]
                
                # Skip empty spaces
                if isinstance(tile, Blank):
                    continue
                
                src = np.array([i, j])
                
                # Try all four directions
                for direction in self.directions:
                    # Try different step sizes
                    for steps in range(1, self.max_steps[direction] + 1):
                        # Create action vector
                        action = np.array(self.game.DIRECTIONS_DICT[direction]) * steps
                        
                        # Check if move is valid
                        if self.is_valid_move(board, src, action, direction):
                            valid_moves.append((src, action))
        
        return valid_moves
    
    def is_valid_move(self, board, src, action, direction):
        """
        Check if a move is valid without actually performing it.
        
        Args:
            board: The current board state
            src: Source position as [row, col]
            action: Action vector as [row_change, col_change]
            direction: Direction string ("up", "down", "left", "right")
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Get destination
        dst = src + action
        
        # Check if destination is within bounds
        if not (0 <= dst[0] < board.m and 0 <= dst[1] < board.n):
            return False
        
        # Get source tile
        src_tile = board[src[0], src[1]]
        src_label = src_tile.label
        
        # Get destination tile
        dst_tile = board[dst[0], dst[1]]
        
        # Check if this is a direct removal
        # Case 1: Direct adjacent with same label
        if np.sum(np.abs(action)) == 1 and isinstance(dst_tile, Tile) and dst_tile.label == src_label:
            return True
        
        # Scan the axis to check more complex cases
        # Temporarily set game state for scanning
        self.game.board = board
        self.game.src = src
        self.game.action = action
        self.game.direction = direction
        self.game.cur_label = src_label
        
        # Use game's scanning function
        n_blanks, neighbours = self.game._scan_axis()
        
        # Case 2: No blanks but there's a matching neighbor
        if n_blanks == 0 and len(neighbours) > 0 and neighbours[0].label == src_label:
            return True
        
        # Case 3: Blanks, no neighbors, and destination has same label
        if n_blanks >= 1 and len(neighbours) == 0 and isinstance(dst_tile, Tile) and dst_tile.label == src_label:
            return True
        
        # Case 4: Blanks, at least one neighbor with same label, and small movement
        if n_blanks >= 1 and len(neighbours) >= 1 and neighbours[0].label == src_label and np.sum(np.abs(action)) == 1:
            return True
        
        # Case 5: Can move and potentially match after movement
        if n_blanks >= np.sum(np.abs(action)):
            # Create temporary board to apply movement
            temp_board = board.copy()
            
            # Apply movement to check if it creates a match
            tile_to_move = temp_board[src[0], src[1]]
            temp_board.move_tile(tile_to_move, dst)
            
            # Check if there's a match in other directions
            for check_dir in [d for d in self.directions if d != direction]:
                cur_tile = temp_board[dst[0], dst[1]]
                while not isinstance(cur_tile, Boundary):
                    cur_tile = cur_tile.get_neighbour(check_dir)
                    if isinstance(cur_tile, Tile) and cur_tile.label == src_label:
                        return True
                    elif not isinstance(cur_tile, Blank):
                        break
            
            # No match found after movement
            return False
        
        # Otherwise, not a valid move
        return False
    
    def apply_move(self, board, src, action):
        """
        Apply a move to the board and return the new board state.
        
        Args:
            board: The current board state
            src: Source position [row, col]
            action: Action vector [row_change, col_change]
            
        Returns:
            New board state after applying the move
        """
        # Create a copy of the board
        new_board = board.copy()
        
        # Get movement direction
        direction_vec = action / max(1, np.sum(np.abs(action)))
        direction = None
        for dir_name, dir_vec in self.game.DIRECTIONS_DICT.items():
            if np.array_equal(direction_vec, dir_vec):
                direction = dir_name
                break
        
        # Set up game state for move processing
        self.game.board = new_board
        self.game.src = src
        self.game.action = action
        self.game.direction = direction
        self.game.dst = src + action
        self.game.cur_label = new_board[src[0], src[1]].label
        
        # Process the move
        self.game.process_move(verbose=0)
        
        # Return the updated board
        return self.game.board
    
    def solve(self, time_limit=None):
        """
        Solve the game using depth-first search.
        
        Args:
            time_limit: Optional time limit in seconds
            
        Returns:
            True if a solution was found, False otherwise
        """
        self.start_time = time.time()
        self.solution_path = []
        self.moves_explored = 0
        
        # Start the search
        result = self._dfs(self.game.board, [], time_limit)
        
        self.end_time = time.time()
        return result
    
    def _dfs(self, board, path, time_limit):
        """
        Recursive depth-first search to find a solution.
        
        Args:
            board: Current board state
            path: Current path of moves
            time_limit: Time limit in seconds
            
        Returns:
            True if a solution was found, False otherwise
        """
        # Check time limit
        if time_limit and time.time() - self.start_time > time_limit:
            print("Time limit reached.")
            return False
        
        # Check if board is cleared
        if board.all_cleared():
            self.solution_path = path.copy()
            return True
        
        # Find valid moves from current state
        valid_moves = self.find_valid_moves(board)
        
        # Try each move
        for src, action in valid_moves:
            self.moves_explored += 1
            
            # Apply the move
            new_board = self.apply_move(board, src, action)
            
            # Add move to path
            path.append((src, action))
            
            # Recursively search
            if self._dfs(new_board, path, time_limit):
                return True
            
            # Backtrack
            path.pop()
        
        # No solution found from this state
        return False
    
    def print_solution(self):
        """
        Print the solution path.
        """
        if not self.solution_path:
            print("No solution found.")
            return
        
        print(f"Solution found with {len(self.solution_path)} moves:")
        print(f"Total moves explored: {self.moves_explored}")
        print(f"Time taken: {self.end_time - self.start_time:.2f} seconds")
        
        for i, (src, action) in enumerate(self.solution_path):
            # Convert action to direction and steps
            direction = None
            steps = np.max(np.abs(action))
            for dir_name, dir_vec in self.game.DIRECTIONS_DICT.items():
                if np.array_equal(action / steps, dir_vec):
                    direction = dir_name
                    break
            
            print(f"Move {i+1}: Tile at {src[0]},{src[1]} -> {direction} {steps}")
    
    def save_solution(self, filename):
        """
        Save the solution to a CSV file.
        
        Args:
            filename: Path to save the solution
        """
        import pandas as pd
        import json
        
        if not self.solution_path:
            print("No solution to save.")
            return
        
        # Convert solution path to dataframe
        data = {"src": [], "action": []}
        for src, action in self.solution_path:
            data["src"].append(json.dumps(src.tolist()))
            data["action"].append(json.dumps(action.tolist()))
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Solution saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Load a board
    with open(".cache/test1/board.pkl", "rb") as f:
        board_array = pickle.load(f)
    
    # Create solver
    solver = BricksNBricksSolver(board_array)
    
    # Solve with a time limit of 60 seconds
    if solver.solve(time_limit=60):
        solver.print_solution()
        # solver.save_solution(".cache/test1/solution.csv")
    else:
        print("Could not find a solution within the time limit.")
