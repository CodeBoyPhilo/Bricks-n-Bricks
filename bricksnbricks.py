import numpy as np
import pandas as pd
import pickle
import os
import re
import pdb
import json
import argparse
from components import Board, Tile, Boundary, Blank
from utils import load_image, segment_objects, template_matching, EMOJI_ICONS
import time


class BricksNBricks:
    def __init__(self, board):
        self.board = Board(board)
        self.DIRECTIONS_DICT = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        
        self.DIRECTIONS_ARRAY = [list(d) for d in self.DIRECTIONS_DICT.values()]
        self.direction:str = None
        self.direction_idx = None
        self.src: np.array = None
        self.src_idx: int = None
        self.action: np.array = None
        self.dst: np.array = None
        self.dst_idx: int = None
        self.cur_label: int = None
        self.verbose: int = None

        self.src_history= []
        self.action_history = []

    def play(self, verbose=0, save_path=None, gamename=None):
        self.verbose = verbose

        while True:
            os.system("cls" if os.name == "nt" else "clear")

            if self.src is not None and self.action is not None:
                self.process_move(verbose=verbose)
                self._reset_move()
            else:
                print("Welcome to Bricks'n Bricks!")
                print("Enter 'e' at any prompt to exit the game.\n")

            self.board.show()

            if self.board.all_cleared():
                print("Wooo~ You win!")
                break

            user_src = input("row, column coordinate (e.g. 3,1): ")
            valid_src, src_val = self._parse_src(user_src)
            if valid_src == "exit":
                break
            if not valid_src:
                continue 

            user_action = input("action (e.g. 'down 7'): ")
            valid_action, direction, action_val = self._parse_action(user_action)
            if valid_action == "exit":
                break
            if not valid_action:
                print("Invalid action format or constraints. Please try again.\n")
                continue

            self.src = src_val
            self.action = action_val
            self.direction = direction
            self.cur_label = self.board[*self.src].label

            self.dst = self.src + self.action
            self.src_idx = self.src[0] * self.board.n + self.src[1]
            self.dst_idx = self.dst[0] * self.board.n + self.dst[1]

        assert save_path is not None 
        assert gamename is not None
        history_to_save = pd.DataFrame({"src": self.src_history, "action": self.action_history})
        history_to_save['src'] = history_to_save['src'].apply(lambda x: json.dumps(x.tolist()))
        history_to_save['action'] = history_to_save['action'].apply(lambda x: json.dumps(x.tolist()))
        history_to_save.to_csv(f"{save_path}/history.csv", index=False)

    def resume(self, history, verbose=0):
        moves = pd.read_csv(history)
        moves['src'] = moves['src'].apply(lambda x: np.array(json.loads(x)))
        moves['action'] = moves['action'].apply(lambda x: np.array(json.loads(x)))
        
        srcs, actions = moves["src"].values, moves["action"].values
        for src, action in zip(srcs, actions):
            # os.system("cls" if os.name == "nt" else "clear")  # Clear the console
            if self.src is not None and self.action is not None:
                self.process_move(verbose=verbose)
                self._reset_move()
            else:
                pass
            self.board.show()
            if self.board.all_cleared():
                print("Wooo~ You win!")
                break
            else:
                direction, steps = self._verbalise_actions(action)
                self.src = src
                self.action = action
                self.direction = direction
                self.cur_label = self.board[*self.src].label

                self.dst = self.src + self.action
                self.src_idx = self.src[0] * self.board.n + self.src[1]
                self.dst_idx = self.dst[0] * self.board.n + self.dst[1]

    def process_move(self, verbose=0):
        n_blanks, neighbours = self._scan_axis()
        # print(n_blanks, [EMOJI_ICONS[neighbour.label] for neighbour in neighbours])
        # --- Step 1: Check for direct removals
        if n_blanks == 0 and len(neighbours) == 0:
            if verbose == 1:
                print("Invalid!")
        elif self.is_direct_clear(n_blanks, neighbours):
            if verbose == 1:
                print("Direct removal!")
            self._clear_tile([self.board[*self.src], self.board[*self.dst]])
        else:
            # --- Step 2: Check for blanks and contiguous neighbours
            invalid_conditions = [
                (n_blanks == 0),
                (n_blanks < np.abs(self.action).sum())
            ]
            if any(invalid_conditions):
                if verbose == 1:
                    print("Not enough space to move!")
            else:
                # --- Step 3: Move boards to check for removals
                self._apply_action_on_temp_board([self.board[*self.src]] + neighbours)
                matched = self.match_tiles(self.dst, temp=True)
                # breakpoint()
                if len(matched) == 0:
                    if verbose == 1:
                        print("No matched tiles found.")
                else:
                    self.board = self.temp_board
                    self._clear_tile([self.board[*self.dst]] + matched)

    def is_direct_clear(self, n_blanks, neighbours):
        
        # sufficient_conditions = [
        #     (n_blanks == 0 and len(neighbours) >= 1),
        #     (n_blanks >= 1 and len(neighbours) <= 1)
        # ]
        # necessary_conditions = [
        #     (self.board[*self.dst].label == self.cur_label),
        #     (neighbours[0].label == self.cur_label)
        # ]

        conditions = [
            (n_blanks == 0 and neighbours[0].label == self.cur_label),
            (n_blanks >= 1 and len(neighbours) == 0 and self.board[*self.dst].label == self.cur_label),
            (n_blanks >= 1 and len(neighbours) >= 1 and neighbours[0].label == self.cur_label and np.sum(np.abs(self.action)) == 1)

        ]
        # TODO: sum of actions == 1

        # if any(sufficient_conditions) and all(necessary_conditions):
        #     return True
        if any(conditions):
            return True 
        else:
            return False

    def match_tiles(self, coord, temp=False):
        matched = []
        for direction in [d for d in ["up", "down", "left", "right"] if d != self.direction]:
            cur_tile = self.board[*coord] if not temp else self.temp_board[*coord]
            while type(cur_tile) is not Boundary:
                cur_tile = cur_tile.get_neighbour(direction)
                if cur_tile.label == self.cur_label:
                    matched.append(cur_tile)
                    break
                elif type(cur_tile) is not Blank:
                    break
                else:
                    continue

        if len(matched) >= 2:
            print("Multiple matched tiles found: " + ", ".join(f"[{tile.row}, {tile.col}]" for tile in matched))
            while True:
                try:
                    to_keep = int(input(f"Which tile to remove? index: (0~{len(matched)}): "))
                    matched= [matched[to_keep]]
                    break
                except Exception:
                    continue
        return matched

    def _parse_src(self, user_src):
        user_src = user_src.strip().lower()
        if user_src == "e":
            return ("exit", None)

        pattern = r'^[0-9]+\s*,\s*[0-9]+$'
        if not re.match(pattern, user_src):
            return (False, None)

        row_str, col_str = user_src.split(",")
        row_str, col_str = row_str.strip(), col_str.strip()

        if not (row_str.isdigit() and col_str.isdigit()):
            return (False, None)

        row, col = int(row_str), int(col_str)
        if row < 0 or row >= self.board.m or col < 0 or col >= self.board.n:
            return (False, None)

        return (True, np.array([row, col]))

    def _parse_action(self, user_action):
        user_action = user_action.strip().lower()
        if user_action == "e":
            return ("exit", None, None)

        pattern = r'^(up|down|left|right)\s+([0-9]+)$'
        match = re.match(pattern, user_action)
        if not match:
            return (False, None, None)

        direction_str, distance_str = match.groups()
        distance = int(distance_str)

        # Additional constraints
        if direction_str in ["up", "down"] and not (0 <= distance < 14):
            return (False, None, None)
        if direction_str in ["left", "right"] and not (0 <= distance < 10):
            return (False, None, None)

        action_vec = np.array(self.DIRECTIONS_DICT[direction_str]) * distance
        return (True, direction_str, action_vec)

    def _clear_tile(self, tiles):
        for tile in tiles:
            row, col = tile.row, tile.col
            self.board.remove_tile(tile)
            if self.verbose == 1:
                print(f"Cleared tile at {row}, {col}")
           
        self.src_history.append(self.src)
        self.action_history.append(self.action)

    def _scan_axis(self):
        cur_tile = self.board[*self.src]
        neighbours = []
        n_blanks = 0
        while True:
            cur_tile = cur_tile.get_neighbour(self.direction)
            stop = [
                # 1. reached boundary
                (type(cur_tile) is Boundary),
                # 2. reached a blocking tile
                (type(cur_tile) is Tile and n_blanks != 0)
            ]

            if any(stop):
                break
            elif type(cur_tile) is Tile and n_blanks == 0:
                neighbours.append(cur_tile)
            elif type(cur_tile) is Blank:
                n_blanks += 1
            else:
                pass
        return n_blanks, neighbours

    def _apply_action_on_temp_board(self, tiles_to_move):
        self.temp_board = self.board.copy()
        for tile in tiles_to_move[::-1]:
            new_coord = np.array([tile.row, tile.col]) + self.action
            self.temp_board.move_tile(tile, new_coord)

    def _reset_move(self):
        self.src = None
        self.action = None
        self.direction = None
        self.src_idx = None
        self.dst = None
        self.dst_idx = None
        self.cur_label = None

    def _verbalise_actions(self, action):
        axis = np.nonzero(action)[0][0]
        steps = np.abs(action[axis])
        direction = action / steps
        direction_idx = self.DIRECTIONS_ARRAY.index(direction.tolist())

        return [k for idx, k in enumerate(self.DIRECTIONS_DICT) if idx == direction_idx][0], steps



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="A minimal replicate of the Bricks'n Bricks game in python."
        )

    parser.add_argument(
            "-p", "--path", type=str, default=None,
            help="BnB game screenshot."
        )

    parser.add_argument(
            "-r", "--resume", action="store_true",
            )
    parser.add_argument(
            "-hi", "--history", type=str, default=None,
            )

    args = parser.parse_args()
    assert args.path is not None
    if args.resume:
        assert args.history is not None, "Please provide the history file to resume the game."

    gamename = os.path.basename(args.path).split(".")[0]

    img, img_gray = load_image(args.path)
    icons = segment_objects(img, img_gray)
    clusters, labels = template_matching(icons)
    board = np.array(labels)
    cluster_icon_imgs = {label: icons[idx[0]] for label, idx in clusters.items()}

    cache_path = f"./.cache/{gamename}"
    os.makedirs(cache_path, exist_ok=True)

    import pickle
    with open(f"{cache_path}/board.pkl", "wb") as f:
        pickle.dump(board, f)
    with open(f"{cache_path}/clusters.pkl", "wb") as f:
        pickle.dump(clusters, f)

    game = BricksNBricks(board)
    start = time.time()
    if args.resume:
        game.resume(args.history)
    game.play(verbose=1, save_path=cache_path, gamename=gamename)
    end = time.time()
    print(end-start)
