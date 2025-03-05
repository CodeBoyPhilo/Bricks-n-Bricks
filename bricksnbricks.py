import json
import os
import re
import argparse

import numpy as np
import pandas as pd

from utils import load_image, segment_objects, template_matching, EMOJI_ICONS


class Board:
    def __init__(self, board, cluster):
        self.board = board
        self.m, self.n = board.shape
        self.clusters = dict(
            sorted(cluster.items(), key=lambda item: len(item[1]), reverse=True)
        )
        self.DIRECTIONS_DICT = {
                "up":[-1, 0], 
                "down":[1, 0], 
                "left":[0, -1], 
                "right":[0, 1]
                }

        self.DIRECTIONS_ARRAY = [np.array(d) for d in self.DIRECTIONS_DICT.values()]
        self.direction = None
        self.direction_idx = None
        self.src = None
        self.src_idx = None
        self.action = None
        self.dst = None
        self.dst_idx = None
        self.cur_label = None
        self.src_history= []
        self.action_history = []

    def show_board(self):

        print("   ", end="")
        for col in range(self.board.shape[1]):
            print(f"{col:2}", end="")
        print()

        for i, row in enumerate(self.board):
            # Print the left axis (row numbers)
            print(f" {i:2} ", end="")

            for value in row:
                if value == 0:
                    print("  ", end="")
                else:
                    icon = EMOJI_ICONS[value]
                    print(f"{icon}", end="")

            print()  

    def play(self, save_path=None, gamename=None):
        """
        Example usage of explicit input validation without try/except.
        """
        while True:
            os.system("cls" if os.name == "nt" else "clear")

            if self.src is not None and self.action is not None:
                self._process_move()
            else:
                print("Welcome to Bricks'n Bricks!")
                print("Enter 'e' at any prompt to exit the game.\n")

            self.show_board()

            if np.all(self.board == 0):
                print("Wooo~ You win!")
                break

            user_src = input("row, column coordinate (e.g. 3,1): ")
            valid_src, src_val = self._parse_src(user_src)
            if valid_src == "exit":
                break
            if not valid_src:
                continue 

            user_action = input("action (e.g. 'down 7'): ")
            valid_action, action_val = self._parse_action(user_action, self.DIRECTIONS_DICT)
            if valid_action == "exit":
                break
            if not valid_action:
                print("Invalid action format or constraints. Please try again.\n")
                continue

            self.src = src_val
            self.action = action_val
            self.cur_label = self.board[*self.src]

            self.dst = self.src + self.action
            self.src_idx = self.src[0] * self.n + self.src[1]
            self.dst_idx = self.dst[0] * self.n + self.dst[1]
            self._update_direction()

        assert save_path is not None 
        assert gamename is not None
        history_to_save = pd.DataFrame({"src": self.src_history, "action": self.action_history})
        history_to_save['src'] = history_to_save['src'].apply(lambda x: json.dumps(x.tolist()))
        history_to_save['action'] = history_to_save['action'].apply(lambda x: json.dumps(x.tolist()))
        history_to_save.to_csv(f"{save_path}/{gamename}_history.csv", index=False)

    def resume(self, history):
        moves = pd.read_csv(history)
        moves['src'] = moves['src'].apply(lambda x: np.array(json.loads(x)))
        moves['action'] = moves['action'].apply(lambda x: np.array(json.loads(x)))
        
        srcs, actions = moves["src"].values, moves["action"].values
        for src, action in zip(srcs, actions):
            os.system("cls" if os.name == "nt" else "clear")  # Clear the console
            if self.src is None:
                pass
            else:
                self._process_move()
            self.show_board()
            if np.all(self.board == 0):
                print("Wooo~ All clear!")
                break
            else:
                self.src = src
                self.action = action
                self.cur_label = self.board[*self.src]
                self.dst = self.src + self.action
                self.src_idx = self.src[0] * self.n + self.src[1]
                self.dst_idx = self.dst[0] * self.n + self.dst[1]
                self._update_direction()
        
    def _parse_src(self,user_src):
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
        if row < 0 or row >= self.m or col < 0 or col >= self.n:
            return (False, None)

        return (True, np.array([row, col]))

    def _parse_action(self,user_action, directions_dict):
        user_action = user_action.strip().lower()
        if user_action == "e":
            return ("exit", None)

        pattern = r'^(up|down|left|right)\s+([0-9]+)$'
        match = re.match(pattern, user_action)
        if not match:
            return (False, None)

        direction_str, distance_str = match.groups()
        distance = int(distance_str)

        # Additional constraints
        if direction_str in ["up", "down"] and not (0 <= distance < 14):
            return (False, None)
        if direction_str in ["left", "right"] and not (0 <= distance < 10):
            return (False, None)

        direction_vec = directions_dict[direction_str]
        action_vec = np.array(direction_vec) * distance
        return (True, action_vec)

    def _update_direction(self):
        axis = np.nonzero(self.action)[0][0]  # 0 for row, 1 for column

        if axis == 0:
            if self.action[axis] > 0:
                self.direction = self.DIRECTIONS_ARRAY[1]  # down
                self.direction_idx = 1
            else:
                self.direction = self.DIRECTIONS_ARRAY[0]  # up
                self.direction_idx = 0
        if axis == 1:
            if self.action[axis] > 0:
                self.direction = self.DIRECTIONS_ARRAY[3]  # right
                self.direction_idx = 3
            else:
                self.direction = self.DIRECTIONS_ARRAY[2]  # left
                self.direction_idx = 2

        return None

    def _process_move(self, verbose=1):
        """
        Check if the move is valid and returns the coordinates of the icons to be removed
        """
        if self.board[*self.dst] == self.cur_label and np.abs(self.action.sum()) == 1:
            if self.board[*self.src] == self.board[*self.dst]:
                if verbose == 1:
                    print("direct removal") 
                self._remove_icons([(self.src, self.src_idx), (self.dst, self.dst_idx)])
                self.board[*self.src] = 0
                self.board[*self.dst] = 0
            else:
                if verbose == 1:
                    print("invalid move")
        else:
            if verbose == 1:
                print("no direct removal")
            neighbours = self._get_all_neighbours()
            immediate_neighbour_coord = self._get_closest_coords(self.src)[
                self.direction_idx
            ]
            if immediate_neighbour_coord is None:
                no_immediate_neighbour = True
                n_blanks = 999
            else:
                no_immediate_neighbour = np.any(
                    np.abs(immediate_neighbour_coord - self.src) > 1
                )
                n_blanks = self._count_blanks()
            steps = np.sum(np.abs(self.action))
            diff = n_blanks - steps

            if (diff >= 0) or (
                no_immediate_neighbour
                and self.board[*self.dst] == self.board[*self.src]
            ):  # has enough room to move, no conflicts occur
                self._apply_change_on_temp_board(neighbours)
                coord_to_remove = self._find_removal(self.dst, temp_board=True)

                if coord_to_remove is not None:

                    if self.board[*self.dst] == self.cur_label and np.any(
                        coord_to_remove != self.dst
                    ):
                        self._remove_icons(
                            [
                                (self.src, self.src_idx),
                                (self.dst, self.dst_idx),
                            ]
                        )
                    else:
                        self._apply_change_on_actual_board(neighbours)
                        coord_to_remove_idx = (
                            coord_to_remove[0] * self.n + coord_to_remove[1]
                        )
                        self._remove_icons(
                            [
                                (self.dst, self.dst_idx),
                                (coord_to_remove, coord_to_remove_idx),
                            ]
                        )
                elif self.board[*self.dst] == self.board[*self.src]:
                    # if np.all(self.src == np.array([3, 6])):
                    #     pdb.set_trace()
                    self._apply_change_on_actual_board(neighbours)
                    self._remove_icons(
                        [
                            (self.dst, self.dst_idx),
                        ]
                    )
                else:
                    self.temp_board = self.board.copy()
                    if verbose == 1:
                        print("invalid move")
            else:
                self.temp_board = self.board.copy()
                if verbose == 1:
                    print("invalid move")

    def _get_all_neighbours(self):
        """
        Get the coordinates of all icons between the src and dst coordinates

        Returns:
        neighbours: list
        the first element is the coord of src, the last element is the coord of dst
        """

        neighbours = [self.src]
        while True:
            if neighbours[-1] is None:
                break
            elif self.board[*neighbours[-1]] == 0:
                break
            elif len(neighbours) > 1 and np.any(
                np.abs(neighbours[-1] - neighbours[-2]) > 1
            ):
                neighbours = neighbours[:-1]
                break
            else:
                neighbours.append(
                    self._get_closest_coords(neighbours[-1])[self.direction_idx]
                )

        if neighbours[-1] is None:
            return neighbours[:-1]
        # if len(neighbours) > 1 and self.board[*self.dst] == 0:
        #     return neighbours[:-1]
        else:
            return neighbours

    def _get_closest_coords(self, coord: np.array):
        """
        Return the coords of the closest non-blank icon (skips the blanks)
        """
        neighbours = [None] * 4
        for i, direction in enumerate(self.DIRECTIONS_ARRAY):
            curr_coord = coord + direction
            x, y = curr_coord
            while x >= 0 and y >= 0:
                try:
                    if self.board[x, y] == 0:
                        curr_coord += direction
                        x, y = curr_coord
                    else:
                        neighbours[i] = np.array([x, y])
                        break
                except Exception:
                    break
        return neighbours

    def _count_blanks(self):
        blanks = 0
        curr_coord = self.src.copy()

        while True:
            curr_coord += self.direction
            hit_boarder = [
                (curr_coord[0] < 0),
                (curr_coord[0] >= self.m),
                (curr_coord[1] < 0),
                (curr_coord[1] >= self.n),
            ]

            if any(hit_boarder):
                break
            elif self.board[tuple(curr_coord)] != 0 and blanks != 0:
                break
            elif self.board[tuple(curr_coord)] == 0:
                blanks += 1
            else:
                pass

        return blanks

    def _find_removal(self, coord, temp_board=False):
        """
        Check whether there exists an icon of the same cluster of the coord given
        """
        board = self.temp_board if temp_board else self.board

        closest_coords = self._get_closest_coords(coord)
        for c_coords in closest_coords:
            if c_coords is not None and board[*c_coords] == self.cur_label:
                return c_coords

        return None
    
    def _remove_icons(self, coords_n_idx):
        for (x, y), original_icon_idx in coords_n_idx:
            self.board[x][y] = 0
            self.clusters[self.cur_label].remove(original_icon_idx)
        self.src_history.append(self.src)
        self.action_history.append(self.action)

    def _apply_change_on_temp_board(self, coords):
        self.temp_board = self.board.copy()
        for coord in coords[::-1]:
            self.temp_board[*(coord + self.action)] = self.board[*coord]
            self.temp_board[*coord] = 0

    def _apply_change_on_actual_board(self, coords):
        for coord in coords[::-1]:
            coord_label = self.board[*coord]
            coord_idx = coord[0] * self.n + coord[1]
            new_coord = coord + self.action
            new_coord_idx = new_coord[0] * self.n + new_coord[1]
            # start moving
            self.board[*new_coord] = coord_label  # move the icon
            self.board[*coord] = 0  # remove the icon from the original position
            # update the cluster
            self.clusters[coord_label].remove(
                coord_idx
            )  # remove the icon from the original cluster
            self.clusters[coord_label].append(
                new_coord_idx
            )  # add the icon to the new cluster


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
            "--history", type=str, default=None,
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
    with open(f"{cache_path}/{gamename}_board.pkl", "wb") as f:
        pickle.dump(board, f)
    with open(f"{cache_path}/{gamename}_clusters.pkl", "wb") as f:
        pickle.dump(clusters, f)

    b = Board(board, clusters)
    if args.resume:
        b.resume(args.history)
    b.play(save_path=cache_path, gamename=gamename)
