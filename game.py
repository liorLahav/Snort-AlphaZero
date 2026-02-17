import random
import numpy as np
from dataclasses import make_dataclass


class SNORT:
    class Move:
        def __init__(
                self,
                x: int,
                y: int,
                is_swap: bool = False,
        ):
            self.x = x
            self.y = y
            self.is_swap = is_swap

        def __repr__(
                self,
        ):
            if self.is_swap:
                return f"Move(SWAP@{self.y},{self.x})"
            return f"Move({self.y}, {self.x})"

    # Constants
    CAT = 1
    CAT_AREA = 3
    DOG = 2
    DOG_AREA = 4
    COMMON_AREA = 5
    BLOCK = 6
    EMPTY = 0

    BOARD_SIZE = 6

    CAT_WIN = 1
    DOG_WIN = -1
    DRAW = 0
    ONGOING = 3

    def __init__(
            self,
    ):
        self.board = [[0 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.player = self.CAT
        self.game_state = self.ONGOING
        self.last_move = None

        self.game_counter = 0

        self.swapped = False

        self.first_move = None

        for _ in range(3):
            while True:
                x = random.randint(0, self.BOARD_SIZE - 1)
                y = random.randint(0, self.BOARD_SIZE - 1)
                if self.board[y][x] == self.BLOCK:
                    continue
                self.board[y][x] = self.BLOCK
                break

    def legal_moves(
            self,
    ):
        area = self.get_area_type()
        moves = [
            SNORT.Move(x, y)
            for y in range(self.BOARD_SIZE)
            for x in range(self.BOARD_SIZE)
            if self.board[y][x] == self.EMPTY or self.board[y][x] == area
        ]

        if (
                self.game_counter == 1
                and not self.swapped
                and self.first_move is not None
        ):
            fm = self.first_move
            moves.append(SNORT.Move(fm.x, fm.y, is_swap=True))

        return moves

    def get_area_type(
            self,
    ):
        if self.player == self.CAT:
            return self.CAT_AREA
        if self.player == self.DOG:
            return self.DOG_AREA
        return 0

    def _swap_board(
            self,
            undo,
    ):
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                v = self.board[y][x]
                undo["cell_changes"].append((y, x, v))

                if v == self.CAT:
                    self.board[y][x] = self.DOG
                elif v == self.DOG:
                    self.board[y][x] = self.CAT
                elif v == self.CAT_AREA:
                    self.board[y][x] = self.DOG_AREA
                elif v == self.DOG_AREA:
                    self.board[y][x] = self.CAT_AREA

    def make(
            self,
            move: "SNORT.Move",
    ):
        undo = {
            "prev_player": self.player,
            "prev_game_state": self.game_state,
            "prev_last_move": self.last_move,
            "prev_game_counter": self.game_counter,
            "prev_swapped": self.swapped,
            "prev_first_move": self.first_move,
            "cell_changes": [],
        }

        self.last_move = move

        # ---- SWAP action (Pie rule) implemented as replaying the first cell ----
        if (
                self.game_counter == 1
                and self.first_move is not None
                and move.x == self.first_move.x
                and move.y == self.first_move.y
                and getattr(move, "is_swap", False)
        ):
            self._swap_board(undo)

            self.swapped = True
            self.game_counter += 1

            # turn moves on after swap
            mover = self.player
            self.player = self.other(self.player)

            if not self.legal_moves():
                self.game_state = mover
            else:
                self.game_state = self.ONGOING

            return undo

        # Normal placement
        y0, x0 = move.y, move.x

        undo["cell_changes"].append((y0, x0, self.board[y0][x0]))
        self.board[y0][x0] = self.player

        # record the very first placement
        if self.game_counter == 0:
            self.first_move = SNORT.Move(x0, y0)

        cur_area = self.get_area_type()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dy, dx in dirs:
            y, x = y0 + dy, x0 + dx
            if not self.in_board(y, x):
                continue

            v = self.board[y][x]

            if v == self.BLOCK or v == self.CAT or v == self.DOG:
                continue

            new_val = v
            if v == self.EMPTY:
                new_val = cur_area
            elif v != cur_area and v != self.COMMON_AREA:
                new_val = self.COMMON_AREA

            if new_val != v:
                undo["cell_changes"].append((y, x, v))
                self.board[y][x] = new_val

        mover = self.player
        self.player = self.other(self.player)
        self.game_counter += 1

        if not self.legal_moves():
            self.game_state = mover
        else:
            self.game_state = self.ONGOING

        return undo

    def encode(
            self,
            include_swap_plane: bool = True,
    ) -> np.ndarray:
        N = self.BOARD_SIZE
        b = np.array(self.board, dtype=np.int8)

        me = self.player
        opp = self.other(me)

        p_my_pieces = (b == me).astype(np.float32)
        p_opp_pieces = (b == opp).astype(np.float32)
        p_blocks = (b == self.BLOCK).astype(np.float32)

        my_area_val = self.CAT_AREA if me == self.CAT else self.DOG_AREA
        opp_area_val = self.DOG_AREA if me == self.CAT else self.CAT_AREA

        p_valid_me = ((b == self.EMPTY) | (b == my_area_val)).astype(np.float32)
        p_valid_opp = ((b == self.EMPTY) | (b == opp_area_val)).astype(np.float32)

        # During ply==1, the first occupied cell becomes a legal "swap move" for current player,
        # so we also mark it as valid in the mask plane.
        if (
                self.game_counter == 1
                and not self.swapped
                and self.first_move is not None
        ):
            p_valid_me[self.first_move.y, self.first_move.x] = 1.0

        p_last_move = np.zeros((N, N), dtype=np.float32)
        if self.last_move is not None and not getattr(self.last_move, "is_swap", False):
            if 0 <= self.last_move.y < N and 0 <= self.last_move.x < N:
                p_last_move[self.last_move.y, self.last_move.x] = 1.0

        planes = [
            p_my_pieces,
            p_opp_pieces,
            p_blocks,
            p_valid_me,
            p_valid_opp,
            p_last_move,
        ]

        return np.stack(planes, axis=-1)

    def in_board(
            self,
            y: int,
            x: int,
    ):
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE

    def other(
            self,
            player: int,
    ) -> int:
        return self.DOG if player == self.CAT else self.CAT

    def clone(
            self,
    ):
        s = SNORT()
        s.player = self.player
        s.game_state = self.game_state
        s.board = [row[:] for row in self.board]
        s.last_move = self.last_move
        s.game_counter = self.game_counter
        s.swapped = self.swapped
        s.first_move = self.first_move
        return s

    def __str__(
            self,
    ):
        def cell_to_char(v: int) -> str:
            if v == self.EMPTY:
                return "."
            if v == self.CAT:
                return "C"
            if v == self.DOG:
                return "D"
            if v == self.BLOCK:
                return "#"
            if v == self.CAT_AREA:
                return "c"
            if v == self.DOG_AREA:
                return "d"
            if v == self.COMMON_AREA:
                return "*"
            return "?"

        turn = "CAT" if self.player == self.CAT else "DOG"
        lines = [
            f"Turn: {turn}",
            "   " + " ".join(f"{x:2d}" for x in range(self.BOARD_SIZE)),
            "   " + "---" * self.BOARD_SIZE,
        ]
        for y in range(self.BOARD_SIZE):
            row = " ".join(f" {cell_to_char(self.board[y][x])}" for x in range(self.BOARD_SIZE))
            lines.append(f"{y:2d}|{row}")
        return "\n".join(lines)
