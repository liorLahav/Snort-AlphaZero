import random

import numpy as np

class SNORT:
    class Move:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        # Added repr for easier debugging
        def __repr__(self):
            return f"Move({self.y}, {self.x})"
    # Constants
    CAT = 1
    CAT_AREA = 3
    DOG = 2
    DOG_AREA = 4
    COMMON_AREA = 5
    BLOCK = 6
    EMPTY = 0

    BOARD_SIZE=6

    CAT_WIN = 1
    DOG_WIN = 2
    DRAW = 0
    ONGOING = 3

    def __init__(self):
        self.board = [[0 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.player = self.CAT
        self.game_state = self.ONGOING

        # Place 5 random blocks
        for i in range(3):
            while True:
                x = random.randint(0, self.BOARD_SIZE -1)
                y = random.randint(0, self.BOARD_SIZE - 1)
                if self.board[y][x] == self.BLOCK:
                    continue
                self.board[y][x] = self.BLOCK
                break

    def legal_moves(self):
        area = self.get_area_type()
        return [
            SNORT.Move(x, y)
            for y in range(self.BOARD_SIZE)
            for x in range(self.BOARD_SIZE)
            if self.board[y][x] == self.EMPTY or self.board[y][x] == area
        ]

    def get_area_type(self):
        if self.player == self.CAT:
            return self.CAT_AREA
        elif self.player == self.DOG:
            return self.DOG_AREA
        return 0

    def make(self, move: "SNORT.Move"):
        undo = {
            "prev_player": self.player,
            "prev_game_state": self.game_state,
            "cell_changes": []
        }

        y0, x0 = move.y, move.x
        undo["cell_changes"].append((y0, x0, self.board[y0][x0]))
        self.board[y0][x0] = self.player

        cur_area = self.get_area_type()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dy, dx in dirs:
            y, x = y0 + dy, x0 + dx
            if not self.in_board(y, x):
                continue

            v = self.board[y][x]

            # Skip blocks or actual pieces
            if v == self.BLOCK or v == self.CAT or v == self.DOG:
                continue

            # --- LOGIC FIX START ---
            new_val = v

            # If empty, it takes on the current player's influence
            if v == self.EMPTY:
                new_val = cur_area

            # If it's the opponent's area, it becomes common (dead zone)
            # If it's already my area or common area, it doesn't change
            elif v != cur_area and v != self.COMMON_AREA:
                new_val = self.COMMON_AREA

            if new_val != v:
                undo["cell_changes"].append((y, x, v))
                self.board[y][x] = new_val
            # --- LOGIC FIX END ---

        mover = self.player
        self.player = self.other(self.player)

        # Terminal check
        if not self.legal_moves():

            self.game_state = mover
        else:
            self.game_state = self.ONGOING
        return undo

    def unmake(self, undo):
        for y, x, old in reversed(undo["cell_changes"]):
            self.board[y][x] = old
        self.player = undo["prev_player"]
        self.game_state = undo["prev_game_state"]

    import numpy as np

    def encode(self) -> np.ndarray:
        """
        Fully-decodable (invertible) encoding for SNORT.

        Planes (C=9), each is N x N:
          0: EMPTY
          1: CAT
          2: DOG
          3: CAT_AREA
          4: DOG_AREA
          5: COMMON_AREA
          6: BLOCK
          7: game_state (constant plane)
          8: turn (constant plane: 1.0 if CAT to move else 0.0)

        Returns: np.ndarray shape (9, N, N) float32
        """
        N = self.BOARD_SIZE
        b = np.array(self.board, dtype=np.int8)

        planes = [
            (b == self.EMPTY).astype(np.float32),
            (b == self.CAT).astype(np.float32),
            (b == self.DOG).astype(np.float32),
            (b == self.CAT_AREA).astype(np.float32),
            (b == self.DOG_AREA).astype(np.float32),
            (b == self.COMMON_AREA).astype(np.float32),
            (b == self.BLOCK).astype(np.float32),
            np.full((N, N), float(self.game_state), dtype=np.float32),
            np.full((N, N), 1.0 if self.player == self.CAT else 0.0, dtype=np.float32),
        ]

        return np.stack(planes, axis=0)

    def decode(self, X: np.ndarray) -> None:
        """
        Decodes the encoding produced by encode() back into:
          - self.board (including areas)
          - self.game_state
          - self.player
        """
        N = self.BOARD_SIZE

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray")
        if X.shape != (9, N, N):
            raise ValueError(f"Bad shape: got {X.shape}, expected {(9, N, N)}")

        board_planes = X[0:7]  # (7, N, N)
        board = [[self.EMPTY for _ in range(N)] for _ in range(N)]

        idx_to_val = [
            self.EMPTY,
            self.CAT,
            self.DOG,
            self.CAT_AREA,
            self.DOG_AREA,
            self.COMMON_AREA,
            self.BLOCK,
        ]

        for y in range(N):
            for x in range(N):
                k = int(np.argmax(board_planes[:, y, x]))
                board[y][x] = idx_to_val[k]

        # game_state and player are stored as constants; read one cell
        self.game_state = int(round(float(X[7, 0, 0])))
        self.player = self.CAT if float(X[8, 0, 0]) >= 0.5 else self.DOG

        self.board = board

    def in_board(self, y, x):
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE

    def other(self, player: int) -> int:
        return self.DOG if player == self.CAT else self.CAT

    def clone(self):
        s = SNORT()
        s.player = self.player
        s.game_state = self.game_state
        s.board = [row[:] for row in self.board]
        return s

    # Optimized to avoid cloning
    def winning_move(self, move: "SNORT.Move") -> bool:
        undo = self.make(move)
        is_win = (self.game_state != self.ONGOING)  # If game ended, current mover (prev_player) won
        self.unmake(undo)
        return is_win

    def __str__(self):
        # (Same as your original code)
        def cell_to_char(v: int) -> str:
            if v == self.EMPTY: return "."
            if v == self.CAT: return "C"
            if v == self.DOG: return "D"
            if v == self.BLOCK: return "#"
            if v == self.CAT_AREA: return "c"  # Lowercase for area
            if v == self.DOG_AREA: return "d"  # Lowercase for area
            if v == self.COMMON_AREA: return "*"
            return "?"

        turn = "CAT" if self.player == self.CAT else "DOG"
        lines = [f"Turn: {turn}", "   " + " ".join(f"{x:2d}" for x in range(self.BOARD_SIZE)), "   " + "---" * self.BOARD_SIZE]
        for y in range(self.BOARD_SIZE):
            row = " ".join(f" {cell_to_char(self.board[y][x])}" for x in range(self.BOARD_SIZE))
            lines.append(f"{y:2d}|{row}")
        return "\n".join(lines)
