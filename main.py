import os
import numpy as np

from game import SNORT
from mcts_player import MCTSPlayer


def move_to_index(game, move) -> int:
    return move.y * game.BOARD_SIZE + move.x


def extract_root_N(game, root_node) -> np.ndarray:
    A = game.BOARD_SIZE * game.BOARD_SIZE
    visits = np.zeros((A,), dtype=np.int32)

    for child in getattr(root_node, "children", []):
        a = move_to_index(game, child.action)
        visits[a] = int(child.n)

    return visits


def visits_to_pi(visits: np.ndarray) -> np.ndarray:
    v = visits.astype(np.float32)
    total = float(v.sum())
    if total == 0.0:
        return np.zeros_like(v, dtype=np.float32)
    return v / total


def winner_to_z(winner: int, player_to_move: int, draw_code: int) -> float:
    if winner == draw_code:
        return 0.0
    return 1.0 if winner == player_to_move else -1.0


def log_mcts_step_and_apply_move(game, mcts, iterations=2000) -> dict:
    # Run MCTS and pick a move (also sets mcts.root_node)
    move = mcts.choose_move(game, iterations=iterations)

    if mcts.root_node is None:
        raise RuntimeError("mcts.root_node was not set. choose_move() must set it.")

    # Log BEFORE applying move
    X = game.encode()  # (9,6,6)
    visits = extract_root_N(game, mcts.root_node)
    pi = visits_to_pi(visits)

    step = {
        "X": X,
        "pi": pi,
        "player_to_move": int(game.player),
        "chosen_action": move_to_index(game, move),  # optional debug
    }

    # Apply move
    game.make(move)
    return step


def play_selfplay_episode(mcts, iterations=2000):
    game = SNORT()
    episode = []

    while game.game_state == game.ONGOING:
        episode.append(log_mcts_step_and_apply_move(game, mcts, iterations=iterations))

    winner = int(game.game_state)  # CAT / DOG / DRAW

    for step in episode:
        step["z"] = float(winner_to_z(winner, step["player_to_move"], draw_code=SNORT.DRAW))

    return episode


def episode_to_arrays(episode):
    X = np.stack([s["X"] for s in episode], axis=0).astype(np.float32)    # (T,9,6,6)
    PI = np.stack([s["pi"] for s in episode], axis=0).astype(np.float32)  # (T,36)
    Z = np.array([s["z"] for s in episode], dtype=np.float32)             # (T,)
    return X, PI, Z


def save_npz_shard(path: str, X: np.ndarray, PI: np.ndarray, Z: np.ndarray) -> None:
    if not path.endswith(".npz"):
        raise ValueError("path must end with .npz (example: data/snort_shard_00000.npz)")

    if X.ndim != 4:
        raise ValueError(f"X must be 4D (T,C,N,N). Got shape {X.shape}")
    if PI.ndim != 2:
        raise ValueError(f"PI must be 2D (T,A). Got shape {PI.shape}")
    if Z.ndim != 1:
        raise ValueError(f"Z must be 1D (T,). Got shape {Z.shape}")

    T = X.shape[0]
    if PI.shape[0] != T or Z.shape[0] != T:
        raise ValueError(f"Mismatched lengths: X={T}, PI={PI.shape[0]}, Z={Z.shape[0]}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, PI=PI, Z=Z)


def generate_shards(
    out_dir: str,
    num_games: int,
    shard_size_positions: int = 20000,
    iterations: int = 2000,
    start_shard_index: int = 0,
    print_every_games: int = 50,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    mcts = MCTSPlayer()

    buf_X, buf_PI, buf_Z = [], [], []
    shard_idx = start_shard_index
    total_positions = 0

    def buffered_positions() -> int:
        return int(sum(arr.shape[0] for arr in buf_X))

    def flush():
        nonlocal shard_idx, total_positions
        if not buf_X:
            return

        X = np.concatenate(buf_X, axis=0)
        PI = np.concatenate(buf_PI, axis=0)
        Z = np.concatenate(buf_Z, axis=0)

        shard_path = os.path.join(out_dir, f"snort_shard_{shard_idx:05d}.npz")
        save_npz_shard(shard_path, X, PI, Z)

        total_positions += int(X.shape[0])
        print(f"Saved {shard_path} | positions={X.shape[0]} | total_positions={total_positions}")

        shard_idx += 1
        buf_X.clear()
        buf_PI.clear()
        buf_Z.clear()

    for g in range(1, num_games + 1):
        episode = play_selfplay_episode(mcts, iterations=iterations)
        X, PI, Z = episode_to_arrays(episode)

        buf_X.append(X)
        buf_PI.append(PI)
        buf_Z.append(Z)

        if buffered_positions() >= shard_size_positions:
            flush()

        if print_every_games and (g % print_every_games == 0):
            print(
                f"Progress: games={g}/{num_games}, "
                f"buffered_positions={buffered_positions()}, "
                f"shards_written={shard_idx - start_shard_index}"
            )

    flush()
    print(f"Done. Shards written={shard_idx - start_shard_index}, total_positions={total_positions}")

import os
import numpy as np

def validate_snort_npz(path: str, strict: bool = True) -> dict:
    """
    Validates a single SNORT shard .npz file.

    Expected:
      - keys: X, PI, Z
      - X shape:  (T, 9, 6, 6) float32/float16/float64
      - PI shape: (T, 36)
      - Z shape:  (T,)
      - PI rows sum ~ 1 (within tolerance)
      - Z values in {-1, 0, 1} (if strict)
      - X values are finite; board planes are in {0,1} if strict

    Returns a summary dict (counts, ranges, issues).
    Raises ValueError on hard failures.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with np.load(path) as data:
        files = set(data.files)
        required = {"X", "PI", "Z"}
        missing = required - files
        if missing:
            raise ValueError(f"{path}: missing arrays: {sorted(missing)} (found {sorted(files)})")

        X = data["X"]
        PI = data["PI"]
        Z = data["Z"]

    # Basic shape checks
    if X.ndim != 4:
        raise ValueError(f"{path}: X must be 4D (T,C,H,W). Got {X.shape}")
    T, C, H, W = X.shape
    if (C, H, W) != (9, 6, 6):
        raise ValueError(f"{path}: X must have shape (T,9,6,6). Got {X.shape}")

    if PI.shape != (T, 36):
        raise ValueError(f"{path}: PI must have shape (T,36). Got {PI.shape}, expected ({T},36)")
    if Z.shape != (T,):
        raise ValueError(f"{path}: Z must have shape (T,). Got {Z.shape}, expected ({T},)")

    # Finite checks
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains NaN/Inf")
    if not np.isfinite(PI).all():
        raise ValueError(f"{path}: PI contains NaN/Inf")
    if not np.isfinite(Z).all():
        raise ValueError(f"{path}: Z contains NaN/Inf")

    # Policy checks
    row_sums = PI.sum(axis=1)
    if strict:
        if (PI < -1e-6).any():
            raise ValueError(f"{path}: PI contains negative probabilities")
    if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=0):
        # Allow some tolerance for floating error; adjust if needed
        bad = np.where(np.abs(row_sums - 1.0) > 1e-3)[0][:10]
        raise ValueError(f"{path}: PI rows not normalized. Example bad rows: {bad.tolist()}")

    # Value target checks
    if strict:
        allowed = {-1.0, 0.0, 1.0}
        uniq = set(np.unique(Z).astype(float).tolist())
        if not uniq.issubset(allowed):
            raise ValueError(f"{path}: Z has invalid values {sorted(uniq)}; expected subset of {-1,0,1}")

    # Encoding checks (strict one-hot on board planes)
    # Planes 0..6 are one-hot board representation
    board = X[:, 0:7, :, :]
    if strict:
        # Values close to 0/1
        if not np.all((board >= -1e-6) & (board <= 1.0 + 1e-6)):
            raise ValueError(f"{path}: board planes contain values outside [0,1]")

        # One-hot: sum over 7 board planes should be exactly 1 everywhere
        sums = board.sum(axis=1)  # (T,6,6)
        if not np.allclose(sums, 1.0, atol=1e-3, rtol=0):
            idx = np.where(np.abs(sums - 1.0) > 1e-3)
            # idx gives (t,y,x)
            example = (int(idx[0][0]), int(idx[1][0]), int(idx[2][0]))
            raise ValueError(f"{path}: board planes not one-hot at example (t,y,x)={example}")

        # Game state and turn planes are constant across the 6x6
        gs = X[:, 7, :, :]
        tr = X[:, 8, :, :]

        if not np.allclose(gs, gs[:, :1, :1], atol=1e-6, rtol=0):
            raise ValueError(f"{path}: game_state plane not constant within some samples")
        if not np.allclose(tr, tr[:, :1, :1], atol=1e-6, rtol=0):
            raise ValueError(f"{path}: turn plane not constant within some samples")

        # Turn plane should be near 0 or 1
        turn_vals = tr[:, 0, 0]
        if not np.all((turn_vals >= -1e-3) & (turn_vals <= 1.0 + 1e-3)):
            raise ValueError(f"{path}: turn values out of [0,1] range")
        if not np.all((np.abs(turn_vals - 0.0) < 1e-3) | (np.abs(turn_vals - 1.0) < 1e-3)):
            raise ValueError(f"{path}: turn values not binary-like (0/1)")

    summary = {
        "path": path,
        "T": int(T),
        "X_dtype": str(X.dtype),
        "PI_dtype": str(PI.dtype),
        "Z_dtype": str(Z.dtype),
        "PI_row_sum_min": float(row_sums.min()),
        "PI_row_sum_max": float(row_sums.max()),
        "Z_unique": np.unique(Z).tolist(),
        "game_state_unique": np.unique(X[:, 7, 0, 0]).tolist(),
        "turn_unique": np.unique(X[:, 8, 0, 0]).tolist(),
    }
    return summary


def validate_snort_dataset(shards_dir: str, pattern: str = "snort_shard_*.npz", strict: bool = True) -> list[dict]:
    """
    Validates all shards in a directory. Returns list of summaries.
    Raises ValueError on first failing shard.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(shards_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No shards found in {shards_dir} matching {pattern}")

    summaries = []
    for p in paths:
        summaries.append(validate_snort_npz(p, strict=strict))
    return summaries

def main():
    # Example: 10k games, ~20k positions per shard
    generate_shards(
        out_dir="data_snort",
        num_games=3,
        shard_size_positions=20000,
        iterations=10000,
        start_shard_index=0,
        print_every_games=10000,
    )


if __name__ == "__main__":
    main()
