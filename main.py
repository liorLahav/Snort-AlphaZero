import numpy as np
import sys
import os
from MCTS_Player.mcts_player import MCTSPlayer
from game import SNORT
from tensorflow.keras.models import load_model

from AlphaZero_Player.alphazero_player import AlphaZeroPlayer


def move_to_index(game, move) -> int:
    return move.y * game.BOARD_SIZE + move.x


def extract_root_N(game, root_node) -> np.ndarray:
    A = game.BOARD_SIZE * game.BOARD_SIZE
    visits = np.zeros((A,), dtype=np.int32)

    for child in root_node.children:
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

    if winner == player_to_move:
        return 1.0
    else:
        return -1.0


def log_mcts_step_and_apply_move(game, mcts, iterations=2000) -> dict:
    move = mcts.choose_move(game, iterations=iterations)

    if mcts.root_node is None:
        raise RuntimeError("mcts.root_node was not set. choose_move() must set it.")

    X = game.encode()
    visits = extract_root_N(game, mcts.root_node)
    pi = visits_to_pi(visits)

    step = {
        "X": X,
        "pi": pi,
        "player_to_move": int(game.player),
    }

    game.make(move)
    return step

def log_alphazero_step_and_apply_move(game, alphazero, iterations=300) -> dict:
    move = alphazero.choose_move(game, iterations=iterations)

    if alphazero.root_node is None:
        raise RuntimeError("mcts.root_node was not set. choose_move() must set it.")

    X = game.encode()
    visits = extract_root_N(game, alphazero.root_node)
    pi = visits_to_pi(visits)

    step = {
        "X": X,
        "pi": pi,
        "player_to_move": int(game.player),
    }

    game.make(move)
    return step


def play_selfplay_episode(mcts, iterations=2000):
    game = SNORT()
    episode = []

    while game.game_state == game.ONGOING:
        episode.append(log_mcts_step_and_apply_move(game, mcts, iterations=iterations))

    winner = int(game.game_state)

    for step in episode:
        step["z"] = float(winner_to_z(winner, step["player_to_move"], draw_code=SNORT.DRAW))

    return episode, int(game.game_state)


def episode_to_arrays(episode):
    X = np.stack([s["X"] for s in episode], axis=0).astype(np.float32)
    PI = np.stack([s["pi"] for s in episode], axis=0).astype(np.float32)
    Z = np.array([s["z"] for s in episode], dtype=np.float32)
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


def get_human_move(game):
    """
    Prompts the user for a move (Row Col) and validates it.
    """
    legal_moves = game.legal_moves()

    # Create a simpler representation for matching input
    # legal_moves contains Move(x, y) objects
    legal_coords = {(m.y, m.x) for m in legal_moves}

    while True:
        try:
            user_input = input(
                f"\nYour move ({'Cat' if game.player == SNORT.CAT else 'Dog'}) [row col]: ").strip().lower()

            if user_input in ['q', 'quit', 'exit']:
                print("Game aborted.")
                sys.exit()

            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid format. Please enter: row col (e.g., '2 3')")
                continue

            r, c = int(parts[0]), int(parts[1])

            if (r, c) in legal_coords:
                # Return the Move object that matches these coordinates
                for m in legal_moves:
                    if m.y == r and m.x == c:
                        return m
            else:
                print(f"Illegal move. That square is either taken, blocked, or in a dead zone.")
                print(f"Valid moves (row, col): {sorted(list(legal_coords))}")

        except ValueError:
            print("Invalid numbers. Please enter integers.")

def play_vs_ai():
    print("=== SNORT: Human vs MCTS ===")

    # 1. Setup
    game = SNORT()
    ai = AlphaZeroPlayer(load_model('final_nn.keras'))
    ai2 =  MCTSPlayer()

    # 2. Choose Side
    while True:
        choice = input("Do you want to play as Cat (starts) or Dog (goes second)? [c/d]: ").lower()
        if choice in ['c']:
            human_player = SNORT.CAT
            ai_player = SNORT.DOG
            print("\nYou are CAT (Player 1). You start.")
            break
        elif choice in ['d']:
            human_player = SNORT.DOG
            ai_player = SNORT.CAT
            print("\nYou are DOG (Player 2). AI starts.")
            break

    # 3. Game Loop
    while game.game_state == SNORT.ONGOING:
        print("\n" + "=" * 20)
        print(game)  # Uses your __str__ method
        print("=" * 20)

        if game.player == human_player:
            # Human Turn
            #move = get_human_move(game)
            move = ai2.choose_move(game,1000000)
            game.make(move)
        else:
            # AI Turn
            print(f"\nAI ({'Cat' if ai_player == SNORT.CAT else 'Dog'}) is thinking...")
            print("5")
            move = ai.choose_move(game, iterations=300)

            print(f"AI chose: Row {move.y}, Col {move.x}")
            game.make(move)

    # 4. End Game
    print("\n" + "=" * 20)
    print("FINAL BOARD")
    print(game)
    print("=" * 20)

    if game.game_state == SNORT.DRAW:
        print("It's a DRAW!")
    elif game.game_state == human_player:
        print("YOU WON!")
    else:
        print("AI WON!")


def train_on_episodes(
        model_path: str = "./nn.keras",
        num_games: int = 1000,
        iterations: int = 300,
        buffer_size: int = 512,
        batch_size: int = 64,
        save_every: int = 500,
        save_dir: str = "./checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)

    cat_wins = 0
    dog_wins = 0
    draws = 0

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    replay_buffer = []

    winner_names = {
        SNORT.CAT: "Cat",
        SNORT.DOG: "Dog",
        SNORT.DRAW: "Draw"
    }

    last_loss = None

    for i in range(1, num_games + 1):
        player = AlphaZeroPlayer(model)

        episode, winner = play_selfplay_episode(player, iterations=iterations)

        replay_buffer.extend(episode)
        if len(replay_buffer) > buffer_size:
            replay_buffer = replay_buffer[-buffer_size:]

        if winner == SNORT.CAT:
            cat_wins += 1
        elif winner == SNORT.DOG:
            dog_wins += 1
        elif winner == SNORT.DRAW:
            draws += 1

        # Train if possible
        if len(replay_buffer) >= batch_size:
            batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
            X = np.array([replay_buffer[j]['X'] for j in batch])
            PI = np.array([replay_buffer[j]['pi'] for j in batch])
            Z = np.array([replay_buffer[j]['z'] for j in batch])

            loss = model.train_on_batch(
                x=X,
                y={'policy': PI, 'value': Z}
            )
            last_loss = loss[0]

        # Save checkpoint every N games
        if i % save_every == 0:
            save_path = os.path.join(save_dir, f"nn_game_{i}.keras")
            model.save(save_path)
            print(f"[Checkpoint saved] {save_path}")

        print(
            f"Game {i}/{num_games} | "
            f"Winner: {winner_names.get(winner, 'Unknown')} | "
            f"CAT: {cat_wins} | "
            f"DOG: {dog_wins} | "
            f"DRAW: {draws} | "
            f"Loss: {last_loss:.4f}" if last_loss is not None else
            f"Game {i}/{num_games} | "
            f"Winner: {winner_names.get(winner, 'Unknown')} | "
            f"CAT: {cat_wins} | "
            f"DOG: {dog_wins} | "
            f"DRAW: {draws} | "
            f"Loss: N/A"
        )

    # Final save
    final_path = os.path.join(save_dir, "nn_final.keras")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    play_vs_ai()