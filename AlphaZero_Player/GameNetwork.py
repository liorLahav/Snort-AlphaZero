from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from pathlib import Path
import os


class GameNetwork:
    def __init__(self, data_path, board_size=8):
        self.data_path = data_path
        self.board_size = board_size
        self.nn: Model = self.build_model(input_shape=(board_size, board_size, 6), num_actions=board_size ** 2)

    def build_model(self, input_shape, num_actions) -> Model:
        def block(x, filters=64):
            shortcut = x
            x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            out = Add()([shortcut, x])
            return Activation("relu")(out)

        inp = Input(shape=input_shape, name="board_input")

        # Initial Convolution
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(4):
            x = block(x, filters=64)

        # Policy Head
        px = Conv2D(2, (1, 1), padding="same", kernel_regularizer=l2(1e-4))(x)
        px = BatchNormalization()(px)
        px = Activation("relu")(px)
        px = Flatten()(px)
        p = Dense(num_actions, activation="softmax", name="policy")(px)

        # Value Head
        vx = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(1e-4))(x)
        vx = BatchNormalization()(vx)
        vx = Activation("relu")(vx)
        vx = Flatten()(vx)
        vx = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(vx)
        v = Dense(1, activation="tanh", name="value")(vx)

        model = Model(inputs=inp, outputs=[p, v], name="snort_resnet")
        return model

    def load_data(self):
        files = sorted(Path(self.data_path).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in: {self.data_path}")

        X_list, PI_list, Z_list = [], [], []

        print(f"Loading {len(files)} shards from {self.data_path}...")

        for fp in files:
            with np.load(fp) as d:
                X = d["X"].astype(np.float32)
                PI = d["PI"].astype(np.float32)
                Z = d["Z"].astype(np.float32)

                if X.ndim == 4:
                    if X.shape[1] == 6 and X.shape[-1] != 6:
                        X = np.transpose(X, (0, 2, 3, 1))

                Z = Z.reshape(-1, 1)
                X_list.append(X)
                PI_list.append(PI)
                Z_list.append(Z)

        X_all = np.concatenate(X_list, axis=0)
        PI_all = np.concatenate(PI_list, axis=0)
        Z_all = np.concatenate(Z_list, axis=0)

        current_board_size = X_all.shape[1]
        num_actions = PI_all.shape[1]

        print(f"Dataset loaded. Board Size: {current_board_size}x{current_board_size}, Actions: {num_actions}")
        print("Applying symmetries (x8 expansion)...")

        X_aug_list = []
        PI_aug_list = []
        Z_aug_list = []

        pi_sq = PI_all.reshape(-1, current_board_size, current_board_size)

        for k in range(4):
            # Rotations
            x_rot = np.rot90(X_all, k=k, axes=(1, 2))
            pi_rot = np.rot90(pi_sq, k=k, axes=(1, 2))

            X_aug_list.append(x_rot)
            PI_aug_list.append(pi_rot.reshape(-1, num_actions))
            Z_aug_list.append(Z_all)

            # Flips (Horizontal flip after rotation)
            x_flip = np.flip(x_rot, axis=2)
            pi_flip = np.flip(pi_rot, axis=2)

            X_aug_list.append(x_flip)
            PI_aug_list.append(pi_flip.reshape(-1, num_actions))
            Z_aug_list.append(Z_all)

        X_final = np.concatenate(X_aug_list, axis=0)
        PI_final = np.concatenate(PI_aug_list, axis=0)
        Z_final = np.concatenate(Z_aug_list, axis=0)

        # Shuffle
        indices = np.arange(X_final.shape[0])
        np.random.shuffle(indices)

        return X_final[indices], PI_final[indices], Z_final[indices], num_actions, X_final.shape[1:]

    def train_model(self, epochs=10, batch_size=128):
        X, PI, Z, num_actions, input_shape = self.load_data()

        if self.nn.input.shape[1:] != input_shape or self.nn.output[0].shape[-1] != num_actions:
            print(f"Rebuilding model for input {input_shape} and actions {num_actions}...")
            self.nn = self.build_model(input_shape, num_actions)

        self.nn.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss={"policy": CategoricalCrossentropy(), "value": MeanSquaredError()},
            metrics={"policy": "accuracy", "value": "mae"}
        )

        print(f"Starting training on {X.shape[0]} samples...")
        history = self.nn.fit(
            X,
            {"policy": PI, "value": Z},
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            shuffle=True,
        )
        return history

    def save_model(self, path="snort_model.keras"):
        self.nn.save(path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\liorl\OneDrive\Desktop\aaa"

    if not os.path.exists(DATA_PATH):
        print(f"Error: Path does not exist: {DATA_PATH}")
    else:
        game = GameNetwork(data_path=DATA_PATH)

        history = game.train_model(epochs=30)
        game.save_model("snort_model_v2.keras")