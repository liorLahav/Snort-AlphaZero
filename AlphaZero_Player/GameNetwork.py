import tensorflow as tf
from keras.src.layers import BatchNormalization, Activation, Add
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
from pathlib import Path
from tensorflow.keras.regularizers import l2

class GameNetwork:
    def __init__(self):
        self.data_path = r"C:\Users\liorl\OneDrive\Desktop\data_snort"
        self.nn: Model = self.build_model(num_actions=36)

    def build_model(
            self,
            num_actions: int
    ) -> Model:
        def res_block(x, filters=64):
            shortcut = x
            x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            out = Add()([shortcut, x])  # The "Skip Connection"
            return Activation("relu")(out)

        t_in = Input(shape=(6, 6, 6))

        # Initial Convolution
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4))(t_in)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Stack of Residual Blocks (The "Brain")
        # 4 blocks is a good balance for a 6x6 board
        for _ in range(4):
            x = res_block(x, filters=64)

        t_out = x
        # We wrap it in 'Model' so it behaves just like your 'Sequential' variable
        t = Model(inputs=t_in, outputs=t_out, name="t")

        inp = Input(shape=(6, 6, 6), name="board")
        x = t(inp)

        # Policy Head (p)
        # We add a small convolution first to boil down features, then Flatten
        px = Conv2D(2, (1, 1), padding="same", kernel_regularizer=l2(1e-4))(x)
        px = BatchNormalization()(px)
        px = Activation("relu")(px)
        px = Flatten()(px)
        p = Dense(num_actions, activation="softmax", name="policy")(px)

        # Value Head (v)
        vx = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(1e-4))(x)
        vx = BatchNormalization()(vx)
        vx = Activation("relu")(vx)
        vx = Flatten()(vx)
        vx = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(vx)
        v = Dense(1, activation="tanh", name="value")(vx)

        return Model(inputs=inp, outputs=[p, v], name="model")

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        files = sorted(Path(self.data_path).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in: {self.data_path}")

        X_list = []
        PI_list = []
        Z_list = []

        num_actions = None

        print(f"Loading {len(files)} shards...")

        for fp in files:
            d = np.load(fp)

            X = d["X"].astype(np.float32)
            PI = d["PI"].astype(np.float32)
            Z = d["Z"].astype(np.float32)

            if num_actions is None:
                num_actions = PI.shape[1]
            elif PI.shape[1] != num_actions:
                d.close()
                raise ValueError(f"{fp.name}: PI has {PI.shape[1]} actions, expected {num_actions}")

            if X.ndim == 4 and X.shape[1] == 6 and X.shape[-1] != 6:
                X = np.transpose(X, (0, 2, 3, 1))

            Z = Z.reshape(-1, 1)

            X_list.append(X)
            PI_list.append(PI)
            Z_list.append(Z)

            d.close()

        X_all = np.concatenate(X_list, axis=0)
        PI_all = np.concatenate(PI_list, axis=0)
        Z_all = np.concatenate(Z_list, axis=0)

        print(f"Original dataset size: {X_all.shape[0]}")

        print("Applying symmetries (x8 expansion)...")

        X_aug_list = []
        PI_aug_list = []
        Z_aug_list = []

        pi_sq = PI_all.reshape(-1, 6, 6)

        for k in range(4):
            x_rot = np.rot90(X_all, k=k, axes=(1, 2))
            pi_rot = np.rot90(pi_sq, k=k, axes=(1, 2))

            X_aug_list.append(x_rot)
            PI_aug_list.append(pi_rot.reshape(-1, 36))
            Z_aug_list.append(Z_all)

            x_flip = np.flip(x_rot, axis=2)
            pi_flip = np.flip(pi_rot, axis=2)

            X_aug_list.append(x_flip)
            PI_aug_list.append(pi_flip.reshape(-1, 36))
            Z_aug_list.append(Z_all)

        X_final = np.concatenate(X_aug_list, axis=0)
        PI_final = np.concatenate(PI_aug_list, axis=0)
        Z_final = np.concatenate(Z_aug_list, axis=0)

        print("Shuffling augmented dataset...")
        indices = np.arange(X_final.shape[0])
        np.random.shuffle(indices)

        X_final = X_final[indices]
        PI_final = PI_final[indices]
        Z_final = Z_final[indices]

        assert num_actions is not None
        print("Final Loaded Data:")
        print("X:", X_final.shape, "PI:", PI_final.shape, "Z:", Z_final.shape, "num_actions:", num_actions)

        return X_final, PI_final, Z_final, num_actions

    def train_model(self):
        X, PI, Z, num_actions = self.load_data()

        if self.nn.output[0].shape[-1] != num_actions:
            self.nn = self.build_model(num_actions=num_actions)

        self.nn.compile(
            optimizer=Adam(1e-3),
            loss={"policy": CategoricalCrossentropy(), "value": MeanSquaredError()},
        )

        history = self.nn.fit(
            X,
            {"policy": PI, "value": Z},
            batch_size=128,
            epochs=30,
            validation_split=0.1,
            shuffle=True,
        )
        return history
    def save_model(self):
        self.nn.save("init_rr_nn.keras")

if __name__ == "__main__":
    game = GameNetwork()
    history = game.train_model()
    game.save_model()
