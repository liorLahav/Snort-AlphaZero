import numpy as np
from AlphaZero_Player.alphazero_node import AlphaZeroNode

class AlphaZeroPlayer():
    def __init__(self,nn):
        self.root_player = 0
        self.root_node = None
        self.nn = nn
    def choose_move(self, game_state, iterations=200):
        self.root_player = game_state.player
        self.root_node = AlphaZeroNode(game_state, root_player=self.root_player)

        self.expand_and_evaluate(self.root_node)

        for i in range(iterations):
            node = self.root_node

            node = self.selection(node)

            if node.is_terminal():
                if node.state.game_state == self.root_player:
                    value = 1.0
                elif node.state.game_state == node.state.DRAW:
                    value = 0.0
                else:
                    value = -1.0

            else:
                value = self.expand_and_evaluate(node)

            self.backpropagation(node, value)

        self.debug_stats(self.root_node)

        return max(self.root_node.children, key=lambda c: (c.n,c.q)).action

    def selection(self,node : AlphaZeroNode):
        while not node.is_terminal() and len(node.children) > 0:
            node = node.best_child()
        return node

    def expand_and_evaluate(self, node):
        encoded = node.state.encode()
        board_batch = np.expand_dims(encoded, axis=0)

        policy_logits, value_batch = self.nn.predict(board_batch, verbose=0)

        raw_policy = policy_logits[0]
        value = value_batch[0][0]

        legal_moves = node.state.legal_moves()
        policy_sum = 0
        valid_children = []

        for move in legal_moves:
            idx = move.y *  node.state.BOARD_SIZE + move.x
            prior = raw_policy[idx]

            new_state = node.state.clone()
            new_state.make(move)

            child = AlphaZeroNode(
                state=new_state,
                root_player=self.root_player,
                action=move,
                parent=node,
                prior=prior
            )
            valid_children.append(child)
            policy_sum += prior

        for child in valid_children:
            if policy_sum > 0:
                child.prior /= policy_sum
            else:
                child.prior = 1.0 / len(valid_children)

            node.children.append(child)

        if node.state.player == self.root_player:
            return value
        else:
            return -value

    def backpropagation(self, node: AlphaZeroNode, value):
        while node is not None:
            node.n += 1
            node.q += (value - node.q) / node.n
            node = node.parent

    def debug_stats(self, root_node):
        print(f"\n--- MCTS Stats (Sims: {root_node.n}) ---")
        print(f"{'Move':<8} | {'N':<6} | {'Q (Win%)':<10} | {'Prior':<8}")
        print("-" * 40)

        children = sorted(root_node.children, key=lambda c: c.n, reverse=True)
        for child in children[:10]:
            move = getattr(child, 'action', '?')
            win_pct = (child.q + 1) * 50

            print(f"{str(move):<8} | {child.n:<6} | {win_pct:>7.1f}% | {child.prior:.4f}")
        print("-" * 40)