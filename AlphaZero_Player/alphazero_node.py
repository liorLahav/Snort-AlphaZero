import math
from typing import List


class AlphaZeroNode:
    def __init__(self, state, root_player, action=None, parent=None, prior=0.0):
        self.root_player = root_player
        self.state = state
        self.action = action
        self.parent = parent

        self.prior = prior

        self.q = 0
        self.n = 0
        self.children: List[AlphaZeroNode] = []

    def child_uct_score(self, child,c= 1):
        if self.state.player == self.root_player:
            exploitation = child.q
        else:
            exploitation = -child.q
        exploration = c * child.prior * math.sqrt(self.n) / (1 + child.n)

        return exploitation + exploration

    def best_child(self, c=1.0):
        return max(self.children, key=lambda ch: self.child_uct_score(ch, c))

    def get_valid_actions(self):
        return self.state.legal_moves()

    def is_terminal(self):
        return self.state.game_state != self.state.ONGOING

    def player(self):
        return self.state.player