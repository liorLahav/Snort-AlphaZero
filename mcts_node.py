import math
from typing import List



class MCTSNode:
    def __init__(self, state,root_player, action=None, parent=None):
        self.root_player = root_player
        self.state = state
        self.action = action
        self.parent = parent

        self.q = 0
        self.n = 0
        self.children : List[MCTSNode] = []

    def child_uct_score(self, child, c=1.4):
        if child.n == 0:
            return float("inf")
        if self.n == 0:
            return float("inf")

        exploration = c * math.sqrt(math.log(self.n) / child.n)

        exploitation = child.q if self.state.player == self.root_player else -child.q

        return exploitation + exploration

    def best_child(self, c=1.4):
        return max(self.children, key=lambda ch: self.child_uct_score(ch, c))

    def get_valid_actions(self):
        return self.state.legal_moves()

    def is_terminal(self):
        return self.state.game_state != self.state.ONGOING

    def player(self):
        return self.state.player()
