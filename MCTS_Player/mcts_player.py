from .mcts_node import MCTSNode
import random

class MCTSPlayer:
    def __init__(self):
        self.root_player = 0
        self.root_node = None
    def choose_move(self, initial_node, iterations=10000):
        self.root_player = initial_node.player
        head = MCTSNode(initial_node,root_player=self.root_player)
        for i in range(iterations):
            cur = self.selection(head)

            if not cur.is_terminal():
                cur_new_child = self.expansion(cur)
                game_res = self.simulation(cur_new_child)
                cur = cur_new_child
            else:
                if cur.state.game_state == self.root_player:
                    game_res = 1
                elif cur.state.game_state == cur.state.other(self.root_player):
                    game_res = -1
                else:
                    game_res = 0

            self.backpropagation(cur, game_res)

        self.root_node = head
        self.debug_stats(head)
        return max(head.children, key=lambda c: c.n).action

    def selection(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal():
            if len(node.children) < len(node.get_valid_actions()):
                return node
            node = node.best_child()
        return node

    def expansion(self, node):
        legal = node.state.legal_moves()
        expanded = [child.action for child in node.children]

        for move in legal:
            if not any(m.y == move.y and m.x == move.x for m in expanded):
                new_state = node.state.clone()
                new_state.make(move)

                child = MCTSNode(
                    state=new_state,
                    root_player=self.root_player,
                    action=move,
                    parent=node
                )
                node.children.append(child)
                return child
        return node

    def simulation(self, node):
        cur = node.state.clone()
        while cur.game_state == cur.ONGOING:
            moves = cur.legal_moves()
            sampled_moves = random.sample(moves, min(len(moves), 5))
            best_move = max(sampled_moves, key=lambda m: self.evaluate_snort_move(cur, m))
            cur.make(best_move)
        return 1 if cur.game_state == self.root_player else -1

    def evaluate_snort_move(self, state, move):
        temp_state = state.clone()
        temp_state.make(move)
        opponent_moves_count = len(temp_state.legal_moves())
        return 100 - opponent_moves_count

    def backpropagation(self, node: MCTSNode, r):
        while node is not None:
            node.n += 1
            node.q += (r - node.q) / node.n
            node = node.parent

    def debug_stats(self, root_node):
        print(f"\n--- MCTS Stats (Total Simulations: {root_node.n}) ---")
        print(f"{'Col':<4} | {'Visits (n)':<10} | {'Avg Value (q)':<12} | {'Win %':<8}")
        print("-" * 50)

        children = root_node.children
        sorted_children = sorted(children, key=lambda c: c.n, reverse=True)
        for child in sorted_children:
            move = getattr(child, 'action', '?')

            avg_val = child.q
            win_pct = (avg_val + 1) * 50

            print(f"{str(move):<4} | {child.n:<10} | {avg_val:>11.4f} | {win_pct:>6.1f}%")
            print("-" * 50)

