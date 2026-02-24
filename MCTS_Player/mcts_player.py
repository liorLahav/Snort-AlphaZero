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

            # אם יש מעט מהלכים, נבדוק את כולם. אם יש הרבה, נבדוק מדגם.
            sampled_moves = random.sample(moves, min(len(moves), 5))

            # בחירת המהלך שמשאיר ליריב הכי פחות אופציות
            best_move = max(sampled_moves, key=lambda m: self.evaluate_snort_move(cur, m))

            cur.make(best_move)

        return 1 if cur.game_state == self.root_player else -1

    def evaluate_snort_move(self, state, move):
        temp_state = state.clone()
        temp_state.make(move)

        # 1. How many moves does the opponent have right now?
        opponent_moves = len(temp_state.legal_moves())

        # 2. How many moves do YOU have waiting for your next turn?
        # (You will need to implement a way to check your own moves.
        # e.g., temp_state.get_moves_for_player(self.root_player))
        my_future_moves = len(temp_state.legal_moves())

        # 3. Add the MCTS noise tie-breaker
        noise = random.uniform(0, 0.1)

        # 4. Maximize the gap between your options and their options
        return (my_future_moves - opponent_moves) + noise

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

            # q is already the average value
            avg_val = child.q
            win_pct = (avg_val + 1) * 50  # Convert [-1,1] to [0,100]

            print(f"{str(move):<4} | {child.n:<10} | {avg_val:>11.4f} | {win_pct:>6.1f}%")
            print("-" * 50)

    def get_neighbors(self, pos):
        # במקום r, c = pos
        # אנחנו ניגשים לערכים מתוך אובייקט ה-Move
        r = pos.y  # או pos.row, תלוי איך הגדרת את ה-class Move
        c = pos.x  # או pos.col

        neighbors = []
        # כאן נשאר אותו דבר, רק שים לב שהגודל (6) מתאים ללוח שלך
        board_size = 6
        if r > 0: neighbors.append((r - 1, c))
        if r < board_size - 1: neighbors.append((r + 1, c))
        if c > 0: neighbors.append((r, c - 1))
        if c < board_size - 1: neighbors.append((r, c + 1))
        return neighbors
