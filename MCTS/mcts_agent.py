import math
import random
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override

    try:
        from typing_extensions import override
    except ImportError:
        def override(func):
            return func

def fast_rollout(board: np.ndarray, player: int) -> int:
    b = board.copy()
    ROWS, COLS = 6, 7
    p = player
    
    while True:
        # find free columns
        free_cols = [c for c in range(COLS) if b[0, c] == 0]
        if not free_cols:
            return 0 # Draw
        
        move = random.choice(free_cols)
        
        # Apply move
        for r in range(ROWS - 1, -1, -1):
            if b[r, move] == 0:
                b[r, move] = p
                row = r
                break
                
        # Check winner for this move
        # Horizontal
        count = 1
        c = move - 1
        while c >= 0 and b[row, c] == p: count += 1; c -= 1
        c = move + 1
        while c < COLS and b[row, c] == p: count += 1; c += 1
        if count >= 4: return p
        
        # Vertical
        count = 1
        r = row + 1
        while r < ROWS and b[r, move] == p: count += 1; r += 1
        if count >= 4: return p
        
        # Diagonal 1
        count = 1
        r, c = row - 1, move - 1
        while r >= 0 and c >= 0 and b[r, c] == p: count += 1; r -= 1; c -= 1
        r, c = row + 1, move + 1
        while r < ROWS and c < COLS and b[r, c] == p: count += 1; r += 1; c += 1
        if count >= 4: return p
        
        # Diagonal 2
        count = 1
        r, c = row - 1, move + 1
        while r >= 0 and c < COLS and b[r, c] == p: count += 1; r -= 1; c += 1
        r, c = row + 1, move - 1
        while r < ROWS and c >= 0 and b[r, c] == p: count += 1; r += 1; c -= 1
        if count >= 4: return p
        
        p = -p

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = state.get_free_cols()
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.is_final()

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.transition(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                q_hat = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = q_hat + exploration
            choices_weights.append(weight)
        
        max_weight = max(choices_weights)
        best_children = [self.children[i] for i, w in enumerate(choices_weights) if w == max_weight]
        return random.choice(best_children)

class MCTSAgent(Policy):
    def __init__(self, iterations=1000, c_param=math.sqrt(2)):
        self.iterations = iterations
        self.c_param = c_param

    @override
    def mount(self) -> None:
        pass
        
    def get_action(self, state: ConnectState) -> int:
        root = MCTSNode(state)

        for _ in range(self.iterations):
            node = root
            
            # 1. Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c_param)
                
            # 2. Expansion
            if not node.is_terminal():
                node = node.expand()
                
            # 3. Simulation
            if node.is_terminal():
                winner = node.state.get_winner()
            else:
                winner = fast_rollout(node.state.board, node.state.player)
            
            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                if winner != 0:
                    player_who_just_moved = -node.state.player
                    if winner == player_who_just_moved:
                        node.value += 1.0
                    else:
                        node.value -= 1.0
                node = node.parent

        # Choose action with highest visit counts (more robust than max q-value)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    @override
    def act(self, s: np.ndarray) -> int:
        player = -1 if np.sum(s) == 0 else 1
        state = ConnectState(s, player)
        return self.get_action(state)
