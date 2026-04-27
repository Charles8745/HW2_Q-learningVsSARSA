class CliffWalkingEnv:
    def __init__(self):
        self.shape = (4, 12)
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.current_state = self.start_state
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = [0, 1, 2, 3]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        y, x = self.current_state
        
        if action == 0:   # Up
            y = max(0, y - 1)
        elif action == 1: # Down
            y = min(self.shape[0] - 1, y + 1)
        elif action == 2: # Left
            x = max(0, x - 1)
        elif action == 3: # Right
            x = min(self.shape[1] - 1, x + 1)
            
        self.current_state = (y, x)
        
        # Check cliff
        if y == 3 and 1 <= x <= 10:
            self.current_state = self.start_state
            return self.current_state, -100, False
            
        # Check goal
        if self.current_state == self.goal_state:
            return self.current_state, -1, True
            
        # Normal step
        return self.current_state, -1, False
