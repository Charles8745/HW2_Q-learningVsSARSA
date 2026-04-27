import numpy as np

class BaseAgent:
    def __init__(self, action_space, state_shape, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table to 0
        self.q_table = np.zeros(state_shape + (len(action_space),))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.choice(self.action_space)
        else:
            # Exploit
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.q_table[next_state])
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

class SarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.q_table[state][action]
        if done:
            next_q = 0.0
        else:
            next_q = self.q_table[next_state][next_action]
            
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q
