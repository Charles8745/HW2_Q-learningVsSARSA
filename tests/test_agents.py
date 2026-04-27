import pytest
import numpy as np
from agents import BaseAgent, QLearningAgent, SarsaAgent

def test_base_agent_initialization():
    agent = BaseAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12))
    assert agent.q_table.shape == (4, 12, 4)
    assert np.all(agent.q_table == 0)
    assert agent.alpha == 0.5
    assert agent.gamma == 1.0
    assert agent.epsilon == 0.1

def test_epsilon_greedy():
    agent = BaseAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12), epsilon=1.0)
    # Epsilon = 1.0 means always random
    actions = [agent.choose_action((0, 0)) for _ in range(100)]
    assert len(set(actions)) > 1 # Should pick multiple actions

    agent.epsilon = 0.0
    agent.q_table[0, 0, 1] = 10.0 # Make action 1 the best
    action = agent.choose_action((0, 0))
    assert action == 1 # Epsilon = 0.0 means always greedy

def test_qlearning_update():
    agent = QLearningAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12), alpha=0.5, gamma=1.0)
    state = (0, 0)
    action = 1
    reward = -1
    next_state = (1, 0)
    
    # next_state max Q is 10 (at action 2)
    agent.q_table[1, 0, 2] = 10.0
    
    # Q(S,A) <- Q(S,A) + alpha * [R + gamma * max Q(S',a') - Q(S,A)]
    # Q(0,0,1) <- 0 + 0.5 * [-1 + 1.0 * 10 - 0] = 0.5 * 9 = 4.5
    agent.update(state, action, reward, next_state, False)
    assert agent.q_table[0, 0, 1] == 4.5

def test_qlearning_terminal_update():
    agent = QLearningAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12), alpha=0.5, gamma=1.0)
    state = (0, 0)
    action = 1
    reward = -100
    next_state = (3, 0)
    
    # If done, max Q(S',a') is 0
    agent.q_table[3, 0, 2] = 10.0 # This should be ignored
    
    # Q(0,0,1) <- 0 + 0.5 * [-100 + 0 - 0] = -50.0
    agent.update(state, action, reward, next_state, True)
    assert agent.q_table[0, 0, 1] == -50.0

def test_sarsa_update():
    agent = SarsaAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12), alpha=0.5, gamma=1.0)
    state = (0, 0)
    action = 1
    reward = -1
    next_state = (1, 0)
    next_action = 2
    
    # next_state action 2 Q is 5.0
    agent.q_table[1, 0, 2] = 5.0
    agent.q_table[1, 0, 3] = 10.0 # SARSA should ignore this max value
    
    # Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
    # Q(0,0,1) <- 0 + 0.5 * [-1 + 1.0 * 5.0 - 0] = 0.5 * 4 = 2.0
    agent.update(state, action, reward, next_state, next_action, False)
    assert agent.q_table[0, 0, 1] == 2.0

def test_sarsa_terminal_update():
    agent = SarsaAgent(action_space=[0, 1, 2, 3], state_shape=(4, 12), alpha=0.5, gamma=1.0)
    state = (0, 0)
    action = 1
    reward = -100
    next_state = (3, 0)
    next_action = 2
    
    agent.q_table[3, 0, 2] = 5.0 # Should be ignored because terminal
    
    # Q(0,0,1) <- 0 + 0.5 * [-100 + 0 - 0] = -50.0
    agent.update(state, action, reward, next_state, next_action, True)
    assert agent.q_table[0, 0, 1] == -50.0
