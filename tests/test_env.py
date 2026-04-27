import pytest
from cliff_walking import CliffWalkingEnv

def test_env_initialization():
    env = CliffWalkingEnv()
    assert env.shape == (4, 12)
    assert env.start_state == (3, 0)
    assert env.goal_state == (3, 11)
    assert env.current_state == (3, 0)

def test_env_reset():
    env = CliffWalkingEnv()
    env.current_state = (1, 1)
    state = env.reset()
    assert state == (3, 0)
    assert env.current_state == (3, 0)

def test_valid_moves():
    env = CliffWalkingEnv()
    env.current_state = (2, 0)
    
    # Move Up
    state, reward, done = env.step(0) # 0: Up
    assert state == (1, 0)
    assert reward == -1
    assert not done
    
    # Move Right
    state, reward, done = env.step(3) # 3: Right
    assert state == (1, 1)
    assert reward == -1
    
    # Move Down
    state, reward, done = env.step(1) # 1: Down
    assert state == (2, 1)
    
    # Move Left
    state, reward, done = env.step(2) # 2: Left
    assert state == (2, 0)

def test_wall_collision():
    env = CliffWalkingEnv()
    # At start state (3, 0), move Left -> should stay at (3, 0)
    state, reward, done = env.step(2) # Left
    assert state == (3, 0)
    assert reward == -1
    assert not done
    
    # Move Down -> should stay at (3, 0)
    state, reward, done = env.step(1) # Down
    assert state == (3, 0)

def test_cliff():
    env = CliffWalkingEnv()
    env.current_state = (2, 1)
    # Move down to cliff at (3, 1)
    state, reward, done = env.step(1) # Down
    assert state == (3, 0) # Reset to start
    assert reward == -100
    assert not done

def test_goal():
    env = CliffWalkingEnv()
    env.current_state = (2, 11)
    # Move down to goal at (3, 11)
    state, reward, done = env.step(1) # Down
    assert state == (3, 11)
    assert reward == -1
    assert done
