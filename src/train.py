import numpy as np
from cliff_walking import CliffWalkingEnv
from agents import QLearningAgent, SarsaAgent

def train_agent(agent, env, episodes):
    rewards_per_episode = np.zeros(episodes)
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state, done)
            elif isinstance(agent, SarsaAgent):
                agent.update(state, action, reward, next_state, next_action, done)
                
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards_per_episode[ep] = total_reward
        
    return rewards_per_episode

def run_experiment(runs=50, episodes=500):
    env = CliffWalkingEnv()
    
    q_learning_rewards = np.zeros((runs, episodes))
    sarsa_rewards = np.zeros((runs, episodes))
    
    q_agent_final = None
    sarsa_agent_final = None
    
    for r in range(runs):
        q_agent = QLearningAgent(env.action_space, env.shape, alpha=0.5, gamma=1.0, epsilon=0.1)
        sarsa_agent = SarsaAgent(env.action_space, env.shape, alpha=0.5, gamma=1.0, epsilon=0.1)
        
        q_learning_rewards[r] = train_agent(q_agent, env, episodes)
        sarsa_rewards[r] = train_agent(sarsa_agent, env, episodes)
        
        if r == runs - 1:
            q_agent_final = q_agent
            sarsa_agent_final = sarsa_agent
            
    avg_q_rewards = np.mean(q_learning_rewards, axis=0)
    avg_sarsa_rewards = np.mean(sarsa_rewards, axis=0)
    
    return avg_q_rewards, avg_sarsa_rewards, q_agent_final, sarsa_agent_final

if __name__ == "__main__":
    from analyze import plot_rewards, print_policy
    from visualize import create_policy_gif
    print("Running experiments...")
    q_rewards, sarsa_rewards, q_agent, sarsa_agent = run_experiment(runs=50, episodes=500)
    
    print("Saving plots...")
    plot_rewards(q_rewards, sarsa_rewards)
    
    print("\nQ-learning Policy:")
    print_policy(q_agent, CliffWalkingEnv())
    
    print("\nSARSA Policy:")
    print_policy(sarsa_agent, CliffWalkingEnv())

    print("Generating Struggle GIFs (Early Training)...")
    env = CliffWalkingEnv()
    
    # Train new agents for just 5 episodes to capture struggle
    struggle_q_agent = QLearningAgent(env.action_space, env.shape, alpha=0.5, gamma=1.0, epsilon=0.1)
    train_agent(struggle_q_agent, env, 5)
    create_policy_gif(struggle_q_agent, env, "q_learning_struggle.gif", is_training=True, is_sarsa=False)
    
    struggle_sarsa_agent = SarsaAgent(env.action_space, env.shape, alpha=0.5, gamma=1.0, epsilon=0.1)
    train_agent(struggle_sarsa_agent, env, 5)
    create_policy_gif(struggle_sarsa_agent, env, "sarsa_struggle.gif", is_training=True, is_sarsa=True)

    print("Fine-tuning final agents for another 1000 episodes to ensure strict convergence for final GIFs...")
    train_agent(q_agent, env, 1000)
    train_agent(sarsa_agent, env, 1000)

    print("Generating Final Policy GIFs...")
    create_policy_gif(q_agent, env, "q_learning_policy.gif", is_training=False)
    create_policy_gif(sarsa_agent, env, "sarsa_policy.gif", is_training=False)
    print("Done!")
