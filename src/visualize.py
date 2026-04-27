import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_policy_gif(agent, env, filename="policy.gif", interval=150, show_arrows=True, is_training=False, is_sarsa=False):
    """
    Generate a GIF.
    - interval=150: double speed (was 300).
    - show_arrows: draw greedy policy arrows on grid.
    - is_training: if True, let the agent learn and explore during the episode to show struggle.
    """
    state = env.reset()
    
    if not is_training:
        agent.epsilon = 0.0 # Greedy for final policy
        
    path = [state]
    q_tables = [np.copy(agent.q_table)]
    done = False
    max_steps = 60 # Cap steps to avoid huge GIF
    steps = 0
    
    action = agent.choose_action(state)
    
    while not done and steps < max_steps:
        next_state, reward, done = env.step(action)
        path.append(next_state)
        
        if is_training:
            if is_sarsa:
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)
                action = agent.choose_action(next_state)
        else:
            action = agent.choose_action(next_state)
            
        q_tables.append(np.copy(agent.q_table))
        state = next_state
        steps += 1
        
    fig, ax = plt.subplots(figsize=(12, 4))
    action_chars = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    def update(frame):
        ax.clear()
        # Draw grid
        ax.set_xlim(-0.5, env.shape[1] - 0.5)
        ax.set_ylim(-0.5, env.shape[0] - 0.5)
        ax.set_xticks(np.arange(-0.5, env.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, env.shape[0], 1))
        ax.grid(True)
        ax.invert_yaxis() # y=0 at top, y=3 at bottom
        
        # Color specific cells
        start_rect = plt.Rectangle((-0.5, 2.5), 1, 1, facecolor='green', alpha=0.5)
        goal_rect = plt.Rectangle((10.5, 2.5), 1, 1, facecolor='gold', alpha=0.5)
        ax.add_patch(start_rect)
        ax.add_patch(goal_rect)
        
        for i in range(1, 11):
            cliff_rect = plt.Rectangle((i - 0.5, 2.5), 1, 1, facecolor='gray', alpha=0.5)
            ax.add_patch(cliff_rect)
            
        ax.text(0, 3, 'Start', ha='center', va='center', fontsize=10, weight='bold')
        ax.text(11, 3, 'Goal', ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw Arrows based on current Q-table
        if show_arrows:
            qt = q_tables[frame]
            for y in range(env.shape[0]):
                for x in range(env.shape[1]):
                    if y == 3 and 1 <= x <= 10:
                        ax.text(x, y, 'C', ha='center', va='center', color='black', alpha=0.5)
                    elif y == 3 and x in (0, 11):
                        pass
                    else:
                        q_vals = qt[y, x]
                        max_q = np.max(q_vals)
                        # To keep arrows stable, prefer first found action if multiple max
                        best_action = np.where(q_vals == max_q)[0][0]
                        ax.text(x, y, action_chars[best_action], ha='center', va='center', fontsize=14, color='gray', alpha=0.8)

        # Draw agent path up to current frame
        current_path = path[:frame+1]
        
        # Split path into continuous segments to avoid drawing lines on cliff reset jumps
        segments = []
        current_segment = [current_path[0]]
        for i in range(1, len(current_path)):
            prev = current_path[i-1]
            curr = current_path[i]
            # If manhattan distance > 1, it's a jump (cliff reset)
            dist = abs(prev[0] - curr[0]) + abs(prev[1] - curr[1])
            if dist > 1:
                segments.append(current_segment)
                current_segment = [curr]
            else:
                current_segment.append(curr)
        segments.append(current_segment)
        
        for seg in segments:
            if len(seg) > 0:
                ys = [p[0] for p in seg]
                xs = [p[1] for p in seg]
                ax.plot(xs, ys, color='blue', marker='.', linestyle='-', linewidth=2, markersize=10, alpha=0.5)
        
        # Current position
        cy, cx = path[frame]
        ax.plot(cx, cy, color='red', marker='o', markersize=15)
        
        status = "Training (Struggling)" if is_training else "Final Policy"
        ax.set_title(f"{status} - Step {frame}")
        ax.set_aspect('equal')
        
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=interval, repeat=False)
    ani.save(filename, writer='pillow')
    plt.close()
