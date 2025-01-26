def generate_rewards(grid_size, goal_state, obstacle_states=[]):
    """
    Generate a reward matrix for a grid world.
    
    Parameters:
        grid_size (tuple): Size of the grid (rows, columns).
        goal_state (int): The state that provides a high reward.
        obstacle_states (list): States that provide a penalty.
        
    Returns:
        reward_matrix: A dictionary of rewards in the form R(S, a, S').
    """
    rows, cols = grid_size
    num_states = rows * cols
    
    # Define rewards
    default_reward = -1  # Default penalty for each move
    goal_reward = 0     # Reward for reaching the goal
    obstacle_penalty = 0  # Penalty for hitting an obstacle
    
    # Initialize reward dictionary
    reward_matrix = {(s, a, sp): default_reward for s in range(num_states) 
                     for a in ["right", "left", "up", "down"] 
                     for sp in range(num_states)}
    
    # Update rewards for the goal state
    for a in ["right", "left", "up", "down"]:
        reward_matrix[(goal_state, a, goal_state)] = goal_reward
    
    # Update rewards for obstacle states
    for obs in obstacle_states:
        for a in ["right", "left", "up", "down"]:
            reward_matrix[(obs, a, obs)] = obstacle_penalty
    
    return reward_matrix






def generate_grid_probabilities(grid_size):
    """
    Generate probability matrices for each action: right, left, up, and down.

    Parameters:
        grid_size (tuple): Size of the grid (rows, columns).

    Returns:
        right_prob, left_prob, up_prob, down_prob: Lists of lists representing transition probabilities for each action.
    """
    rows, cols = grid_size
    num_states = rows * cols
    
    # Initialize empty lists for each action
    right_prob = [[] for _ in range(num_states)]
    left_prob = [[] for _ in range(num_states)]
    up_prob = [[] for _ in range(num_states)]
    down_prob = [[] for _ in range(num_states)]
    
    for r in range(rows):
        for c in range(cols):
            current_state = r * cols + c  # Flatten 2D grid to 1D state
            
            # Transition probabilities for moving RIGHT
            if c < cols - 1:  # Not at the right edge
                right_prob[current_state] = [0] * num_states
                right_prob[current_state][current_state + 1] = 1
            else:  # Stay in the same state if hitting the boundary
                right_prob[current_state] = [0] * num_states
                right_prob[current_state][current_state] = 1
            
            # Transition probabilities for moving LEFT
            if c > 0:  # Not at the left edge
                left_prob[current_state] = [0] * num_states
                left_prob[current_state][current_state - 1] = 1
            else:  # Stay in the same state if hitting the boundary
                left_prob[current_state] = [0] * num_states
                left_prob[current_state][current_state] = 1
            
            # Transition probabilities for moving UP
            if r > 0:  # Not at the top edge
                up_prob[current_state] = [0] * num_states
                up_prob[current_state][current_state - cols] = 1
            else:  # Stay in the same state if hitting the boundary
                up_prob[current_state] = [0] * num_states
                up_prob[current_state][current_state] = 1
            
            # Transition probabilities for moving DOWN
            if r < rows - 1:  # Not at the bottom edge
                down_prob[current_state] = [0] * num_states
                down_prob[current_state][current_state + cols] = 1
            else:  # Stay in the same state if hitting the boundary
                down_prob[current_state] = [0] * num_states
                down_prob[current_state][current_state] = 1
    
    return right_prob, left_prob, up_prob, down_prob




def policy_evaluation(non_terminal_states, all_states, actions, probs_dict, reward_matrix, discount, action_policy_dict):
    value_function = [0 for _ in range(16)]
    for i in range(15):
        print(f"iteration {i+1}")
        for state in non_terminal_states:
            value_function_temp = 0;
            for action in actions:
                state_reward_accumulation = 0;
                for state_old in all_states: 
                    state_reward_accumulation += probs_dict[action][state][state_old] * (reward_matrix[(state,action,state_old)] + discount*value_function[state_old]);
                value_function_temp += action_policy_dict[state][action] * state_reward_accumulation;
            value_function[state] = round(value_function_temp, 1);
        print(value_function)
    return value_function;



def policy_improvement(value_function,non_terminal_states, all_states, actions, probs_dict, reward_matrix, discount, action_policy_dict):
    policy_stable = True;
    for state in non_terminal_states:
        old_action = action_policy_dict[state];
        action_values = {action: 0 for action in actions}
        
        for action in actions:
            for next_state in all_states:
                action_values[action] += probs_dict[action][state][next_state] * (reward_matrix[(state,action,next_state)] + discount * value_function[next_state])
        
        max_action = max(action_values, key=action_values.get);
        action_policy_dict[state] = {action: 0 for action in actions};
        action_policy_dict[state][max_action] = 1;

        if old_action != action_policy_dict[state]:
            policy_stable = False;
    
    return policy_stable, action_policy_dict;