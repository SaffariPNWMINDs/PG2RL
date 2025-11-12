import numpy as np

def policy_iteration(P, R, gamma=0.9, theta=1e-6):
    """
    Policy Iteration Algorithm.

    Parameters:
        P (dict): State transition probabilities. P[state][action] = [(prob, next_state, reward, done), ...]
        R (numpy.ndarray): Reward function. R[state, action] = reward.
        gamma (float): Discount factor.
        theta (float): Convergence threshold for policy evaluation.

    Returns:
        policy (numpy.ndarray): Optimal policy.
        V (numpy.ndarray): Value function for the optimal policy.
    """
    n_states = len(P)
    n_actions = len(P[0])
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)

    def policy_evaluation(policy, V):
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                action = policy[s]
                V[s] = sum(prob * (reward + gamma * V[next_state])
                           for prob, next_state, reward, _ in P[s][action])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

    def policy_improvement(policy, V):
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                action_values[a] = sum(prob * (reward + gamma * V[next_state])
                                       for prob, next_state, reward, _ in P[s][a])
            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False
        return policy_stable

    while True:
        policy_evaluation(policy, V)
        if policy_improvement(policy, V):
            break

    return policy, V

def value_iteration(P, R, gamma=0.9, theta=1e-6):
    """
    Value Iteration Algorithm.

    Parameters:
        P (dict): State transition probabilities. P[state][action] = [(prob, next_state, reward, done), ...]
        R (numpy.ndarray): Reward function. R[state, action] = reward.
        gamma (float): Discount factor.
        theta (float): Convergence threshold for value function.

    Returns:
        policy (numpy.ndarray): Optimal policy.
        V (numpy.ndarray): Optimal value function.
    """
    n_states = len(P)
    n_actions = len(P[0])
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            V[s] = max(
                sum(prob * (reward + gamma * V[next_state])
                    for prob, next_state, reward, _ in P[s][a])
                for a in range(n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = sum(prob * (reward + gamma * V[next_state])
                                   for prob, next_state, reward, _ in P[s][a])
        policy[s] = np.argmax(action_values)

    return policy, V

P = {
    0: {
        0: [(1.0, 0, 0, False)],
        1: [(1.0, 1, 0, False)]
    },
    1: {
        0: [(1.0, 0, 1, False)],
        1: [(1.0, 1, 1, False)]
    }
}

R = np.array([
    [0, 0],
    [1, 1]
])

policy, value = policy_iteration(P, R, gamma=0.9)
print("Optimal Policy (Policy Iteration):", policy)
print("Value Function (Policy Iteration):", value)

policy, value = value_iteration(P, R, gamma=0.9)
print("Optimal Policy (Value Iteration):", policy)
print("Value Function (Value Iteration):", value)
