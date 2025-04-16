# MONTE CARLO CONTROL ALGORITHM

## AIM
To implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of goal-reaching probability and average return.

## PROBLEM STATEMENT
The task involves solving a Markov Decision Process (MDP) using Monte Carlo Control. The environment is likely a grid world where an agent must navigate through states to reach a goal while maximizing returns. The goal is to compute an optimal policy that achieves the highest probability of success (reaching the goal) and maximizes the average undiscounted return.

## MONTE CARLO CONTROL ALGORITHM

Initialize the policy randomly.

Generate episodes: Simulate episodes in the environment using the current policy.

Update action-value function Q(s,a): For each state-action pair encountered in the episode, update the expected return based on the actual rewards received during the episode.

Policy improvement: Update the policy greedily based on the updated action-value estimates.

Repeat the process until convergence.


## MONTE CARLO CONTROL FUNCTION
```
from tqdm import tqdm
def mc_control(env, gamma=1.0,
               init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    ns, na = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((ns, na), dtype=np.float64)
    Q_track = np.zeros((n_episodes, ns, na), dtype=np.float64)
    pi_track = []

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((ns, na), dtype=bool)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * np.array([x[2] for x in trajectory[t:]]))
            Q[state][action] += alphas[e] * (G - Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/fb3e9786-0712-4c12-9c15-02944c87a272)

![image](https://github.com/user-attachments/assets/7ac59400-a730-4d97-b2c5-6d92d581e697)


### Name: SUDHARSAN S
### Register Number:212224040335

Mention the Action value function, optimal value function, optimal policy, and success rate for the optimal policy.

## RESULT:

Thus to implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of goal-reaching probability and average return is executed successfully.
