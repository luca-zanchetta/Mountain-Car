import gymnasium as gym
from gym import Env
import numpy as np
import random

#Policies

class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def __call__(self, obs) -> int:
        return random.randint(0, self.n_actions - 1)

class GreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
    def __call__(self, obs) -> int:
        return np.argmax(self.Q[obs])

class EpsGreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
        self.n_actions = len(Q[0])
    def __call__(self, obs, eps: float) -> int:
        greedy = random.random() > eps
        if greedy:
            return np.argmax(self.Q[obs])
        else:
            return random.randint(0, self.n_actions - 1)

from collections import defaultdict

def qlearn(
    env: gym.Env,
    alpha0: float,
    gamma: float,
    max_steps: int,
):
    """Q-learning training loop.
    
    Returns estimated optimal Q-table.
    """
    # New Q-function and policy
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = EpsGreedyPolicy(Q)

    # Iterate steps
    done = True
    for step in range(max_steps):

        # Should I reset?
        if done:
            obs = env.reset()
        
        # Select action
        eps = (max_steps - step) / max_steps
        action = policy(obs, eps)

        # Act
        obs2, rew, done, info, _ = env.step(action)

        # Update
        Q[obs][action] += alpha0 * (rew + gamma * np.max(Q[obs2]) - Q[obs][action])
        obs = obs2
    
    return Q
  

#ROLLouts function
def rollouts(
    env: gym.Env,
    policy,
    gamma: float,
    n_episodes: int,
    render=False,
) -> float:
    """Perform rollouts and compute the average discounted return."""
    sum_returns = 0.0

    # Init
    done = False
    obs = env.reset()
    discounting = 1
    ep = 0
    if render:
        env.render()

    # Iterate steps
    while True:

        # Should I reset?
        if done:
            if render:
                print("New episode")
            obs = env.reset()
            discounting = 1
            ep += 1
            if ep >= n_episodes:
                break
        
        # Select action
        action = policy(obs)

        # Act
        obs, rew, done, _, _ = env.step(action)

        # Sum
        sum_returns += rew * discounting
        discounting *= gamma

        # Maybe watch
        if render:
            env.render()
    
    
    return sum_returns / n_episodes


env = gym.make("MountainCar-v0", render_mode="human")
env.metadata['render_fps'] = 144
from gym.wrappers import TimeLimit
env = TimeLimit(env, max_episode_steps=50)

# avg_return = rollouts(
#     env=env,
#     policy=RandomPolicy(env.action_space.n),
#     gamma=0.95,
#     n_episodes=5,
#     render=True,
# )

#print("Avg return", avg_return)

qtable = qlearn(env=env, alpha0=0.1, gamma=0.95, max_steps=200000)

policy_obj = GreedyPolicy(qtable)
greedy_policy = {obs: policy_obj(obs) for obs in range(env.observation_space.n)}
print(greedy_policy)

avg_return = rollouts(env=env, policy=GreedyPolicy(qtable), gamma=0.95, n_episodes=20)
print(avg_return)

rollouts(env=env, policy=GreedyPolicy(qtable), gamma=0.97, n_episodes=1, render=True)