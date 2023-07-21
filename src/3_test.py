# Import dependencies
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque


# Hyperparameters
EXP_MAX_SIZE = 5000             # Maximum size of replay memory
BATCH_SIZE = EXP_MAX_SIZE // 10 # Batch size
EPISODES = 2000                 # Number of training episodes
TRAIN_EVERY = 20                # We train the neural network every 10 episodes
RAND_EPISODES = 400             # Exploration episodes

EPS_MAX = 85                    # Initial exploration probability
EPS_MIN = 5                     # Final exploration probability
GAMMA = .9                      # Discount factor


# Environment Setup
env = gym.make("MountainCar-v0")
env.reset()

state_size = env.observation_space.shape[0]         # 2 states
action_size = env.action_space.n                    # 3 actions

experience = deque([],EXP_MAX_SIZE)                 # Past experience arranged as a queue

c_reward = 0                                        # Current cumulative reward
checkpoint_first =  './checkpoints_first/cp.ckpt'   # File to record network configuration in the first approach
checkpoint_second = './checkpoints_second/cp.ckpt'  # File to record network configuration in the second approach
checkpoint_third =  './checkpoints_third/cp.ckpt'   # File to record network configuration in the third approach


epsilon = EPS_MAX                                   # Initialization of the epsilon


# Neural Network
def createModel():
    model = Sequential()
    model.add(Dense(256, input_shape=( state_size , ),activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size,activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Model setup
model = createModel()
model.summary()


# Show the reward with respect to each episode
def showPlotReward(value):
    plt.plot(value)
    plt.xlabel('episode')
    plt.ylabel('total reward')

    plt.show()


# Testing

# Load the pre-trained model
model = createModel()
model.load_weights(checkpoint_third)

env = gym.make("MountainCar-v0", render_mode= "human")
state, _ = env.reset()
total_reward = 0
truncated = False
done = False
while not done and not truncated:
        q_values = model.predict(tf.constant([state]), verbose=0)
        action = np.argmax(q_values[0])
        new_state, reward, done, truncated, info = env.step(action)
       
        total_reward += reward
        state = new_state
        env.render()

print("Reward: {}".format(total_reward))
env.close()