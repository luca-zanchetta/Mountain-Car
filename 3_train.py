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


# Training

# For plotting purposes
rewardList = []

for episode in range(1, EPISODES+1):
    state, _ = env.reset()
    total_reward = 0

    truncated = False
    done = False
    actionRecap = [0,0,0]   # For debugging purposes: we count the number of predicted actions in each episode 

    while not done and not truncated:
        # Default choice is random
        action = env.action_space.sample()
        
        # Choose between greedy and random policy
        if np.random.random()*100 >= epsilon and episode > RAND_EPISODES:
            # We use the model to predict the next action
            q_values = model.predict_on_batch(tf.constant([state]))
            action = np.argmax(q_values[0])
            actionRecap[action] += 1

        # Perform a step
        new_state, reward, done, truncated, info = env.step(action)        
        total_reward += reward

        # Reward policy
        if new_state[0] - state[0] > 0 and action == 2: 
            reward = reward + 1
        if new_state[0] - state[0] < 0 and action == 0: 
            reward = reward + 1
        else:
            reward = reward

        # Popping memory policy
        if len(experience)>= EXP_MAX_SIZE:
            experience.popleft()

        # Fill the experience replay memory with the experience of the current episode
        experience.append([*[state, action, reward, new_state, done]])
        state = new_state
        
        
    if len(experience) >= BATCH_SIZE and episode % TRAIN_EVERY == 0:    # It's time to train!
        # Create a batch by randomly sampling the experience replay memory
        batch = random.sample(experience, BATCH_SIZE)
        datasetGen = []

        for i in range(0, len(batch)):
            # Single entry of the computed batch
            entry = batch[i]

            state = entry[0]        # Gather the current state
            action = entry[1]       # Gather the current action
            reward = entry[2]       # Gather the current reward
            new_state = entry[3]    # Gather the next state
            done = entry[4]         # Gather the information about whether the current state was a terminal state or not

            # By default, the q-values of the next state are the reward 
            # (this is true only if the current state is a terminal state)
            qValueNext = reward
            if not done:    # Not the terminal state
                qValueNext += GAMMA * np.max(model.predict_on_batch(tf.constant([new_state])))      # DQN Bellman equation              
                
            qcurrent = model.predict_on_batch(tf.constant([state]))[0]
            qcurrent[action] = qValueNext
            datasetGen.append([*[*state, *qcurrent]])

        # Compute the dataset used for training
        dataset = np.array(datasetGen)
        X = dataset[:,:state_size]  # Observations
        Y = dataset[:,state_size:]  # Q-values of the actions
        
        # Train the model
        model.fit(tf.constant(X),tf.constant(Y), validation_split=0.2)
        
        # Linear epsilon decay
        if episode > RAND_EPISODES:
            epsilon = ((EPS_MIN - EPS_MAX) * (episode - RAND_EPISODES - 1 )) / (EPISODES*.80 - 1) + EPS_MAX
            if epsilon <= EPS_MIN:
                epsilon = EPS_MIN


    rewardList.append(total_reward)
    print("Episode: {}/{}, Total Reward: {}, Exploration Rate: {:.2f}, Actions: 0 - {}; 1 - {}; 2 - {}".format(
        episode, EPISODES, total_reward, epsilon, actionRecap[0],actionRecap[1],actionRecap[2]))

# Save weights
model.save_weights(checkpoint_third)

# Plot the rewards wrt each training episode
showPlotReward(rewardList)

env.close()