#Import necessary libraries
import gym
import random

#Deep Learning Model dependencies
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#Keras dependencies for RL
from rl.agents import DQNAgent #https://keras-rl.readthedocs.io/en/latest/
from rl.policy import BoltzmannQPolicy #We'll be using policy based RL instead of value based RL
from rl.memory import SequentialMemory


#Setup RL Environment
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
	state = env.reset()
	done = False
	score = 0

	while not done:
		env.render() #Renders env, allowing us to cart in action
		action = random.choice([0,1])#Taking a random step either going left or right
		n_state, reward, done, info = env.step(action) #Apply that action thus getting back various parameters
		score+= reward
	# print("Episode: {} Score: {}".format(episode, score))


def build_model(states, actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	return model

model = build_model(states, actions)


#Build Agent with Keras-RL
def build_agent(model, actions):
	policy = BoltzmannQPolicy()
	memory = SequentialMemory(limit=50000, window_length=1)
	dqn = DQNAgent(model = model, memory = memory, policy = policy,nb_actions=actions, nb_steps_warmup=10,
		target_model_update=1e-2)
	return dqn

# # Compile and Fit the DQN Agent
dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

dqn.save_weights("./weights/dqn_weights.h5f", overwrite = True)