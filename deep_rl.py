#Import necessary libraries
import gym
import random

#Deep Learning Model dependencies
import numpy as np
from tensorflow.keras.models import Sequential
from tensonflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam




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
	print("Episode: {} Score: {}".format(episode, score))


def build_model(states, actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1, states)))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	return model

model = build_model(states, actions)

model.summary()