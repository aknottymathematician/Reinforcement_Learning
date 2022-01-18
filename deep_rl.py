#Import necessary libraries
import gym
import random


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
