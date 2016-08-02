#!/usr/bin/python

import random
import math
import csv
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		''' Initialize the Q matrix
			- The Q matrix is designed as dictionary of dictionaries
			- Main dictionary: states are keys, action-reward functions are values
			- Inner dictionary: actions are keys, rewards are values '''
		self.Q_matrix = {}
		''' Initialize Q value for yet unknown action-reward functions 
			-  Neutral initialization at 0 a'''
		self.Q_0 = 0
		''' Initialize learning rate alpha
			- To what extent the newly acquired information will override the old one
			- 0 <=> agent does not learn at all
			- 1 <=> agent considers only the most recent information '''
		self.alpha = 0.1  
		''' Initialize discount factor gamma
		- Determines the importance of future rewards
		- 0 <=> agent acts short-sightedly by only considering current rewards
		- 1+ <=> agent goes for long-term high reward '''
		self.gamma = 0.8
		''' Initialize parameter for the epsilon-greedy action selection strategy
			- Model the Exploration-Exploitation dilemma
			- 0 <=> agent only exploits
			- 1 <=> agent only explores '''
		self.epsilon = 0.01
		''' Initialize performance metrics basics
			- Initialize trial counter
			- Initialize successful trial counter
			- Initialize successful trial w/o negative reward counter
			- V2: initialize the tracker list for the last ten trials '''
		self.number_of_trials = 0
		self.number_of_successful_trials = 0
		self.number_of_successful_trials_wo_n_r = 0
		self.tracker = [] 

	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		''' Set up performance measures
			- Update the number of trials
			- Reset the flag for negative rewards per trial	'''
		self.number_of_trials += 1
		self.negative_rewards = 'No'
		
	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state 
		''' The state consists of the relevant inputs
			- The planner input indicating the next waypoint
			- The traffic light
			- The direction of oncoming traffic: 
				- If headed forward, no left turn is possible with green light
				- If headed left, no right turn is possible with red light
			- The direction of left coming traffic:
				- If headed forward, no right turn is possible with red light ''' 
		self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left']) 
		''' Memorize state for the later update of the Q matrix '''
		state_t0 = self.state
		
		# TODO: Select action according to your policy
		''' The current state and epsilon are needed in order to determine the action at t_0 '''
		action = self.best_action_select(state=self.state, epsilon=self.epsilon)

		# Execute action and get reward
		reward = self.env.act(self, action)
		
		#self.Q_0 = reward ## try out
		
		''' Prepare performance metrics	
			- Set a flag in case of at least one negative reward during trial '''
		if reward < 0:
			self.negative_rewards = 'Yes'
		''' Prepare and display performance metrics
			- Show if agent has obtained negative rewards during trial
			- Share of successful trials
			- Share of successful trials without negative rewards '''
		if reward > 2:
			self.number_of_successful_trials += 1
			if self.negative_rewards == 'No':
				self.number_of_successful_trials_wo_n_r += 1
			print("\nAlpha: {}, Gamma: {}, Epsilon: {}".format(str(self.alpha),str(self.gamma),str(self.epsilon)))
			print("Has agent obtained negative rewards during this trial? {}".format(self.negative_rewards))
			print("{} out of {} trials were successful ({percent:.2%})".format(str(self.number_of_successful_trials), str(self.number_of_trials), percent=float(self.number_of_successful_trials)/self.number_of_trials))
			print("{} out of {} successful trials were without negative rewards ({percent:.2%})\n".format(str(self.number_of_successful_trials_wo_n_r), str(self.number_of_successful_trials), percent=float(self.number_of_successful_trials_wo_n_r)/self.number_of_successful_trials))				   	
	
		# TODO: Learn policy based on state, action, reward
		''' Q-learning
			- Evaluate Bellman equations from data: Estimate Q from transitions
				- (1) Get the Q value (Q(s_t, a_t)) from the action at time t
				- (2) Update the state after the first action
				- (3) Perform the action at time t+1
				- (4) Get the Q value (Q_s_t+1, a_t+1)) from the second action
				- (5) Corrects the reward for learning purposes
				- (6) Update the Q matrix for the first state and action, in hindsight '''
		Q_t0 = self.Q_matrix[self.state][action] ## (1)
		inputs = self.env.sense(self) ## (2)
		self.state = (self.planner.next_waypoint(), inputs['light'], inputs['oncoming'], inputs['left'])
		action_t1 = self.best_action_select(state=self.state, epsilon=self.epsilon) ## (3)
		Q_t1 = self.Q_matrix[self.state][action_t1] ## (4)
		learning_reward = reward
		if reward > 2:
			learning_reward = reward - 10 ## (5)
		self.Q_matrix[state_t0][action] = Q_t0 + self.alpha * (learning_reward + self.gamma * Q_t1 - Q_t0) ## (6)
		
		''' V2 START: scrutinize the last 10 trials for negative rewards 
			- question: what are the reasons behind negative rewards? '''	
		if self.number_of_trials > 90:
			print_action = action
			if action == None:
				print_action = 'None'
			self.tracker.append([self.number_of_trials, t, state_t0, print_action, reward, Q_t0, self.Q_matrix[state_t0][action], self.Q_matrix[state_t0][None], self.Q_matrix[state_t0]['forward'], self.Q_matrix[state_t0]['right'], self.Q_matrix[state_t0]['left']])
					
		if self.number_of_trials == 100:
			print(self.tracker)
			''' uncomment the following section to produce a file output '''
			#outfile  = open("/path/to/file/smartcab_scrutiny.csv", "wb") 
			#writer = csv.writer(outfile, delimiter=';')
			#for row in self.tracker:
			#	writer.writerow(row)
		''' V2 END '''		

		#if reward < 0:
		#	print(self.number_of_trials, t, action, reward)	## debug
		#	print(state_t0)									## debug
		#	print(self.Q_matrix)							## debug
		#	print("-----------")  							## debug

	''' Function to determine the best action at a given time, based on Q values
		- If the state is not known yet, initializes the action-reward functions to Q_0, and in this case chooses a random action
		- Else, 
			- If random choice indicates exploration, chooses a random action
			- Else,
				- Searches the action-reward functions for the action with the highest reward. Chooses randomly among the actions with the highest rewards if more than one actions with equally high rewards exist '''
	def best_action_select(self, state=None, epsilon=None):
		if state not in self.Q_matrix:
			action_reward = {}
			for action in self.env.valid_actions:
				action_reward[action] = self.Q_0
				self.Q_matrix[state] = action_reward
			best_action = random.choice(self.env.valid_actions)
		else:
			if random.uniform(0,1) <= epsilon:
				best_action = random.choice(self.env.valid_actions)
			else: 
				best_actions = [w[0] for w in self.Q_matrix[state].iteritems() if w[1] == max([v[1] for v in self.Q_matrix[state].iteritems()])]
				if len(best_actions) == 1:
					best_action = best_actions[0]
				else:
					best_action = random.choice(best_actions)           
		return best_action  

def run():
		"""Run the agent for a finite number of trials."""

		# Set up environment and agent
		e = Environment()  # create environment (also adds some dummy traffic)
		a = e.create_agent(LearningAgent)  # create agent
		e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
		# NOTE: You can set enforce_deadline=False while debugging to allow longer trials

		# Now simulate it
		sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
		# NOTE: To speed up simulation, reduce update_delay and/or set display=False

		sim.run(n_trials=100)  # run for a specified number of trials
		# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
	run()