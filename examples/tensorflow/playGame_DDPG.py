import sys
sys.path.append('./sample_DDPG_agent/')

import numpy as np
np.random.seed(1337)
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3

import collections as col
import random
import argparse
import tensorflow as tf
import timeit
import math
import sys

from configurations import *
from ddpg import *

import gc
gc.enable()

# config parameters are printed in main
USE_SIMSTAR = True
OTHER_AGENT = False
OTHER_AGENT_AUTOPILOT = True
from simstarEnv import SimstarEnv

def playGame(f_diagnostics, train_indicator, port=3101):    # 1 means Train, 0 means simply Run

	action_dim = 3  #Steering/Acceleration/Brake
	state_dim = 23+18  #Number of sensors input  #23 for simstar
	if(USE_SIMSTAR):
		env_name = 'Simstar_Env'
	else:
		env_name = 'Torcs_Env'
	
	agent = DDPG(env_name, state_dim, action_dim)

	# Generate a Torcs environment
	print("I have been asked to use port: ", port)
	if(USE_SIMSTAR):
		env = SimstarEnv(synronized_mode=True,
		speed_up=4,hz=5,add_agent=True,agent_set_speed=0,
		agent_rel_pos=35,autopilot_agent=OTHER_AGENT_AUTOPILOT)
		ob = env.reset()
	else:
		env = TorcsEnv(vision=False, throttle=True, gear_change=False)

		client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
		client.MAX_STEPS = np.inf

		client.get_servers_input(0)  # Get the initial input from torcs

		obs = client.S.d  # Get the current full-observation from torcs
		ob = env.make_observation(obs)

	s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,ob.opponents))
	state_other_vehicle = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,ob.opponents))

	EXPLORE = total_explore
	episode_count = max_eps
	max_steps = max_steps_eps
	epsilon = epsilon_start
	done = False
	epsilon_steady_state = 0.01 # This is used for early stopping.

	totalSteps = 0
	best_reward = -100000
	running_avg_reward = 0.

	print("TORCS Experiment Start.")
	for i in range(episode_count):

		save_indicator = 1
		early_stop = 1
		# Counting the total reward and total steps in the current episode
		total_reward = 0.
		info = {'termination_cause':0}
		distance_traversed = 0.
		speed_array=[]
		trackPos_array=[]

		print('\n\nStarting new episode...\n')

		for step in range(max_steps):

			# Take noisy actions during training
			if (train_indicator):
				epsilon -= 1.0 / EXPLORE
				epsilon = max(epsilon, epsilon_steady_state)
				a_t = agent.noise_action(s_t,epsilon) #Take noisy actions during training
				a_t = agent.action(s_t)
			else:
				a_t = agent.action(s_t)		# a_t is of the form: [steer, accel, brake]

			if(USE_SIMSTAR):
				if not OTHER_AGENT_AUTOPILOT:
					# control second vehicle 
					obs_other = env.get_agent_obs()
					state_other_vehicle = np.hstack((obs_other.angle, obs_other.track, obs_other.trackPos, obs_other.speedX, obs_other.speedY,obs_other.opponents))
					other_vehicle_action = agent.action(state_other_vehicle)
					env.set_agent_action(other_vehicle_action)
					
				ob, r_t, done, info = env.step(a_t)
			else:
				ob, r_t, done, info = env.step(step, client, a_t, early_stop)
			if done:
				break
			analyse_info(info, printing=False)

			s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,ob.opponents))
			distance_traversed += ob.speedX*np.cos(ob.angle) #Assuming 1 step = 1 second
			speed_array.append(ob.speedX*np.cos(ob.angle))
			trackPos_array.append(ob.trackPos)


			#Checking for nan rewards: TODO: This was actually below the following block
			if (math.isnan( r_t )):
				r_t = 0.0
				for bad_r in range( 50 ):
					print( 'Bad Reward Found' )
				break #Introduced by Anirban


			# Add to replay buffer only if training
			if (train_indicator):
				agent.perceive(s_t,a_t,r_t,s_t1,done) # Add experience to replay buffer


			total_reward += r_t
			s_t = s_t1

			# Displaying progress every 15 steps.
			if ( (np.mod(step,15)==0) ):
			    print("Episode", i, "Step", step, "Epsilon", epsilon , "Action", a_t, "Reward", r_t )

			totalSteps += 1
			if done:
				break

		# Saving the best model.
		if ((save_indicator==1) and (train_indicator ==1 )):
			if (total_reward >= best_reward):
				print("Now we save model with reward " + str(total_reward) + " previous best reward was " + str(best_reward))
				best_reward = total_reward
				agent.saveNetwork()

			if np.mod(i, 20) == 0:
				print("***************************************************************************************************************************")
				agent.saveNetwork()

		running_avg_reward = running_average(running_avg_reward, i+1, total_reward)

		print("TOTAL REWARD @ " + str(i) +"-th Episode  : Num_Steps= " + str(step) + "; Max_steps= " + str(max_steps) +"; Reward= " + str(total_reward) +"; Running average reward= " + str(running_avg_reward))
		print("Total Step: " + str(totalSteps))
		print("")

		print(info)
		if(USE_SIMSTAR):
			ob= env.reset()
		else:
			if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
				print('Hard reset by some agent')
				ob, client = env.reset(client=client)
			else:
				ob, client = env.reset(client=client, relaunch=True)
		s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,ob.opponents))

		# document_episode(i, distance_traversed, speed_array, trackPos_array, info, running_avg_reward, f_diagnostics)

	env.end()  # Shut down TORCS
	print("Finish.")

def document_episode(episode_no, distance_traversed, speed_array, trackPos_array, info, running_avg_reward, f_diagnostics):
	"""
	Note down a tuple of diagnostic values for each episode
	(episode_no, distance_traversed, mean(speed_array), std(speed_array), mean(trackPos_array), std(trackPos_array), info[termination_cause], running_avg_reward)
	"""
	f_diagnostics.write(str(episode_no)+",")
	f_diagnostics.write(str(distance_traversed)+",")
	f_diagnostics.write(str(np.mean(speed_array))+",")
	f_diagnostics.write(str(np.std(speed_array))+",")
	f_diagnostics.write(str(np.mean(trackPos_array))+",")
	f_diagnostics.write(str(np.std(trackPos_array))+",")
	f_diagnostics.write(str(info['termination_cause'])+",")
	f_diagnostics.write(str(running_avg_reward)+"\n")

def running_average(prev_avg, num_episodes, new_val):
	total = prev_avg*(num_episodes-1)
	total += new_val
	return np.float(total/num_episodes)

def analyse_info(info, printing=True):
	pass
	#simulation_state = ['Normal', 'Terminated as car is OUT OF TRACK', 'Terminated as car has SMALL PROGRESS', 'Terminated as car has TURNED BACKWARDS']
	#if printing and info['termination_cause']!=0:
	#	print(simulation_state[info['termination_cause']])

if __name__ == "__main__":

	try:
		port = 3101 #port = int(sys.argv[1])
	except Exception as e:
		# raise e
		print("Usage : python %s <port>" % (sys.argv[0]))
		sys.exit()

	print('is_training : ' + str(is_training))
	print('Starting best_reward : ' + str(start_reward))
	print(total_explore)
	print(max_eps)
	print(max_steps_eps)
	print(epsilon_start)
	print('config_file : ' + str(configFile))

	# f_diagnostics = open('output_logs/diagnostics_for_window_' + sys.argv[1]+'_with_fixed_episode_length', 'w') #Add date and time to file name
	f_diagnostics = ""
	playGame(f_diagnostics, train_indicator=1, port=port)
	# f_diagnostics.close()

