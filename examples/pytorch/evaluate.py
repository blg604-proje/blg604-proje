import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGagent
import torch
from simstarEnv import SimstarEnv
from collections import namedtuple
from collections import defaultdict
import os
import random
import time
import simstar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME_TO_EVAL = "checkpoints/best.dat"

NUM_EVAL_EPISODE = 5
NUM_EVAL_STEPS = 4000

def evaluate(port=8080):
    env = SimstarEnv(track=simstar.TrackName.Austria,port=port,
    synronized_mode=True,speed_up=2,hz=10,
    add_agent=True,
    agent_locs = [10,20,30,40,50],
    num_agents=5)
    
    # total length of chosen observation states
    insize = 4 + env.track_sensor_size + env.opponent_sensor_size

    hyperparams = {
                "lrvalue": 5e-4,
                "lrpolicy": 1e-4,
                "gamma": 0.97,
                "buffersize": 100000,
                "tau": 1e-2,
                "batchsize": 64,
                "start_sigma": 0.3,
                "end_sigma": 0,
                "sigma_decay_len": 15000,
                "theta": 0.15,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    # Load actor network from checkpoint
    agent = DDPGagent(env, hyprm, insize=insize,device=device)
    agent.to(device)
    load_checkpoint(agent)

    total_reward = 0

    for eps in range(NUM_EVAL_EPISODE):
        obs = env.reset()
        state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY,obs.opponents))

        lap_start_time = time.time()
        epsisode_reward = 0

        for i in range(NUM_EVAL_STEPS):
            action = agent.get_action(state)
            a_1 = np.clip(action[0],-1,1)
            a_2 = np.clip(action[1],0,1)
            a_3 = np.clip(action[2],0,1)

            action = np.array([a_1, a_2, a_3])

            obs, reward, done, summary = env.step(action)

            next_state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY,obs.opponents))

            epsisode_reward += reward

            if done:
                # do not restart at accidents
                if "accident" != summary['end_reason']:
                    break
                
                

            state = next_state
        lap_progress = env.get_lap_progress()
        lap_time_passed = time.time() - lap_start_time
        total_reward += epsisode_reward
        print("Episode: %d, Reward: %.1f, lap progress%.2f time passed: %.0fs "%(i,epsisode_reward,lap_progress,lap_time_passed))
    
    print("Average reward over %d episodes: %.1f"%(NUM_EVAL_EPISODE,total_reward/NUM_EVAL_EPISODE))

def load_checkpoint(agent): 
    path = MODEL_NAME_TO_EVAL

    try:
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        if 'epsisode_reward' in checkpoint: reward = float(checkpoint['epsisode_reward']) 
    except FileNotFoundError:
        raise FileNotFoundError("model weights are not found")

if __name__ == "__main__":
    evaluate()


