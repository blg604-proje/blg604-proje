import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
import torch
from simstarEnv import SimstarEnv
from collections import namedtuple
from collections import defaultdict
import os
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL_EACH = 2000
TRAIN = True
ADD_AGENT = True
START_FROM_CHECKPOINT = True
SAVE_FOLDER = "checkpoints/"

def train():
    env = SimstarEnv(synronized_mode=True,speed_up=5,hz=6,
    add_agent=ADD_AGENT,agent_rel_pos=100,agent_set_speed=30)
    # total length of chosen observation states
    insize = 4 + env.track_sensor_size + env.opponent_sensor_size

    outsize = env.action_space.shape[0]
    hyperparams = {
                "lrvalue": 5e-4,
                "lrpolicy": 1e-3,
                "gamma": 0.97,
                "episodes": 30000,
                "buffersize": 100000,
                "tau": 1e-2,
                "batchsize": 64,
                "start_sigma": 0.3,
                "end_sigma": 0,
                "sigma_decay_len": 15000,
                "theta": 0.15,
                "maxlength": 5000,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    datalog = defaultdict(list)
    
    agent = DDPGagent(env, hyprm, device=device)
    noise = OUNoise(env.action_space, hyprm)
    agent.to(device)
    step_counter = 0
    best_reward = 0
    #agent.to(device)

    if(START_FROM_CHECKPOINT):
        step_counter,best_reward = load_checkpoint(agent)


    for eps in range(hyprm.episodes):
        obs = env.reset()
        noise.reset()
        state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY,obs.opponents))


        epsisode_reward = 0
        episode_value = 0


        for i in range(hyprm.maxlength):
            action = agent.get_action(state)
            if TRAIN:
                action = noise.get_action(action, step_counter)

            a_1 = np.clip(action[0],-1,1)
            a_2 = np.clip(action[1],0,1)
            a_3 = np.clip(action[2],0,1)

            action = np.array([a_1, a_2, a_3])

            obs, reward, done, _ = env.step(action)

            next_state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY,obs.opponents)

            agent.memory.push(state, action, reward, next_state, done)

            epsisode_reward += reward

            if TRAIN:
                if len(agent.memory) > hyprm.batchsize:
                    agent.update(hyprm.batchsize)

            if done:
                break

            state = next_state
            step_counter+=1

            if not np.mod(step_counter,SAVE_MODEL_EACH):
                save_checkpoint(agent,step_counter,epsisode_reward)
                if epsisode_reward > best_reward:
                    save_checkpoint(agent,step_counter,epsisode_reward,save_name="best")
                
        datalog["epsiode length"].append(i)
        datalog["total reward"].append(epsisode_reward)

        avearage_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        print("\r Processs percentage: {:2.1f}%, Average reward: {:2.3f}".format(eps/hyprm.episodes*100, avearage_reward), end="", flush=True)

    print("")


def save_checkpoint(agent,step_counter,epsisode_reward,save_name="checkpoint_opponent"):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    path = SAVE_FOLDER + save_name +"_opponent.dat"
    torch.save({
                'steps': step_counter,
                'agent_state_dict': agent.state_dict(),
                'opt_policy_state_dict': agent.critic_optimizer.state_dict(),
                'opt_value_state_dict':agent.actor_optimizer.state_dict(),
                'epsisode_reward':epsisode_reward
                }, path)

def load_checkpoint(agent):
    steps = 0
    reward = 0 
    path = SAVE_FOLDER + "checkpoint_opponent.dat"
    try:
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['opt_policy_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['opt_value_state_dict'])
        steps = int(checkpoint['steps'])
        if 'epsisode_reward' in checkpoint: reward = float(checkpoint['epsisode_reward']) 
    except FileNotFoundError:
        print("checkpoint not found")
    return steps,reward

if __name__ == "__main__":
    train()


