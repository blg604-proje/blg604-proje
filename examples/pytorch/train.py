import numpy as np
import torch
import gym
import random
import time
import os
from collections import namedtuple
from collections import defaultdict
from agent.ddpg import Ddpg
from agent.simple_network import SimpleNet
from agent.simple_network import DoubleInputNet
from agent.random_process import OrnsteinUhlenbeckProcess
from simstarEnv import SimstarEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL_EACH = 5e3
START_FROM_CHECKPOINT = True
SAVE_FOLDER = "checkpoints/"

def train():
    env = SimstarEnv()
    # total length of chosen observation states
    insize = 23
    outsize = env.action_space.shape[0]
    hyperparams = {
                "lrvalue": 0.001,
                "lrpolicy": 0.001,
                "gamma": 0.985,
                "episodes": 30000,
                "buffersize": 300000,
                "tau": 0.01,
                "batchsize": 32,
                "start_sigma": 0.9,
                "end_sigma": 0.1,
                "theta": 0.15,
                "maxlength": 1000,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    datalog = defaultdict(list)
    
    valuenet = DoubleInputNet(insize, outsize, 1)
    policynet = SimpleNet(insize, outsize, activation=torch.nn.functional.tanh)
    agent = Ddpg(valuenet, policynet, buffersize=hyprm.buffersize)
    step_counter = 0
    agent.to(device)
    if(START_FROM_CHECKPOINT):
        step_counter = load_checkpont(agent)

    

    for eps in range(hyprm.episodes):
        obs = env.reset()
        state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY))

        epsisode_reward = 0
        episode_value = 0
        sigma = (hyprm.start_sigma-hyprm.end_sigma)*(max(0, 1-eps/hyprm.episodes)) + hyprm.end_sigma
        randomprocess = OrnsteinUhlenbeckProcess(hyprm.theta, sigma, outsize)
        for i in range(hyprm.maxlength):
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action, value = agent.act(torch_state)
            action = randomprocess.noise() + action.to("cpu").squeeze()
            action.clamp_(-1, 1)
            obs, reward, done, _ = env.step(action.detach().numpy())
            next_state = np.hstack((obs.angle, obs.track,
                    obs.trackPos, obs.speedX, obs.speedY))

            agent.push(state, action, reward, next_state, done)

            epsisode_reward += reward

            if len(agent.buffer) > hyprm.batchsize:
                value_loss, policy_loss = agent.update(hyprm.gamma, hyprm.batchsize, hyprm.tau, hyprm.lrvalue, hyprm.lrpolicy, hyprm.clipgrad)
                if random.uniform(0, 1) < 0.01:
                    datalog["td error"].append(value_loss)
                    datalog["avearge policy value"].append(policy_loss)

            if done:
                break
            state = next_state
            step_counter+=1
            if not np.mod(step_counter,SAVE_MODEL_EACH):
                save_checkpoint(agent,step_counter)
                
        datalog["epsiode length"].append(i)
        datalog["total reward"].append(epsisode_reward)

        avearage_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        print("\r Processs percentage: {:2.1f}%, Average reward: {:2.3f}".format(eps/hyprm.episodes*100, avearage_reward), end="", flush=True)

    print("")


def save_checkpoint(agent,step_counter):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    path = SAVE_FOLDER + "checkpoint.dat"
    torch.save({
                'steps': step_counter,
                'agent_state_dict': agent.state_dict(),
                'opt_policy_state_dict': agent.opt_policy.state_dict(),
                'opt_value_state_dict':agent.opt_value.state_dict(),
                }, path)

def load_checkpont(agent):
    steps = 0
    path = SAVE_FOLDER + "checkpoint.dat"
    try:
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.opt_policy.load_state_dict(checkpoint['opt_policy_state_dict'])
        agent.opt_value.load_state_dict(checkpoint['opt_value_state_dict'])
        steps = int(checkpoint['steps'])
    except FileNotFoundError:
        print("checkpoint not found")
    return steps

if __name__ == "__main__":
    train()