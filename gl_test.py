"""
Author: Steve Paul 
Date: 4/28/22 """
from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from UAMEnv import UAMEnv
import json
import datetime as dt
import torch as th
from CustomNN import CustomNN
from Policies.CustomPolicies import ActorCriticGCAPSPolicy
import pickle

import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot
# from CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines_al.common.utils import set_random_seed


from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv

def as_tensor(obs):
    obs["agent_taking_decision_coordinates"] = th.tensor(obs["agent_taking_decision_coordinates"])
    obs["agents_destination_coordinates"] = th.tensor(obs["agents_destination_coordinates"])
    obs["depot"] = th.tensor(obs["depot"])
    obs["first_dec"] = th.tensor(obs["first_dec"])
    obs["location"] = th.tensor(obs["location"])
    obs["mask"] = th.tensor(obs["mask"])
    obs["topo_laplacian"] = th.tensor(obs["topo_laplacian"])
    return obs

warnings.filterwarnings('ignore')
n_vertiports = 8 # number of vertiports
n_evtols = 19 # number of evtols
max_x_y_axis = 50
vertiport_locations = th.rand((n_vertiports,2))*max_x_y_axis # actual coordinates of the vertiports
evtols_initial_locations = th.randint(0, n_vertiports, (n_evtols, 1)) # initial locations of the evtols
demand = th.randint(0, 100, (n_vertiports, n_vertiports)) # constant demand for now
ticket = demand*.05 # price of the ticket
for i in range(n_vertiports):
    demand[i,i] = 0
time_horizon = 18.00 # time before the last decision
env = UAMEnv(
        n_vertiports=n_vertiports,
        n_evtols=n_evtols,
        vertiport_locations=vertiport_locations,
        evtols_initial_locations = evtols_initial_locations,
        demand_model=demand,
        time_horizon = time_horizon,
        max_x_y_axis = max_x_y_axis
    )

policy_kwargs=dict(
    # features_extractor_class=GCAPCNFeatureExtractor,
    # features_extractor_kwargs=dict(features_dim=128,node_dim=2),
    activation_fn=th.nn.LeakyReLU,
    net_arch=[dict(vf=[128],pi=[128])]
)

model = PPO(
    ActorCriticGCAPSPolicy,
    env,
    gamma=1.00,
    verbose=1,
    n_epochs=100,
    batch_size=5000,
    tensorboard_log="logger/",
    # create_eval_env=True,
    n_steps=15000,
    learning_rate=0.000001,
    policy_kwargs = policy_kwargs,
    ent_coef=0.4,
    vf_coef=0.5
)

log_dir = "Trained_models/."

# model.learn(total_timesteps=2000000)
#
# obs = env.reset()
#
# log_dir = "."
# model.save(log_dir + "r1")
method = "r_mlp"
dir = "Results/"
model = PPO.load(log_dir + method, env=env)

obs = env.reset()
reward_agg = []
profit_agg = []
data = {}
test_envt_list = []
with open("test_envts_2.pkl", 'rb') as f:
    test_data = pickle.load(f)
test_envt_list = test_data["env"]
first_obs_list = test_data["first_obs"]
j = 0
env = test_envt_list[j]
obs = first_obs_list[j]
env.traj_log_list = []
for i in range(1000000):
    # obs = as_tensor(obs)
    # mask = obs["mask"]
    # mask_non_zeros = mask.nonzero()[:, 0]
    #
    # ind = th.randint(0, mask_non_zeros.shape[0], (1, 1))
    obs["evtol_taking_decision_location"] = obs["evtol_taking_decision_location"][None,:]

    action = model.predict(obs)
    # print(mask_non_zeros)
    # print(action[0])
    obs, reward, done, _ = env.step(action[0])
    if done:
        print(env.total_reward)
        reward_agg.append(env.total_reward)
        profit = env.total_reward*((env.time_varying_demand_model["demand"]*env.passenger_pricing_model).sum()).item()
        profit_agg.append(profit)
        if len(reward_agg) == 100: # testing for 100 different scenarios
            data["total_reward"] = np.array(reward_agg)
            data["profit"] = np.array(profit_agg)
            with open(dir+method+"_results_2.pkl", 'wb') as f:
                pickle.dump(data, f)
            print(np.array(reward_agg).mean(), np.array(reward_agg).std(), np.array(profit_agg).mean())
            break
        j = j+1
        env = test_envt_list[j]
        obs = first_obs_list[j]