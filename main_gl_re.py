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

import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot
# from CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines_al.common.utils import set_random_seed


from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv

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

log_dir = "."

# model.learn(total_timesteps=2000000)
#
# obs = env.reset()
#
# log_dir = "."
# model.save(log_dir + "r1")
# model = PPO.load(log_dir + "r5", env=env)

model.learn(total_timesteps=4000000)
model.save(log_dir + "r_gcaps")