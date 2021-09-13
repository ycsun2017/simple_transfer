#!/usr/bin/env python

# Python imports.
from pickle import TRUE
import sys
import os

from numpy.lib.npyio import load
from torch import rand

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent, LinearQLearningAgent, DeepQLearningAgent
from simple_rl.tasks import EncodeGridWorldMDP, ContGridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.abstraction import AbstractionWrapper

import random
import torch

def source(n=4):
    mdp = EncodeGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], step_cost=0.1)
    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=n**2, random=False, 
                load=False, save=False, learn=True, learn_dynamics=False, load_dynamics=False, name="source")
    rand_agent = RandomAgent(actions=mdp.get_actions())
    
    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, cumulative_plot=False,
            episodes=200, steps=40, reset_at_terminal=True, verbose=False, open_plot=False)

def target(n=4):
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)

    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, random=False, 
                load=False, save=False, learn=True, learn_dynamics=False, load_dynamics=False)
    deep_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2,
                save=False, learn_type="single", name="deep", lr=1e-4)
    
    # Run experiment and make plot.
    run_agents_on_mdp([deep_agent], mdp, instances=5, cumulative_plot=False,
            episodes=200, steps=40, reset_at_terminal=True, verbose=False, open_plot=False)

def compare(n = 4, open_plot=True):
    # Setup MDP, Agents.
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)
    
    agents = []
    xs = list(range(50, 2000, 100)) + list(range(2000, 10000, 1000))
    for ck in xs:
        agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                save_dir="../models/linear_4-4/", name="Transfer-Q_ck{}".format(ck),
                learn=True, load_dynamics=True, load_encoder=True, 
                checkpoint="checkpoints/encoder_{}.pkl".format(ck))
        agents.append(agent)

    run_agents_on_mdp(
            agents, 
            mdp, instances=5, cumulative_plot=False, open_plot=False,
            episodes=200, steps=40, reset_at_terminal=True, verbose=False)

if __name__ == "__main__":
    # compare()
    # source()
    target()