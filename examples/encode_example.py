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


import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--type", type=str, default="single")
parser.add_argument("--envname", type=str, default="contgridworld")
parser.add_argument('--types', nargs='+', type=str)
parser.add_argument('--names', nargs='+', type=str)
parser.add_argument("--env", type=str, default="target")
parser.add_argument("--dir", type=str, default="deep-q")
parser.add_argument("--load-from", type=str, default="dynamics")
parser.add_argument("--size", type=int, default=4)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--encode-size", type=int, default=0)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--instances", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--cost", type=float, default=0.0)
parser.add_argument("--reward-coeff", type=float, default=1.0)

parser.add_argument('--load', dest='load', action='store_true')
parser.set_defaults(load=False)

parser.add_argument('--cumplot', dest='cumplot', action='store_true')
parser.set_defaults(cumplot=False)

parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

def deep(args, open_plot=True):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Setup MDP, Agents.
    n = args.size
    path = "models/" + args.envname
    if not os.path.exists(path):
        os.makedirs(path)

    if args.env == "source":
        mdp = EncodeGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                    # lava_locs=[(1,n), (n,1)], 
                    step_cost=args.cost, name=args.envname+"_source")
        state_dim = n**2
        encode_size = args.encode_size if args.encode_size > 0 else None

    elif args.env == "target":
        mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                    # lava_locs=[(1,n), (n,1)], 
                    step_cost=args.cost, name=args.envname)
        state_dim = 2
        encode_size = args.encode_size if args.encode_size > 0 else n**2

    agents = []
    for type in args.types:
        dql_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=state_dim, 
                encode_size=encode_size, name=type, save=True, learn_type=type, 
                load_from=args.load_from, save_dir=path, lr=args.lr, epoch=args.epoch,
                reward_coeff=args.reward_coeff) 
        agents.append(dql_agent)
    
    # Run experiment and make plot.
    run_agents_on_mdp(agents, mdp, instances=args.instances, cumulative_plot=args.cumplot,
            episodes=args.episodes, steps=40, reset_at_terminal=True, verbose=args.verbose, open_plot=False)

def deep_target(open_plot=True):
    # Setup MDP, Agents.
    n = 4 
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)
    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=n**2, random=False, 
                load=False, save=False, learn=True, learn_dynamics=False, load_dynamics=False)
    # pretrain_dql_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2,
    #             save=True, learn_type="encoder", name="Encoder") 

    transfer_dql_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2,
                save=False, learn_type="transfer", name="Transfer-pretrain") 
    regularize_dql_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2,
                save=False, learn_type="regularize", name="Transfer-regularize") 
    single_dql_agent = DeepQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2,
                save=False, learn_type="single", name="Single Target") 
    # transfer_dql_agent.test_dynamics()
    # transfer_dql_agent.test_encoder()
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp(
            # [pretrain_dql_agent],
            [single_dql_agent, transfer_dql_agent, regularize_dql_agent], 
            mdp, instances=3, cumulative_plot=False,
            episodes=500, steps=40, reset_at_terminal=True, verbose=False)

def main(open_plot=True):
    # Setup MDP, Agents.
    n = 4
    mdp = EncodeGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], step_cost=0.1)
    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=n**2, random=False, 
                load=False, save=False, learn=True, learn_dynamics=False, load_dynamics=False)
                # load=True, save=False, learn=False, learn_dynamics=False, load_dynamics=True)
    rand_agent = RandomAgent(actions=mdp.get_actions())
    
    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=1, cumulative_plot=True,
            episodes=200, steps=40, reset_at_terminal=True, verbose=True)

def main_cont(open_plot=True):
    # Setup MDP, Agents.
    n = 4
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)
    
    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                save_dir="../models/linear_4-4/", name="Transfer-Q",
                # random=True, learn=False, load_dynamics=True, learn_encoder=True)
                learn=True, load_dynamics=True, load_encoder=True)
    
    ql_agent_2 = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                save_dir="../models/linear_4-4/", name="Transfer-Q_10000",
                checkpoint="checkpoints/encoder_10000.pkl",
                learn=True, load_dynamics=True, load_encoder=True)

    ql_source = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=None, 
                random=False, learn=True, learn_dynamics=False, name="Target-Linear-Q") 

    base_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                learn=True, baseline_encoder=True, name="RBF Encoder")

    rand_agent = RandomAgent(actions=mdp.get_actions())
    # ql_agent.test_dynamics()
    # Run experiment and make plot.
    run_agents_on_mdp(
            [ql_agent_2],
            # [ql_agent, ql_source, base_agent, rand_agent], 
            mdp, instances=1, cumulative_plot=True,
            episodes=1000, steps=40, reset_at_terminal=True, verbose=False)

def learn_encoder(n=4):
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)
    
    ql_agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                save_dir="../models/linear_4-4/", name="Learn Encoder", save_interval=50,
                random=True, learn=False, load_dynamics=True, learn_encoder=True) 
    run_agents_on_mdp(
            [ql_agent],
            mdp, instances=1, cumulative_plot=True,
            episodes=500, steps=40, reset_at_terminal=True, verbose=False)

def compare(n = 4, open_plot=True):
    # Setup MDP, Agents.
    mdp = ContGridWorldMDP(width=n, height=n, init_loc=(1, 1), goal_locs=[(n, n)], 
                step_cost=0.1)
    
    agents = []
    for i in range(1, 6):
        ck = i * 100 
        agent = LinearQLearningAgent(actions=mdp.get_actions(), state_dim=2, encode_size=n**2, 
                save_dir="../models/linear_4-4/", name="Transfer-Q_ck{}".format(ck),
                learn=True, load_dynamics=True, load_encoder=True, 
                checkpoint="checkpoints/encoder_{}.pkl".format(ck))
        agents.append(agent)

    run_agents_on_mdp(
            agents, 
            mdp, instances=1, cumulative_plot=False,
            episodes=1000, steps=40, reset_at_terminal=True, verbose=False)

if __name__ == "__main__":
    # main(open_plot=not sys.argv[-1] == "no_plot")
    # learn_encoder()
    # compare()
    # main_cont(open_plot=not sys.argv[-1] == "no_plot")
    # deep()
    # deep_target()

    args = parser.parse_args()
    deep(args)