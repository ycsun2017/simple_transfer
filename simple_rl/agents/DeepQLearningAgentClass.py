''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import os
import math
import time
from collections import defaultdict
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle

from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Other imports.
from simple_rl.agents.AgentClass import Agent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class L2Norm(nn.Module):
    def forward(self, x):
        if len(x.size()) > 1:
            return x / x.norm(p=2, dim=1, keepdim=True)
        else:
            return x / x.norm(p=2)

class NonLinearModel(nn.Module):

    def __init__(self, inputs, outputs, hiddens=32):
        super(NonLinearModel, self).__init__()
        self.l1 = nn.Linear(inputs, hiddens)
        self.l2 = nn.Linear(hiddens, outputs)
        self.norm = L2Norm()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.norm(x)
        return x

class LinearModel(nn.Module):

    def __init__(self, inputs, outputs):
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(inputs, outputs)

    def forward(self, x):
        return self.l1(x)

class DynamicModel(nn.Module):
    def __init__(self, encode_size, action_size):
        super(DynamicModel, self).__init__()
        self.encode_size = encode_size
        self.action_size = action_size
        # self.transition = NonLinearModel(self.encode_size, self.encode_size)
        # self.reward = NonLinearModel(self.encode_size, 1)
        self.action_transition_mapping = NonLinearModel(self.action_size, self.encode_size)
        self.action_reward_mapping = NonLinearModel(self.action_size, self.encode_size)
        self.transition = LinearModel(self.encode_size, self.encode_size)
        self.reward = LinearModel(self.encode_size, 1)
        self.norm = L2Norm()

    def forward(self, feature, action): 
        # if len(feature.size()) > 1:
        #     state_action = torch.cat((feature, action), dim=1)
        # else:
        #     state_action = torch.cat((feature, action))
        # predict_next = self.transition(feature)
        # predict_reward = self.reward(feature)

        transition_action = self.action_transition_mapping(action)
        reward_action = self.action_reward_mapping(action)
        predict_next = self.transition(
            torch.multiply(feature, transition_action)
        )
        predict_reward = self.reward(
            torch.multiply(feature, reward_action)
        )
        # n, m = feature.size()
        # predict_reward = torch.bmm(feature.view(n, 1, m), reward_action.view(n, m, 1))
        predict_next = self.norm(predict_next)
        return predict_next, predict_reward

class ComposeModel(nn.Module):
    def __init__(self, inputs, outputs, hiddens=16):
        super(ComposeModel, self).__init__()
        self.encoder = NonLinearModel(inputs, hiddens)
        self.output = NonLinearModel(hiddens, outputs)

    def forward(self, x):
        x = self.encoder(x)
        return self.output(x)

class DeepQLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, state_dim, encode_size=None, name="Deep-Q", 
                alpha=0.1, gamma=0.99, epsilon=0.1, random=False, lr=1e-4,
                explore="uniform", anneal=False, custom_q_init=None, default_q=0, 
                save_dir="../models/deep-q/", save=False, learn_type="source",
                save_interval=None, load_from="dynamics", batch_size=32, epoch=1,
                reward_coeff=1.0):
        '''
        Args:
            actions (list): Contains strings denoting the actions.
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
            custom_q_init (defaultdict{state, defaultdict{action, float}}): a dictionary of dictionaries storing the initial q-values. Can be used for potential shaping (Wiewiora, 2003)
            default_q (float): the default value to initialize every entry in the q-table with [by default, set to 0.0]
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name + name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        self.epsilon, self.epsilon_init = epsilon, epsilon
        self.step_number = 0
        self.anneal = anneal
        self.default_q = default_q # 0 # 1 / (1 - self.gamma)
        self.explore = explore
        self.custom_q_init = custom_q_init

        self.state_dim = state_dim
        self.encode_size = encode_size
        self.save_dir = save_dir
        self.save = save
        self.learn_type = learn_type
        self.load_from = load_from
        self.lr = lr
        self.save_interval = save_interval
        self.normalize = True
        self.random = False
        self.batch_size = batch_size
        self.epoch = epoch
        self.reward_coeff = reward_coeff
        self.tau = 0.1

        self.action_map = {}
        for i in range(len(self.actions)):
            self.action_map[self.actions[i]] = i
        print(self.action_map)

        if not self.encode_size:
            self.encode_size = self.state_dim
        self.encoder = None
            
        self.memory = ReplayMemory(1000000)

        self.set_models()
    
    def reward_setup(self):
        return LinearModel(self.encode_size+len(self.actions), 1)

    def dynamic_setup(self):
        return DynamicModel(self.encode_size, len(self.actions))
        # return LinearModel(self.encode_size+len(self.actions), self.encode_size)
        # return NonLinearModel(self.encode_size+len(self.actions), self.encode_size, hiddens=self.encode_size)
    
    def linear_q_setup(self):
        return LinearModel(self.encode_size, len(self.actions))
    
    def nonlinear_q_setup(self):
        return ComposeModel(self.state_dim, len(self.actions), hiddens=self.encode_size)
    
    def encoder_setup(self):
        return NonLinearModel(self.state_dim, self.encode_size)
    
    def soft_update(self, source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(
                source.parameters(), target.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                torch.add(
                    target_param.data,
                    source_param.data,
                    alpha=tau,
                    out=target_param.data,
                )

    def set_models(self):
        if self.learn_type == "source":
            self.q_model = self.nonlinear_q_setup()
            self.target_model = self.nonlinear_q_setup()
            self.soft_update(self.q_model, self.target_model, 1.0)
            self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)
            print("q model", self.q_model)
            self.dynamics = self.dynamic_setup()
            self.optim_dynamics = optim.Adam(self.dynamics.parameters(), lr=self.lr)
            print("dynamics", self.dynamics)
        
        elif self.learn_type == "dynamics":
            self.dynamics = self.dynamic_setup()
            self.optim_dynamics = optim.Adam(self.dynamics.parameters(), lr=self.lr)
            print("dynamics", self.dynamics)
            self.random = True
            
        elif self.learn_type == "encoder":
            self.encoder = self.encoder_setup()
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
            print("encoder", self.encoder)
            self.dynamics = self.dynamic_setup()
            if self.load_from == "dynamics":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "dynamics.model")))
            elif self.load_from == "source":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "source.model"))["dynamics"])
            # for name, param in self.dynamics.named_parameters():
            #     print(name, param)
            self.random = True

        elif self.learn_type == "transfer":
            self.encoder = self.encoder_setup()
            print("encoder", self.encoder) 
            self.encoder.load_state_dict(torch.load(os.path.join(self.save_dir, "encoder_{}.model".format(self.load_from)))) 
            self.dynamics = self.dynamic_setup()
            if self.load_from == "dynamics":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "dynamics.model")))
            elif self.load_from == "source":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "source.model"))["dynamics"])
            self.q_model = self.linear_q_setup()
            self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)
            print("q model", self.q_model)
        
        elif self.learn_type == "regularize":
            self.q_model = self.nonlinear_q_setup()
            self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)
            print("q model", self.q_model)
            self.dynamics = self.dynamic_setup()
            print("dynamics", self.dynamics)
            if self.load_from == "dynamics":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "dynamics.model")))
            elif self.load_from == "source":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "source.model"))["dynamics"])
            else:
                assert 0, "must specify a pretrained dynamics model"
        
        elif self.learn_type == "pretrain+regularize":
            self.q_model = self.nonlinear_q_setup()
            self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)
            print("q model", self.q_model)
            self.dynamics = self.dynamic_setup()
            if self.load_from == "dynamics":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "dynamics.model")))
            elif self.load_from == "source":
                self.dynamics.load_state_dict(torch.load(os.path.join(self.save_dir, "source.model"))["dynamics"])
            self.q_model.encoder.load_state_dict(torch.load(os.path.join(self.save_dir, "encoder_{}.model".format(self.load_from)))) 
            print("loaded encoder")
            # for name, param in self.q_model.named_parameters():
            #     print(name, param)
        
        elif self.learn_type == "single":
            self.q_model = self.nonlinear_q_setup()
            self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.lr)
            print("q model", self.q_model)
        
        elif self.learn_type == "auxiliary":
            self.q_model = self.nonlinear_q_setup()
            print("q model", self.q_model)
            self.dynamics = self.dynamic_setup()
            print("dynamics", self.dynamics)
            params = list(self.q_model.parameters()) + list(self.dynamics.parameters())
            self.optimizer = optim.Adam(params, lr=self.lr)

        else:
            print("learn type is incorrect")
    
    def create_base_encoder(self):
        observation_examples = numpy.random.rand(10000, self.state_dim).astype('float32') * math.sqrt(self.encode_size)
        # print(observation_examples)
        # self.scaler = sklearn.preprocessing.StandardScaler()
        # self.scaler.fit(observation_examples)

        self.featurizer = RBFSampler(gamma=1.0, n_components=int(self.encode_size))
        self.featurizer.fit(observation_examples)
        # self.featurizer = sklearn.pipeline.FeatureUnion([
        #         ("rbf2", RBFSampler(gamma=2.0, n_components=int(self.encode_size//2))),
        #         ("rbf3", RBFSampler(gamma=1.0, n_components=int(self.encode_size-self.encode_size//2))),
        #         ]) 
        # self.featurizer.fit(self.scaler.transform(observation_examples)) 

    def test_dynamics(self):
        sz = self.encode_size
        for i in range(sz):
            x = numpy.zeros(sz)
            x[i] = 1
            print("\nstate", x)
            x = torch.from_numpy(x).float()
            for action in self.actions:
                print("action", action)
                onehot = F.one_hot(torch.LongTensor([self.action_map[action]]), num_classes=len(self.actions))
                # state_action = torch.cat((x, onehot))
                print("predict", self.dynamics(x.unsqueeze(0), onehot.float())) 
    
    def test_encoder(self):
        n = int(math.sqrt(self.encode_size))
        for i in range(n):
            for j in range(n):
                x = numpy.array([random.random() + i, random.random() + j])
                print("\nstate", x)
                x = torch.from_numpy(x).float()
                encode = self.encoder(x)
                print("encode", encode)
                for action in self.actions:
                    print("action", action)
                    onehot = F.one_hot(torch.LongTensor([self.action_map[action]]), num_classes=len(self.actions))
                    # state_action = torch.cat((encode, onehot))
                    print("predict", self.dynamics(encode.unsqueeze(0), onehot.float())) 

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)

        param_dict["alpha"] = self.alpha
        param_dict["gamma"] = self.gamma
        param_dict["epsilon"] = self.epsilon_init
        param_dict["anneal"] = self.anneal
        param_dict["explore"] = self.explore

        return param_dict

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------

    def act(self, state, reward, learning=True, verbose=False):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)

        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''
        if self.prev_state and self.prev_action:
            self.save2memory(self.prev_state, self.prev_action, state, reward)

        for i in range(self.epoch):
            if self.learn_type == "source":
                self.update()
                self.update_dynamics()
            elif self.learn_type == "dynamics":
                self.update_dynamics()
            elif self.learn_type == "encoder":
                self.update_encoder()
            elif self.learn_type == "regularize" or self.learn_type == "pretrain+regularize":
                self.update_regular()
            elif self.learn_type == "auxiliary":
                self.update_regular()
            else:
                self.update()

        if verbose and self.prev_state and self.prev_action:
            print("state", self.prev_state.encode(), "action", self.prev_action)
            if self.learn_type == "source":
                # print("q", self.q_model(self._get_encode(self.prev_state)))
                print("next encode", self.target_model.encoder(self._get_encode(state)))
                onehot = F.one_hot(torch.LongTensor([self.action_map[self.prev_action]]), num_classes=len(self.actions))
                # state_action = torch.cat((self._get_encode(self.prev_state), onehot))
                pred_next, pred_reward = self.dynamics(self.target_model.encoder(self._get_encode(self.prev_state)).unsqueeze(0), onehot.float())
                print("pred", pred_next, "error", torch.norm(pred_next-self.target_model.encoder(self._get_encode(state))))
                print("pred", pred_reward, "error", torch.norm(pred_reward-reward))
            elif self.learn_type == "dynamics":
                print("next", state.encode())
                onehot = F.one_hot(torch.LongTensor([self.action_map[self.prev_action]]), num_classes=len(self.actions))
                # state_action = torch.cat((self._get_encode(self.prev_state), onehot))
                pred_next, pred_reward = self.dynamics(self._get_encode(self.prev_state).unsqueeze(0), onehot.float())
                print("pred", pred_next, "error", torch.norm(pred_next-self._get_encode(state)))
                print("pred", pred_reward, "error", torch.norm(pred_reward-reward))
            elif self.learn_type == "encoder":
                print("encode", self._get_encode(self.prev_state))
                print("next", state.encode())
                onehot = F.one_hot(torch.LongTensor([self.action_map[self.prev_action]]), num_classes=len(self.actions))
                pred_next, pred_reward = self.dynamics(self._get_encode(self.prev_state).unsqueeze(0), onehot.float())
                print("pred", pred_next, "error", torch.norm(pred_next-self._get_encode(state)))
                print("pred", pred_reward, "error", torch.norm(pred_reward-reward))

        if self.random:
            action = numpy.random.choice(self.actions)
        elif self.explore == "softmax":
            # Softmax exploration
            action = self.soft_max_policy(state)
        else:
            # Uniform exploration
            action = self.epsilon_greedy_q_policy(state)

        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        if learning and self.anneal:
            self._anneal()
        
        if self.save_interval and self.step_number % self.save_interval == 0:
            self._save_checkpoint()

        return action
    
    def save2memory(self, state, action, next_state, reward):
        action = torch.LongTensor([self.action_map[action]])
        reward = torch.FloatTensor([reward])
        self.memory.push(self._get_encode(state, raw=True), action, self._get_encode(next_state, raw=True), reward)

    def _save_checkpoint(self):
        if self.learn_encoder:
            path = os.path.join(self.save_dir, "checkpoints")
            if not os.path.isdir(path):
                os.makedirs(path)
            pickle.dump(self.encoder, open(path + "/encoder_{}.pkl".format(self.step_number), 'wb'))
    
    def _get_encode(self, state, raw=False):
        encode = state.encode()
        encode = torch.from_numpy(encode).float()
        if raw:
            return encode
        elif self.encoder:
            return self.encoder(encode)
        # elif self.learn_type == "source":
        #     return self.q_model.encoder(encode)
        else:
            return encode

    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): action.
        '''
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.

        Returns:
            (str): action.
        '''
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update_dynamics(self):
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size) 
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).view(-1, self.state_dim)
        next_states = torch.cat(batch.next_state).view(-1, self.state_dim)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        onehot = F.one_hot(action_batch, num_classes=len(self.actions)).squeeze()
        # state_action = torch.cat((state_batch, onehot), dim=1)

        if self.learn_type == "source":
            state_batch = self.target_model.encoder(state_batch)
            next_states = self.target_model.encoder(next_states)

        criterion = nn.MSELoss()
        predict_next, predict_reward = self.dynamics(state_batch, onehot.float())

        self.optim_dynamics.zero_grad()
        loss = criterion(predict_next, next_states) + self.reward_coeff * criterion(predict_reward, reward_batch)
        loss.backward()
        self.optim_dynamics.step()

        self.soft_update(self.q_model, self.target_model, self.tau)
    
    def update_encoder(self):
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size) 
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).view(-1, self.state_dim)
        next_states = torch.cat(batch.next_state).view(-1, self.state_dim)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        onehot = F.one_hot(action_batch, num_classes=len(self.actions)).squeeze()
        
        cur_feature = self.encoder(state_batch)
        next_feature = self.encoder(next_states).detach()
        # print("raw state", state_batch)
        # print("raw next", next_states)
        # print("cur encode", cur_feature)
        # print("next encode", next_feature)

        # feature_action = torch.cat((cur_feature, onehot), dim=1)
        predict_next, predict_reward = self.dynamics(cur_feature, onehot.float())
        # print("feature action", feature_action)
        # print("predict next", predict_next)
        # if self.normalize:
        #     predict_norm = numpy.linalg.norm(predict_next)
        #     if predict_norm:
        #         predict_next /= predict_norm
        criterion = nn.MSELoss() 
        self.optimizer.zero_grad()
        loss = criterion(predict_next, next_feature) + self.reward_coeff * criterion(predict_reward, reward_batch)
        loss.backward()
        self.optimizer.step()
        # print("loss", loss)

    def update(self):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size) 
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).view(-1, self.state_dim)
        next_states = torch.cat(batch.next_state).view(-1, self.state_dim)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        # print("state", state_batch)
        # print("next", next_states)
        # print("action", action_batch)
        # print("reward", reward_batch)
        # print("q", self.q_model(state_batch))
        # print("action q", self.q_model(state_batch).gather(1, action_batch))
        # print("next q", self.q_model(next_states))
        if self.encoder:
            state_batch = self.encoder(state_batch)
            next_states = self.encoder(next_states)
            # print("encoded state", state_batch)
            # print("encoded next", next_states)

        # Update the Q Function.
        state_action_values = self.q_model(state_batch).gather(1, action_batch)

        next_state_values = self.q_model(next_states).max(1)[0].detach().unsqueeze(1)
        # print("next state value", next_state_values)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # print("expected", expected_state_action_values)
        # print("action", action_batch)
        # print("value", state_action_values)
        # print("eps", self.epsilon)
        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)
        # print("loss", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("after")
        # print(self.q_model(state_batch).gather(1, action_batch))
    
    def update_regular(self):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size) 
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).view(-1, self.state_dim)
        next_states = torch.cat(batch.next_state).view(-1, self.state_dim)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        onehot = F.one_hot(action_batch, num_classes=len(self.actions)).squeeze()
  
        # Update the Q Function.
        state_action_values = self.q_model(state_batch).gather(1, action_batch)

        next_state_values = self.q_model(next_states).max(1)[0].detach().unsqueeze(1)
        # print("next state value", next_state_values)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.MSELoss()
        q_loss = criterion(state_action_values, expected_state_action_values)

        cur_feature = self.q_model.encoder(state_batch)
        next_feature = self.q_model.encoder(next_states).detach()

        # feature_action = torch.cat((cur_feature, onehot), dim=1)
        predict_next, predict_reward = self.dynamics(cur_feature, onehot.float())  
        
        model_loss = criterion(predict_next, next_feature) + self.reward_coeff * criterion(predict_reward, reward_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss = model_loss + q_loss
        loss.backward()
        self.optimizer.step()

    def _anneal(self):
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        self.alpha = self.alpha_init / (1.0 +  (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )
        self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 1000.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        state = self._get_encode(state)
        return self.q_model(state).max(0)

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self.actions[self._compute_max_qval_action_pair(state)[1].item()]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0].item()

    def get_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        encode = self._get_encode(state)
        return self.q_model(encode)[self.action_map[action]]

    def get_action_distr(self, state, beta=0.2):
        '''
        Args:
            state (State)
            beta (float): Softmax temperature parameter.

        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        all_q_vals = []
        for i, action in enumerate(self.actions):
            all_q_vals.append(self.get_q_value(state, action))

        # Softmax distribution.
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]

        return softmax

    def reset(self):
        self.step_number = 0
        self.episode_number = 0
        if self.save:
            if self.learn_type == "source":
                torch.save({
                    "q": self.q_model.state_dict(),
                    "dynamics": self.dynamics.state_dict()}, 
                    os.path.join(self.save_dir, "source.model")
                )

            elif self.learn_type == "dynamics":
                self.test_dynamics()
                torch.save(
                    self.dynamics.state_dict(), 
                    os.path.join(self.save_dir, "dynamics.model")
                )
            
            elif self.learn_type == "encoder":
                for i in range(100):
                    self.update_encoder()
                self.test_encoder()
                torch.save(
                    self.encoder.state_dict(), 
                    os.path.join(self.save_dir, "encoder_{}.model".format(self.load_from))
                )
        
        self.set_models()
        
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        if self.anneal:
            self._anneal()
        Agent.end_of_episode(self)

    def print_v_func(self):
        '''
        Summary:
            Prints the V function.
        '''
        for state in self.q_func.keys():
            print(state, self.get_value(state))

    def print_q_func(self):
        '''
        Summary:
            Prints the Q function.
        '''
        if len(self.q_func) == 0:
            print("Q Func empty!")
        else:
            for state, actiond in self.q_func.items():
                print(state)
                for action, q_val in actiond.items():
                    print("    ", action, q_val)
