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

# Other imports.
from simple_rl.agents.AgentClass import Agent

class LinearQLearningAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, actions, state_dim, encode_size=None, name="Linear-Q", 
                alpha=0.1, gamma=0.99, epsilon=0.1, random=False,
                explore="uniform", anneal=False, custom_q_init=None, default_q=0, 
                save_dir="../models/linear-q/", save=False, load=False, learn=True,
                learn_dynamics=False, load_dynamics=False, learn_encoder=False, 
                load_encoder=False, baseline_encoder=False, save_interval=None,
                checkpoint=None):
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

        self.random = random
        self.state_dim = state_dim
        self.encode_size = encode_size
        self.save_dir = save_dir
        self.save = save
        self.learn = learn
        self.learn_dynamics = learn_dynamics
        self.learn_encoder = learn_encoder
        self.normalize = True
        self.baseline_encoder = baseline_encoder
        self.save_interval = save_interval

        # Q Function:
        # if self.custom_q_init:
        #     self.q_func = self.custom_q_init
        # else:
        #     self.q_func = defaultdict(lambda: defaultdict(lambda: self.default_q))
        if load:
            self.q_models = pickle.load(open(save_dir+"models.pkl", 'rb'))
            print("loading")
        else:
            self.q_models = {}
            sz = self.encode_size if self.encode_size else self.state_dim
            for action in self.actions:
                model = SGDRegressor(learning_rate="constant")
                model.partial_fit([numpy.zeros(sz)], [0])
                self.q_models[action] = model
        
        if load_dynamics:
            self.dynamics_regressors = pickle.load(open(save_dir+"dynamics.pkl", 'rb'))
            print("loading dynamics")
        elif self.learn_dynamics:
            self.dynamics_regressors = {}
            for action in self.actions:
                self.dynamics_regressors[action]=MultiOutputRegressor(SGDRegressor(random_state=0))
        else:
            self.dynamics_regressors = None

        if load_encoder:
            if checkpoint:
                self.encoder = pickle.load(open(save_dir+checkpoint, 'rb'))
                print("loading encoder", checkpoint)
            else:
                self.encoder = pickle.load(open(save_dir+"encoder.pkl", 'rb'))
                print("loading encoder")
        elif self.learn_encoder:
            self.encoder = MultiOutputRegressor(SGDRegressor(random_state=0))
            self.encoder.partial_fit([numpy.zeros(state_dim)], [numpy.zeros(encode_size)])
        elif self.baseline_encoder:
            self.create_base_encoder()
        else:
            self.encoder = None
    
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
        sz = self.encode_size if self.encode_size else self.state_dim
        for i in range(sz):
            x = numpy.zeros(sz)
            x[i] = 1
            print("\nx", x)
            for action in self.actions:
                print("action", action)
                predict = self.dynamics_regressors[action].predict([x])[0]
                print("predict", predict)

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
        # print("state", state)
        if learning and self.learn:
            self.update(self.prev_state, self.prev_action, reward, state)
        if learning and self.learn_dynamics:
            self.update_dynamics(self.prev_state, self.prev_action, reward, state)
        if learning and self.learn_encoder:
            self.update_encoder(self.prev_state, self.prev_action, reward, state)
        if verbose and self.prev_state and self.prev_action:
            print("state", self.prev_state.encode(), "action", self.prev_action)
            if self.baseline_encoder:
                print("encode", self._get_encode(state))
            elif self.encoder:
                encode = self.encoder.predict([self.prev_state.encode()])[0]
                print("encode", encode)
                predict = self.dynamics_regressors[self.prev_action].predict([encode])[0]
                print("next", self.encoder.predict([state.encode()])[0], "predict", predict)
            elif self.dynamics_regressors:
                predict = self.dynamics_regressors[self.prev_action].predict([self.prev_state.encode()])[0]
                print("next", state, "predict", predict)

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
    
    def _save_checkpoint(self):
        if self.learn_encoder:
            path = os.path.join(self.save_dir, "checkpoints")
            if not os.path.isdir(path):
                os.makedirs(path)
            pickle.dump(self.encoder, open(path + "/encoder_{}.pkl".format(self.step_number), 'wb'))
    
    def _get_encode(self, state):
        if self.baseline_encoder:
            encode = self.featurizer.transform([state])[0]
            # scaled = self.scaler.transform([state])
            # encode = self.featurizer.transform(scaled)[0]
        elif self.encoder:
            encode = self.encoder.predict([state.encode()])[0]
        else:
            encode = state.encode()
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

    def update_dynamics(self, state, action, reward, next_state):
        if not action or not state:
            return
        self.dynamics_regressors[action].partial_fit([state.encode()], [next_state.encode()])
        # predict = self.dynamics_regressors[action].predict([state.encode()])[0]
        # loss = numpy.linalg.norm(predict-next_state.encode())
    
    def update_encoder(self, state, action, reward, next_state):
        if not action or not state:
            return
        cur_feature = self.encoder.predict([state.encode()])[0]
        predict_next = self.dynamics_regressors[action].predict([cur_feature])[0]
        if self.normalize:
            predict_norm = numpy.linalg.norm(predict_next)
            if predict_norm:
                predict_next /= predict_norm
        self.encoder.partial_fit([next_state.encode()], [predict_next])
        # encoded_next = self.encoder.predict([next_state.encode()])[0]
        # loss = numpy.linalg.norm(predict_next-encoded_next)
        # print("loss", loss)

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
        if state is None:
            self.prev_state = next_state
            return

        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        y = reward + self.gamma*max_q_curr_state
        # print("state id", state.id, "y", y)
        # prev_q_val = self.get_q_value(state, action)
        # print("prev q", prev_q_val)
        encode = self._get_encode(state)
        self.q_models[action].partial_fit([encode], [y])
        # q_val = self.get_q_value(state, action)
        # print("curr q", q_val)
        # self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)
        

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
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        # Find best action (action w/ current max predicted Q value)
        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)

        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)

        Returns:
            (float): denoting the max q value in the given @state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

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
        # state_encoding = numpy.zeros(self.state_dim)
        # state_encoding[state.id] = 1
        encode = self._get_encode(state)
        return self.q_models[action].predict([encode])[0]
        # return self.q_func[state][action]

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
            pickle.dump(self.q_models, open(self.save_dir+"models.pkl", 'wb'))
        if self.learn_dynamics:
            pickle.dump(self.dynamics_regressors, open(self.save_dir+"dynamics.pkl", 'wb'))
        if self.learn_encoder:
            pickle.dump(self.encoder, open(self.save_dir+"encoder.pkl", 'wb'))
        
        self.q_models = {}
        sz = self.encode_size if self.encode_size else self.state_dim
        for action in self.actions:
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([numpy.zeros(sz)], [0])
            self.q_models[action] = model
        # if self.custom_q_init:
        #     self.q_func = self.custom_q_init
        # else:
        #     self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
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
