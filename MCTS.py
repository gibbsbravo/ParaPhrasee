"""Monte Carlo Tree Search implementation for both FrozenLake and ParaPhrasee environments"""
"""Source: https://github.com/brilee/python_uct/blob/master/numpy_impl.py"""

import collections
import numpy as np
import math

import torch
from torch.distributions import Categorical
from copy import deepcopy

import data
import config
import model_evaluation

DEVICE = config.DEVICE

#Load data
vocab_index = data.VOCAB_INDEX

# Define environment wrappers as layer between MCTS code and full environments

class ParaPhraseeEnvWrapper():
    def __init__(self, input_env):
        self.env = deepcopy(input_env)
        self.env.max_length = 11
        self.max_steps = 11
        
    def take_action(self, input_state, action):
        self.env.state = input_state
        state, _, terminal, _ = self.env.step(torch.tensor(action, device=DEVICE), self.env.state[1])
        self.env.done = False
        return state, terminal
    
    def get_reward(self):
        return model_evaluation.performance_metrics(
                target_sentence=self.env.target_sentence, pred_sentence=self.env.pred_sentence(), 
                similarity_model=self.env.similarity_model, fluency_model=self.env.fluency_model,
                ESIM_model=self.env.ESIM_model, logr_model=self.env.logr_model, std_scaler=self.env.std_scaler,
                similarity_dist=self.env.similarity_dist, fluency_dist=self.env.fluency_dist, 
                ESIM_dist=self.env.ESIM_dist, vocab_index=vocab_index, metric=self.env.reward_function)

#input_map = frozen_lake_env.generate_random_map(5)
#env = frozen_lake_env.FrozenLakeEnv(input_map, map_frozen_prob=0.75, changing_map=False)
#
def FrozenLake_get_reward(input_map, state):
    if input_map[state] == 'G':
        return 20
    elif input_map[state] == 'H':
        return -10
    else:
        return -1

class FrozenLakeEnvWrapper():
    def __init__(self, input_env):
        self.env = deepcopy(input_env)
        self.max_steps = 25
        
    def take_action(self, input_state, action):
        self.env.state = input_state
        state, _, terminal, _ = self.env.step(action)
        self.env.done = False
        return state, terminal
    
    def get_reward(self, input_state):
        return FrozenLake_get_reward(self.env.input_map, input_state)

#%% MCTS implementation - Builds out search tree """Based heavily on: https://github.com/brilee/python_uct/blob/master/numpy_impl.py"""

class UCTNode():
    """Defines nodes and their properties, as well as code handling the construction of the tree.
        Uses a vectorized implementation in Numpy to improve speed"""
    def __init__(self, state, hidden_state, move, action_space, parent=None, terminal=False):
        self.state = state
        self.hidden_state = hidden_state
        self.move = move
        self.action_space = action_space
        self.is_expanded = False
        self.parent = parent    # Optional[UCTNode]
        self.children = {}    # Dict[move, UCTNode]
        self.child_probs = np.zeros([self.action_space], dtype=np.float32)
        self.child_total_value = np.zeros([self.action_space], dtype=np.float32)
        self.child_number_visits = np.zeros([self.action_space], dtype=np.float32)
        
        # Update terminal condition based on env
        #self.terminal = True if self.move == config.EOS_token else False # For ParaPhrasee
        self.terminal = terminal

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        """Adjusts the value based on the number of visits"""
        return self.child_total_value / (0.01 + self.child_number_visits)
    
    def child_U(self):
        """Calculates the score for determining which node should be explored"""
        return math.sqrt(self.number_visits) * (
                self.child_probs / (0.01 + torch.tensor(self.child_number_visits, device=config.DEVICE)))
    
    def best_child(self):
        """Selects the best child node as main"""
        return torch.argmax(torch.tensor(self.child_Q(), device=config.DEVICE) + self.child_U()).item()

    def select_leaf(self, env_wrapper, actor_model):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(env_wrapper, best_move, actor_model)
        return current

    def expand(self, child_probs):
        self.is_expanded = True
        self.child_probs = child_probs

    def maybe_add_child(self, env_wrapper, move, actor_model):
        """Explores the tree and expands as needed when reaching an unexplored child"""
        if env_wrapper.env.name == 'ParaPhrasee':
            if move not in self.children:
                state, terminal = env_wrapper.take_action((self.state, self.hidden_state), move)
                _, hidden_state = actor_model(self.state, self.hidden_state)
                
                self.children[move] = UCTNode(
                        state[0], hidden_state, move, self.action_space, parent=self, terminal=terminal)
            return self.children[move]
            
        elif env_wrapper.env.name == 'FrozenLake':
            if move not in self.children:
                state, terminal = env_wrapper.take_action(self.state, move)
                _, hidden_state = actor_model(self.state, self.hidden_state)
                
                self.children[move] = UCTNode(
                        state, hidden_state, move, self.action_space, parent=self, terminal=terminal)
            return self.children[move]
        else:
            print('Select either ParaPhrasee or FrozenLake env')

    def backup(self, value_estimate: float):
        """Propogates the estimated score back to the relevant nodes"""
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate)
            current = current.parent

class DummyNode(object):
    """Defines empty node to be used as root"""
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        
def sample_rollout(input_env, actor_model, temperature, input_state, input_hidden_state, max_steps):
    """Randomly uses rollout until reaching terminal node rather than using NN to approximate value"""
    rollout_env = deepcopy(input_env)
    rollout_env.state = input_state 
    
    state = rollout_env.state
    hidden_state = input_hidden_state
    
    ep_env_reward = 0
    #selected_actions = []
    
    for step_i in range(max_steps):
        probs, hidden_state = actor_model(state, hidden_state, temperature)
        m = Categorical(probs)
        action = m.sample().item()
        
        state, env_reward, done, _ = rollout_env.step(action)
        ep_env_reward += env_reward
        #selected_actions.append(action)
        
        if done:
                break
        
    return ep_env_reward

def UCT_search(env, input_state, hidden_state, 
               actor_model, critic_model, temperature, action_space, n_iters):
    """Main function which runs the pipeline for a given root / starting state to 
        produce the MCTS prediction"""
    if env.name == 'ParaPhrasee':
        env_wrapper = ParaPhraseeEnvWrapper(env)
    elif env.name == 'FrozenLake':
        env_wrapper = FrozenLakeEnvWrapper(env)
    else:
        print('Select either ParaPhrasee or FrozenLake env')
    
    root = UCTNode(input_state, hidden_state, move=None, action_space=action_space, 
                   parent=DummyNode(), terminal=False)
    
    for _ in range(n_iters):
        leaf = root.select_leaf(env_wrapper, actor_model)
        
        child_probs = actor_model(leaf.state, leaf.hidden_state, temperature)[0].detach()[0]
        if leaf.terminal:
            if env_wrapper.env.name == 'ParaPhrasee':
                value_estimate = env_wrapper.get_reward()
            elif env_wrapper.env.name == 'FrozenLake':
                value_estimate = env_wrapper.get_reward(leaf.state)
            else:
                print('Select either ParaPhrasee or FrozenLake env')
        else:
            if critic_model is not None:
                value_estimate = critic_model(leaf.state, leaf.hidden_state).detach().item()
            else:
                value_estimate = sample_rollout(env_wrapper.env, actor_model, temperature, leaf.state,
                                                leaf.hidden_state, env_wrapper.max_steps)
        
        leaf.expand(child_probs)
        leaf.backup(value_estimate)
    
    MCTS_action = np.argmax(root.child_number_visits)
    MCTS_hidden_state = root.children[MCTS_action].hidden_state
    
    return MCTS_action, MCTS_hidden_state, root

#%% 