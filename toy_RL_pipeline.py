"""Defines and trains RL models for either CartPole or FrozenLake environments"""

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque
import os
import argparse
from copy import deepcopy

import config
import data
import cart_pole_env
import frozen_lake_env
import utils
import MCTS

DEVICE = config.DEVICE
EPS = config.EPS
torch.manual_seed(config.SEED)

# Define command line arguments for experiment
parser = argparse.ArgumentParser(description='Train_RL_Model')
parser.add_argument('--train_models', action='store_true', help='enable training of RL models')
parser.add_argument('--folder_name', type=str,
                    help='Brief description of experiment (no spaces)')

parser.add_argument('--sparse_env', type=int, choices={0, 1}, default=1,
                    help='sparsifies environment rewards')
parser.add_argument('--relative_rewards', type=int, choices={0, 1}, default=0, 
                    help='use rewards relative to supervised model')

parser.add_argument('--env_name', type=str, default='CartPole',
                    choices=['CartPole', 'FrozenLake'], help='select RL environment')
parser.add_argument('--use_pretrained_critic', type=int, choices={0, 1}, default=1,
                    help='critic is initialized with pretrained model')
parser.add_argument('--pretrain_critic_n_episodes', type=int, default=0,
                    help='number of iterations to pretrain the critic (default: 0)')
parser.add_argument('--n_episodes', type=int, default=2500,
                    help='max number of iterations to train the RL model (default: 2500)')
parser.add_argument('--verbose_training', type=int, choices={0, 1}, default=0,
                    help='print results during training')

parser.add_argument('--init_critic', type=int, choices={0, 1}, default=1, help='initializes critic model')
parser.add_argument('--transfer_weights', type=int, choices={0, 1}, default=1, 
                    help='transfers weights from supervised model to actor')
parser.add_argument('--use_policy_distillation', type=int, choices={0, 1}, default=0, 
                    help='adds policy distillation error to reward function')
parser.add_argument('--MCTS_thresh', type=float, default=0,
                    help='Uses MCTS unless max certainty is above specified prob (default: 0)')

# Mostly for helper functions and debugging
parser.add_argument('--update_RL_models', type=int, choices={0, 1}, default=1, 
                    help='allows the update of the rl models')
parser.add_argument('--use_MLE', type=int, choices={0, 1}, default=0, 
                    help='can use MLE instead of sample')

parser.add_argument('--load_models', action='store_true', help='Load pretrained model from prior point')
parser.add_argument('--load_model_folder_name', type=str,
                    help='folder which contains the saved models to be used')

args = parser.parse_args()
args.save_models = 0

if args.train_models:
    args.save_models = 1
    saved_RL_model_results = data.SaveRLModelResults(args.env_name, args.folder_name)
    saved_RL_model_results.check_folder_exists()

# Specify which encoder / decoder is used
args.FROZENLAKE_ENCODER = 'FrozenLake/FrozenLakeEncoder_medium.pt'
args.FROZENLAKE_DECODER = 'FrozenLake/FrozenLakeDecoder_medium.pt'

args.CARTPOLE_DECODER = 'CartPole/CartpoleDecoder.pt'
args.CP_PRETRAINED_CRITIC = 'CP_critic_model_585.pt'
args.FL_PRETRAINED_CRITIC = 'FL_critic_model_5000.pt'

#%% Manual Testing - turn train models on while keeping save models off then modify as you like

#Run test
#args.train_models = 1
#args.env_name = 'FrozenLake'
#args.verbose_training = 1
#saved_RL_model_results = data.SaveRLModelResults(args.env_name, 'Test')
# 
#args.n_episodes = 3000
#args.load_models = 1
#args.load_model_folder_name = os.path.join(config.saved_RL_model_path, 'FrozenLake/Medium/Test3/Transfer_Weights10/')
#args.update_RL_models = 0
#args.use_MLE = 0
#args.MCTS_thresh = 0.65


#%% Load environment data

def load_FrozenLake_data():
    """Loads FrozenLake training data"""
    input_map_df = data.load_np_data('Data/RL_Data/FL_input_map_df.npy')
    visited_states_df = np.load('Data/RL_Data/FL_states_df.npy', allow_pickle=True)
    selected_actions_df = np.load('Data/RL_Data/FL_selected_action_sequence_df.npy', allow_pickle=True)
    
    train_data, val_data, test_data = data.train_test_split(
            list(zip(input_map_df, visited_states_df, selected_actions_df)))
    
    assert all([len(train_data[i][1]) == len(train_data[i][2]) for i in range(len(train_data))]), \
        'Lengths of states and actions are not equal'
    
    return train_data, val_data, test_data

def load_CartPole_data():
    """Loads CartPole training data"""
    visited_states_df = np.load('Data/RL_Data/CP_states_df.npy')
    selected_actions_df = np.load('Data/RL_Data/CP_selected_action_sequence_df.npy')
    input_map_df = np.zeros((len(visited_states_df),1))
    
    train_data, val_data, test_data = data.train_test_split(
            list(zip(input_map_df, visited_states_df, selected_actions_df)))
    
    assert all([len(train_data[i][1]) == len(train_data[i][2]) for i in range(len(train_data))]), \
        'Lengths of states and actions are not equal'
    
    return train_data, val_data, test_data

def load_env_data(env_name):
    """Initializes data for relevant environment"""
    if env_name == 'CartPole':
        train_data, val_data, test_data = load_CartPole_data()
    
    elif env_name == 'FrozenLake':
        train_data, val_data, test_data = load_FrozenLake_data()
    
    else:
        print("Please select one of the following environments: ['FrozenLake', 'CartPole']")
    
    return train_data, val_data, test_data

train_data, val_data, test_data = load_env_data(args.env_name)

#%% Define supervised models 

class MLPStateEncoder(nn.Module):
    """Uses a MLP state encoder for FrozenLake"""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPStateEncoder, self).__init__()
        self.name = 'MLPStateEncoder'
        self.hidden_size = hidden_size

        self.h1 = nn.Linear(input_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.6)
        self.h2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).view(1,1,-1)
        x = self.h1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.h2(x)
        return action_scores

class CNNStateEncoder(nn.Module):
    """CNN encoder model designed for FrozenLake with a 5x5 input state space.
        Would need to manually update hyperparameters for a different state space size"""
    def __init__(self, output_size):
        super(CNNStateEncoder, self).__init__()
        self.name = 'CNNStateEncoder'
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.linear_output = nn.Linear(144, output_size)
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).view(1,1,5,5)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(1,1,-1)
        output = self.linear_output(x)
        return output

class GeneralDecoderRNN(nn.Module):
    """Vanilla decoder (WITH NO EMBEDDINGS) which decodes based on single context vector"""
    def __init__(self, input_size, hidden_size, output_size):
        super(GeneralDecoderRNN, self).__init__()
        self.name = 'GeneralDecoderRNN'
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = torch.tensor(input, dtype=torch.float32, device=DEVICE).view(1,1,-1)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class TeacherRNN(nn.Module):
    """Vanilla decoder (WITH NO EMBEDDINGS) which decodes based on single context vector 
        Has the same architecture as the General Decoder RNN - the only difference is the addition
        of temperature and softmax instead of log softmax"""
    def __init__(self, input_size, hidden_size, output_size):
        super(TeacherRNN, self).__init__()
        self.name = 'GeneralDecoderRNN'
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden, temperature=1):
        output = torch.tensor(input, dtype=torch.float32, device=DEVICE).view(1,1,-1)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]) / temperature)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

#%% Train and evaluate models

def train_supervised(input_map, visited_states, selected_actions, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion):
    """Train supervised models for CartPole or FrozenLake"""
    if encoder != None:
        encoder.zero_grad()
        hidden_state = encoder(input_map)
    
    else:
        hidden_state = decoder.initHidden()
    
    decoder.zero_grad()
    
    target_length = len(selected_actions)
    loss = 0
    
    for i in range(target_length):
        state_input = visited_states[i]
        model_output, hidden_state = decoder(
            state_input, hidden_state)
        
        loss += criterion(model_output, torch.tensor([selected_actions[i]],
                                                     dtype=torch.long, device=DEVICE))
       
    loss.backward()
    if encoder != None:
        encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def trainIters(input_data, encoder, decoder, 
               encoder_optimizer, decoder_optimizer, n_iters=20, print_every=10):
    """Applies training loop to train models on data"""
    if encoder != None:
        encoder.train()
    decoder.train()
    start = time.time()
    
    print_loss_total = 0  # Reset every print_every
    criterion = nn.NLLLoss()
    
    # Sample n random pairs
    selected_indices = np.random.choice(len(input_data), n_iters, replace=False)
    
    # For EACH pair train model to decrease loss
    for idx, selected_idx in enumerate(selected_indices):
        loss = train_supervised(input_data[selected_idx][0], input_data[selected_idx][1], 
                     input_data[selected_idx][2], encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        
        iter = idx+1
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

def evaluate(input_map, visited_states, selected_actions, encoder, decoder, criterion):
    """Evaluate the performance of the trained models"""
    with torch.no_grad():
        if encoder != None:
            hidden_state = encoder(input_map)
        else:
            hidden_state = decoder.initHidden()
    
        target_length = len(visited_states)
        loss = 0 
        
        for i in range(target_length):
            state_input = visited_states[i]
            model_output, hidden_state = decoder(
                state_input, hidden_state)
            
            loss += criterion(model_output, torch.tensor([selected_actions[i]],
                                                         dtype=torch.long, device=DEVICE))
    return loss.item() / target_length

def validationError(input_data, encoder, decoder, verbose=True):
    """Evalutes the error on a set of input pairs in terms of loss. 
    Is intended to be used on a validation or test set to evaluate performance"""
    if encoder !=  None:
        encoder.eval()
    
    decoder.eval()
    criterion = nn.NLLLoss()
    loss = 0
    
    for selected_idx in range(len(input_data)):
        pair_loss = evaluate(input_data[selected_idx][0], input_data[selected_idx][1], 
                     input_data[selected_idx][2],encoder, decoder, criterion)
        loss += pair_loss
        
    avg_loss = loss / len(input_data)
    
    if verbose:
        print('The average validation loss is {:.3} based on {} samples'.format(avg_loss, len(input_data)))
    return avg_loss

# %% Train supervised model

# Define encoder / decoder models
def train_CartPole_supervised_models():
    """Trains the CartPole supervised model"""
    encoder = None
    encoder_optimizer = None
    decoder = GeneralDecoderRNN(input_size=4, hidden_size=128, output_size=2).to(DEVICE)
    
    # Set optimizer
    decoder_optimizer = optim.Adam(decoder.parameters())
    
    n_iterations = 5
    
    for i in range(n_iterations):
        print('Iteration Number: {}'.format(i))
        trainIters(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   n_iters=50, print_every=10)
        
        validationError(val_data[:10], encoder, decoder)
    
    return encoder, decoder

def train_FrozenLake_supervised_models():
    """Trains the FrozenLake supervised model"""
    encoder = CNNStateEncoder(128).to(DEVICE)
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder = GeneralDecoderRNN(input_size=2, hidden_size=128, output_size=4).to(DEVICE)
    
    # Set optimizer
    decoder_optimizer = optim.Adam(decoder.parameters())
    
    n_iterations = 5
    
    for i in range(n_iterations):
        print('Iteration Number: {}'.format(i))
        trainIters(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   n_iters=2500, print_every=500)
        
        validationError(val_data[:10], encoder, decoder)
    
    return encoder, decoder

def supervised_model_reward(input_env, start_state, supervised_encoder, supervised_model):
    """Returns the baseline reward of running the supervised model"""
    supervised_env = deepcopy(input_env)
    supervised_env.state = start_state
    supervised_env.done, supervised_env.ep_reward = False, 0
    state = supervised_env.state
    
    if supervised_env.name == 'FrozenLake':
        masked_input_map = frozen_lake_env.mask_map(supervised_env.input_map, flatten=False)
        hidden_state = supervised_encoder(masked_input_map).detach()
    
    else:
        hidden_state = supervised_model.initHidden()
    
    for i in range(env.max_steps):
        probs, hidden_state = supervised_model(state, hidden_state)
        
        _, topi = probs.data.topk(1)
        action = topi.squeeze().item()
        
        state, env_reward, done, _ = supervised_env.step(action)
        
        if done:
            break
    return supervised_env.ep_reward

#data.save_model(decoder, os.path.join(config.saved_RL_model_path, 'CartPole/CartpoleDecoder.pt'))
#data.save_model(encoder, os.path.join(config.saved_RL_model_path, 'FrozenLake/FrozenLakeEncoder_superexpert.pt'))
#data.save_model(decoder, os.path.join(config.saved_RL_model_path, 'FrozenLake/FrozenLakeDecoder_superexpert.pt'))

#%% Define reinforcement learning models

class RLActor(nn.Module):
    """Defines the RL Actor model"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RLActor, self).__init__()
        self.name = 'RL actor model'
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, self.hidden_size) # Add state space
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        self.gamma = 0.9999
        self.saved_action_values = []
        self.rewards = []

    def forward(self, input_state, hidden, temperature=1):
        input_state = torch.tensor(input_state, dtype=torch.float32, device=DEVICE).view(1,1,-1)
        output, hidden = self.gru(input_state, hidden)
        output = self.softmax(self.out(output[0]) / temperature)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class RLCritic(nn.Module):
    """Defines the RL Critic model"""
    def __init__(self, input_size, hidden_size):
        super(RLCritic, self).__init__()
        self.name = 'RL critic model'
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(input_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, 1)
        
        self.saved_state_values = []

    def forward(self, state_input, hidden):
        state_input = torch.tensor(state_input, dtype=torch.float32, device=DEVICE).view(1,1,-1)
        x = torch.cat((state_input[0], hidden[0]), 1)
        x = self.h1(x)
        x = F.relu(x)
        value_score = self.h2(x)
        return value_score
    
    #optionally use dropout in the network
    #self.dropout = nn.Dropout(p=0.6)
    #x = self.dropout(x)

def select_action(input_state, input_hidden_state, actor_model, critic_model=None,
                  teacher_model=None, K=1, use_MLE=False, MCTS_thresh=0):
    """Applies the model on a given input and hidden state to make a prediction of which action to take
        Can use MLE, MCTS, or sampling to select an action"""
    probs, hidden_state = actor_model(input_state, input_hidden_state)
    m = Categorical(probs)
    
    # Use MLE instead of sampling distribution
    if use_MLE:
        _, topi = probs.data.topk(1)
        action = topi.squeeze()
        
    elif torch.max(probs).detach() < MCTS_thresh:
        action, hidden_state, _ = MCTS.UCT_search(
                env, input_state, input_hidden_state, actor_model, critic_model,
                5, env.action_space, 1000)
        action = torch.tensor(action, device=config.DEVICE)
    
    else:
        action = m.sample()
    
    actor_model.saved_action_values.append(m.log_prob(action))
    
    if critic_model != None:
        state_value = critic_model(input_state, input_hidden_state)
        critic_model.saved_state_values.append(state_value)
        
    if teacher_model != None:
        # Add policy distillation error
        actor_probs, _ = actor_model(input_state, input_hidden_state, K)
        supervised_probs, _ = teacher_model(input_state, input_hidden_state, K)
        KL_error = utils.KL_divergence(actor_probs, supervised_probs, K)
        return action.item(), hidden_state, KL_error.item()

    return action.item(), hidden_state, None

def REINFORCE_update(actor_model, actor_optimizer):
    """Update the model when using REINFORCE instead of Actor-Critic"""
    R = 0
    policy_loss = []
    returns = []
    
    # Discount the rewards back to present
    for r in actor_model.rewards[::-1]:
        R = r + actor_model.gamma * R
        returns.insert(0, R)
    
    # Scale the rewards
    returns = torch.tensor(returns)
    ###returns = (returns - returns.mean()) / (returns.std() + EPS)
    
    # Calculate the loss 
    for log_prob, R in zip(actor_model.saved_action_values, returns):
        policy_loss.append(-log_prob * R)
    
    # Update network weights
    actor_optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    actor_optimizer.step()
    
    # Clear memory
    del actor_model.rewards[:]
    del actor_model.saved_action_values[:]

def actor_critic_update(actor_model, actor_optimizer, critic_model, critic_optimizer,
                        only_update_critic=False):
    """Update the model when using Actor-Critic"""
    R = 0
    saved_actions = actor_model.saved_action_values
    saved_states = critic_model.saved_state_values
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in actor_model.rewards[::-1]:
        # calculate the discounted value
        R = r + actor_model.gamma * R
        returns.insert(0, R)

    # Scale the rewards
    returns = torch.tensor(returns)
    ###returns = (returns - returns.mean()) / (returns.std() + EPS) # scaling reduced performance

    for log_prob, value, R in zip(saved_actions, saved_states, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R], device=DEVICE)))

    # reset gradients
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    if only_update_critic:
        loss = torch.stack(value_losses).sum()
    
        # perform backprop
        loss.backward()
        critic_optimizer.step()
        
    else:
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
        # perform backprop
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    # reset rewards and action buffer
    del actor_model.rewards[:]
    del actor_model.saved_action_values[:]
    del critic_model.saved_state_values[:]

# %% Define environment, models and transfer weights

def load_FrozenLake_models():
    """Instantiate supervised model and load saved weights (CNN model)"""
    supervised_encoder = CNNStateEncoder(128).to(DEVICE)
    supervised_model = GeneralDecoderRNN(input_size=2, hidden_size=128, output_size=4).to(DEVICE)
    
    data.load_model(supervised_encoder, os.path.join(
            config.saved_RL_model_path, args.FROZENLAKE_ENCODER))
    data.load_model(supervised_model, os.path.join(
            config.saved_RL_model_path, args.FROZENLAKE_DECODER))
    
    return supervised_encoder, supervised_model

def load_CartPole_models():
    """Instantiate supervised model and load saved weights"""
    supervised_encoder = None
    supervised_model = GeneralDecoderRNN(input_size=4, hidden_size=128, output_size=2).to(DEVICE)
    
    data.load_model(supervised_model, os.path.join(config.saved_RL_model_path, args.CARTPOLE_DECODER))
    
    return supervised_encoder, supervised_model

def create_environment(env_name):
    """Instantiates the environment with the selected hyperparameters"""
    if env_name == 'FrozenLake':
        input_map = frozen_lake_env.generate_random_map(5)
        env = frozen_lake_env.FrozenLakeEnv(input_map, map_frozen_prob=0.75,
                                            sparse=args.sparse_env, changing_map=True) 
    
    elif env_name == 'CartPole':
        env = cart_pole_env.CartPoleEnv(args.sparse_env)
    
    else:
        print("Please select one of the following environments: ['FrozenLake', 'CartPole']")
    
    env.seed(config.SEED)
    return env

def init_actor_critic_models(state_space, action_space, init_critic=True, 
                             transfer_weights=True, supervised_model=None):
    """Instantiates the actor and critic models as well as the optimizers"""
    # Define actor and critic
    actor_model = RLActor(input_size=state_space, hidden_size=128, output_size=action_space).to(DEVICE)
    
    # Transfer weights to actor and set optimizer
    if transfer_weights: 
        actor_model.load_state_dict(supervised_model.state_dict())
    
    actor_optimizer = optim.Adam(actor_model.parameters())
    
    if init_critic:
        critic_model = RLCritic(input_size=(state_space + actor_model.hidden_size),
                                hidden_size=128).to(DEVICE)
        critic_optimizer = optim.Adam(critic_model.parameters())
    
    else:
        critic_model = None
        critic_optimizer = None
        
    return actor_model, critic_model, actor_optimizer, critic_optimizer

#%% Load models and environment

def init_RL_environment(env_name, init_critic=True, transfer_weights=True):
    """Instantiates the RL environment and models through using the subfunctions"""
    
    assert env_name in ['FrozenLake', 'CartPole'], \
        "Please select one of the following environments: ['FrozenLake', 'CartPole']"
    
    # Load supervised models
    if env_name == 'FrozenLake':
        supervised_encoder, supervised_model = load_FrozenLake_models()        
    elif env_name == 'CartPole':
        supervised_encoder, supervised_model = load_CartPole_models()
    
    # Create environment
    env = create_environment(env_name)
    
    # Create environment
    actor_model, critic_model, actor_optimizer, critic_optimizer = init_actor_critic_models(
            state_space=env.state_space, action_space=env.action_space,
            init_critic=init_critic, transfer_weights=transfer_weights, supervised_model=supervised_model)
    
    return supervised_encoder, supervised_model, env, \
            actor_model, critic_model, actor_optimizer, critic_optimizer

class HyperParams(object):
    """Sets the experiment hyperparameters"""
    def __init__(self, env_name, n_episodes):
        assert env_name in ['FrozenLake', 'CartPole'], \
        "Please select one of the following environments: ['FrozenLake', 'CartPole']"
        
        if env_name == 'CartPole':
            self.print_every = 5
            self.t_weight = utils.EpsilonDecay(1, 0, n_episodes, 'Linear') # only Linear works
            self.K = 5
            self.early_stopping_reward_thresh = 295
            self.early_stopping_n_mean = 3
            
            self.beta = 0.25
            self.sm_baseline_reward = 250
            self.distillation_n_mean = 20
        
        elif env_name == 'FrozenLake':
            self.print_every = 20
            self.t_weight = utils.EpsilonDecay(1, 0, n_episodes, 'Linear') # only Linear works
            self.K = 5
            self.early_stopping_reward_thresh = 11
            self.early_stopping_n_mean = 50
            
            self.beta = 1
            self.sm_baseline_reward = 3.5
            self.distillation_n_mean = 500

#%% Training loop

def train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                    supervised_encoder, supervised_model, teacher_model,
                    use_policy_distillation, update_RL_models, only_update_critic, use_MLE, MCTS_thresh, n_episodes):
    """Main training loop to train the actor and critic models"""
    
    for i_episode in range(1, n_episodes+1):
        # Reset environment at beginning of episode
        state, ep_reward, done = env.reset(), 0, False
        
        ep_env_reward = 0
        ep_KL_penalty = 0
        start_state = deepcopy(state)
        
        if env.name == 'FrozenLake':
            masked_input_map = frozen_lake_env.mask_map(env.input_map, flatten=False)
            hidden_state = supervised_encoder(masked_input_map).detach()
        
        else:
            hidden_state = actor_model.initHidden()
        
        # Play environment until maximum number of steps or environment terminates
        for step_i in range(1, env.max_steps+1):
            # Select action
            action, hidden_state, KL_error = select_action(input_state=state, input_hidden_state=hidden_state,
                                                           actor_model=actor_model, critic_model=critic_model,
                                                           teacher_model=teacher_model, K=hp.K, use_MLE=use_MLE,
                                                           MCTS_thresh=MCTS_thresh)
            
            # Apply action to environment to transition to next step
            state, env_reward, done, _ = env.step(action)
            
            # Can optionally use relative rewards to supervised model as reward
            if args.sparse_env and args.relative_rewards and done:
                env_reward = env_reward - supervised_model_reward(
                        env, start_state, supervised_encoder, supervised_model)
            
            # Can optionally use policy distillation
            if use_policy_distillation:
                avg_RL_reward = np.mean(saved_RL_model_results.env_rewards[-hp.distillation_n_mean:]) \
                        if len(saved_RL_model_results.env_rewards) > hp.distillation_n_mean else 0
                        
                lambda_value= utils.lambda_value(beta=hp.beta, sm_baseline_reward=hp.sm_baseline_reward,
                                             avg_rewards=avg_RL_reward)
                KL_penalty = lambda_value * -KL_error
                
                reward = env_reward + KL_penalty
                ep_env_reward += env_reward
                ep_KL_penalty += KL_penalty
            
            else:
                reward = env_reward
                ep_reward += reward
            
            actor_model.rewards.append(reward)
            
            if done:
                break
        
        # Update models based on reward performance
        if update_RL_models:
            if critic_model != None:
                actor_critic_update(actor_model, actor_optimizer, critic_model, critic_optimizer,
                                only_update_critic=only_update_critic)
            else:
                REINFORCE_update(actor_model, actor_optimizer)
        
        
        if use_policy_distillation:
            saved_RL_model_results.env_rewards.append(ep_env_reward)
            saved_RL_model_results.KL_penalty.append(ep_KL_penalty)
            
            if args.verbose_training and (i_episode % hp.print_every == 0):
                avg_env_reward = np.mean(saved_RL_model_results.env_rewards[-hp.print_every:])
                avg_KL_penalty = np.mean(saved_RL_model_results.KL_penalty[-hp.print_every:])
                
                print('Episode {} | Avg env reward: {:.2f} | Avg KL penalty: {:.2f} | Lambda value: {:.2f}'.format(
                      i_episode, avg_env_reward, avg_KL_penalty, lambda_value))
        
            early_stopping_value = np.mean(saved_RL_model_results.env_rewards[-hp.early_stopping_n_mean:]) \
                    if len(saved_RL_model_results.env_rewards) > hp.early_stopping_n_mean else 0
            if (early_stopping_value >= hp.early_stopping_reward_thresh) or (i_episode == n_episodes-1):
                if args.save_models:
                    saved_RL_model_results.save_top_models(actor_model, 'actor_{:.1f}.pt'.format(early_stopping_value))
                    if args.init_critic:
                        saved_RL_model_results.save_top_models(critic_model, 'critic_{:.1f}.pt'.format(early_stopping_value))
                    saved_RL_model_results.export_rewards('model_performance.txt')
                break
            
        else:
            saved_RL_model_results.env_rewards.append(ep_reward)
            
            if args.verbose_training and (i_episode % hp.print_every == 0):
                avg_env_reward = np.mean(saved_RL_model_results.env_rewards[-hp.print_every:])
                print('Episode {} | Average reward: {:.2f}'.format(i_episode, avg_env_reward))
            
            early_stopping_value = np.mean(saved_RL_model_results.env_rewards[-hp.early_stopping_n_mean:]) \
                    if len(saved_RL_model_results.env_rewards) > hp.early_stopping_n_mean else 0
            if (early_stopping_value >= hp.early_stopping_reward_thresh) or (i_episode == n_episodes-1):
                if args.save_models:
                    saved_RL_model_results.save_top_models(actor_model, 'actor_{:.1f}.pt'.format(early_stopping_value))
                    if args.init_critic:
                        saved_RL_model_results.save_top_models(critic_model, 'critic_{:.1f}.pt'.format(early_stopping_value))
                    saved_RL_model_results.export_rewards('model_performance.txt')
                break

def load_RL_models(folder_name, actor_file_name='best', critic_file_name='best'):
    """Instantiate RL models and load trained weights"""
    actor_model = RLActor(input_size=env.state_space, hidden_size=128, output_size=env.action_space).to(DEVICE)
    
    if actor_file_name == 'best':
        actor_file_name = 'actor_{:.1f}.pt'.format(
                data.get_top_n_models(
                        os.path.join(config.saved_RL_model_path, args.env_name, folder_name), 'actor', n=1)[0])
    
    data.load_model(actor_model, os.path.join(config.saved_RL_model_path, args.env_name,
                                              folder_name, actor_file_name))
    
    if args.init_critic:
        critic_model = RLCritic(input_size=(env.state_space + actor_model.hidden_size),
                                    hidden_size=128).to(DEVICE)
        
        if critic_file_name == 'best':
            critic_file_name = 'critic_{:.1f}.pt'.format(
                    data.get_top_n_models(
                            os.path.join(config.saved_RL_model_path, args.env_name, folder_name), 'critic', n=1)[0])
            
        data.load_model(critic_model, os.path.join(config.saved_RL_model_path, args.env_name,
                                                   folder_name, critic_file_name))
    
        return actor_model, critic_model
    
    else:
        return actor_model, None

def load_pretrained_critic(env_name):
    """Intantial RL critic and load trained weights"""
    if env_name =='CartPole':
        data.load_model(critic_model, os.path.join(config.saved_RL_model_path, env_name,
                                                   args.CP_PRETRAINED_CRITIC))
    elif env_name =='FrozenLake':
        data.load_model(critic_model, os.path.join(config.saved_RL_model_path, env_name,
                                                   args.FL_PRETRAINED_CRITIC))
    else:
        print("Please select one of the following environments: ['FrozenLake', 'CartPole']")

#%% Train and Evaluate Model

if args.train_models:
    """Instantiates models and environment, trains and evaluates the model"""
    
    # Instantiate models and environment 
    supervised_encoder, supervised_model, env, \
            actor_model, critic_model, actor_optimizer, critic_optimizer = init_RL_environment(
                    env_name=args.env_name, init_critic=args.init_critic, transfer_weights=args.transfer_weights)
    
    # Create folder if saving models
    if args.save_models:
        saved_RL_model_results.init_folder(args, actor_model, critic_model)
    
    # Optionally load trained models
    if args.load_models:
        actor_model, critic_model = load_RL_models(args.load_model_folder_name)
    
    # Instantiate teacher model if using policy distillation
    if args.use_policy_distillation:
        teacher_model = TeacherRNN(input_size=env.state_space, hidden_size=128,
                                   output_size=env.action_space).to(DEVICE)
        teacher_model.load_state_dict(supervised_model.state_dict())    
    else:
        teacher_model = None
    
    # Load pretrained critic
    if args.use_pretrained_critic and critic_model is not None:
        load_pretrained_critic(args.env_name)

    # Optionally pretrain critic
    if args.pretrain_critic_n_episodes > 0:
        hp = HyperParams(args.env_name, args.pretrain_critic_n_episodes)
        
        train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                        supervised_encoder, supervised_model, teacher_model,
                        use_policy_distillation=False, update_RL_models=True, 
                        only_update_critic=True, use_MLE=False, MCTS_thresh=0,
                        n_episodes=args.pretrain_critic_n_episodes)
    
    # Instantiate hyperparameters
    hp = HyperParams(args.env_name, args.n_episodes)
    
    # Train models
    train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                        supervised_encoder, supervised_model, teacher_model,
                        use_policy_distillation=args.use_policy_distillation, 
                        update_RL_models=args.update_RL_models,
                        only_update_critic=False, use_MLE=args.use_MLE,
                        MCTS_thresh=args.MCTS_thresh, n_episodes=args.n_episodes)
