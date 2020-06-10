"""Defines and trains RL models for ParaPhrasee environment"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import os
import argparse
import logging

import config
import data
import utils
import supervised_model as sm
from train_ESIM import RLAdversary, load_ESIM_model
import paraphrasee_env

import MCTS

DEVICE = config.DEVICE

MAX_LENGTH = config.MAX_LENGTH

SOS_token = config.SOS_token
EOS_token = config.EOS_token

#Load data
train_pairs = data.TRAIN_PAIRS
val_pairs = data.VAL_PAIRS
test_pairs = data.TEST_PAIRS
vocab_index = data.VOCAB_INDEX

# Define command line arguments for experiment
parser = argparse.ArgumentParser(description='Train_ParaPhrasee_Model')
parser.add_argument('--train_models', action='store_true', help='enable training of RL models')
parser.add_argument('--test_MCTS', action='store_true', help='enable testing of RL models')
parser.add_argument('--folder_name', type=str,
                    help='Brief description of experiment (no spaces)')
parser.add_argument('--checkpoint_n_episodes', type=int, default=5000,
                    help='Brief description of experiment (no spaces)')
parser.add_argument('--reward_function', type=str, default='BLEU1',
                    choices=config.perf_metrics_list,
                    help='select reward function')
parser.add_argument('--similarity_model_name', type=str, default='BERT',
                    choices=['BERT', 'InferSent'],
                    help='select reward function')

parser.add_argument('--use_pretrained_critic', type=int, choices={0, 1}, default=1,
                    help='critic is initialized with pretrained model')
parser.add_argument('--pretrain_critic_n_episodes', type=int, default=0,
                    help='number of iterations to pretrain the critic (default: 0)')
parser.add_argument('--n_episodes', type=int, default=30000,
                    help='max number of iterations to train the RL model (default: 2500)')
parser.add_argument('--verbose', action='store_true', help='print results during training')

parser.add_argument('--init_critic', type=int, choices={0, 1}, default=1, help='initializes critic model')
parser.add_argument('--transfer_weights', type=int, choices={0, 1}, default=1, 
                    help='transfers weights from supervised model to actor')
parser.add_argument('--use_policy_distillation', type=int, choices={0, 1}, default=0, 
                    help='adds policy distillation error to reward function')
parser.add_argument('--MCTS_thresh', type=float, default=0,
                    help='Uses MCTS unless max certainty is above specified prob (default: 0)')
parser.add_argument('--use_adversarial_training', type=int, choices={0, 1}, default=0, 
                    help='update adversary')

# Mostly for helper functions and debugging
parser.add_argument('--update_RL_models', type=int, choices={0, 1}, default=1, 
                    help='allows the update of the rl models')
parser.add_argument('--use_MLE', type=int, choices={0, 1}, default=0, 
                    help='can use MLE instead of sample')

parser.add_argument('--load_models', action='store_true', help='Load pretrained model from prior point')
parser.add_argument('--load_model_folder_name', type=str,
                    help='folder which contains the saved models to be used')

args = parser.parse_args()
args.env_name = 'ParaPhrasee'
args.save_models = 0

if args.train_models:
    args.save_models = 1
    saved_RL_model_results = data.SaveRLModelResults(args.env_name, args.folder_name)
    saved_RL_model_results.check_folder_exists()

args.SM_FOLDER = 'VanillaEncoder'
args.SM_ENCODER_FILE_NAME = 'encoder_3.150.pt'
args.SM_DECODER_FILE_NAME = 'decoder_3.150.pt'

args.PRETRAINED_CRITIC = args.reward_function+'_pretrained_critic_125k.pt'

def set_early_stopping_thresh(reward_function):
    if reward_function in config.perf_metrics_list:
        return 10.00
    else:
        print("Please specify one of the following reward functions: {}".format(
                config.perf_metrics_list))

args.early_stopping_reward_thresh = set_early_stopping_thresh(args.reward_function)

#%%
#args.train_models = 1
#args.verbose = 1
#saved_RL_model_results = data.SaveRLModelResults('ParaPhrasee', 'Test')
#args.reward_function = 'BLEU1'

#args.update_RL_models = 0
#args.MCTS_thresh = 0.90

#%% 

class RLActor(nn.Module):
    """Vanilla decoder which decodes based on single context vector"""
    def __init__(self, embedding_size, hidden_size, output_size):
        super(RLActor, self).__init__()
        self.name = 'VanillaDecoderRNNActor'
        self.is_agent = True
        self.uses_attention = False
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        self.gamma = 0.9999
        self.saved_action_values = []
        self.rewards = []

    def forward(self, input, hidden, temperature=1):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]) / temperature)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class RLCritic(nn.Module):
    """Critic which predicts the value of a given state"""
    def __init__(self, embedding_size, hidden_size, output_size):
        super(RLCritic, self).__init__()
        self.name = 'CriticRNN'
        self.is_agent = True
        self.uses_attention = False
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        
        self.saved_state_values = []

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class TeacherRNN(nn.Module):
    """Vanilla decoder which decodes based on single context vector,
        Has the same architecture as the DecoderRNN - the only difference is the addition
        of temperature and softmax instead of log softmax"""
    def __init__(self, embedding_size, hidden_size, output_size):
        super(TeacherRNN, self).__init__()
        self.name = 'VanillaDecoderRNN'
        self.is_agent = False
        self.uses_attention = False
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden, temperature=1):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]) / temperature)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

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
    
    # Note: MCTS only works during validation (when the model is not tracking gradients)
    elif torch.max(probs).detach() < MCTS_thresh:
        action, hidden_state, _ = MCTS.UCT_search(
                env, input_state, input_hidden_state, actor_model, critic_model,
                5, env.action_space, 100)
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
        return action, hidden_state, KL_error.item()

    return action, hidden_state, None


#%% 
    
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

#%% 

def init_actor_critic_models(supervised_decoder, init_critic=True, transfer_weights=True):
    """Instantiates the actor and critic models as well as the optimizers"""
    # Define actor and critic
    actor_model = RLActor(supervised_decoder.embedding_size, supervised_decoder.hidden_size,
                           vocab_index.n_words).to(DEVICE)
    
    # Transfer weights to actor and set optimizer
    if transfer_weights:
        actor_model.load_state_dict(supervised_decoder.state_dict())
    
    actor_optimizer = optim.SGD(actor_model.parameters(), lr=0.001)
    
    if init_critic:
        critic_model = RLCritic(supervised_decoder.embedding_size, supervised_decoder.hidden_size,
                                vocab_index.n_words).to(DEVICE)
        critic_optimizer = optim.SGD(critic_model.parameters(), lr=0.001)
    
    else:
        critic_model = None
        critic_optimizer = None
    
    return actor_model, critic_model, actor_optimizer, critic_optimizer

def train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                    supervised_encoder, teacher_model,
                    use_policy_distillation, update_RL_models, only_update_critic,
                    use_MLE, MCTS_thresh, n_episodes):
    """Main training loop to train the actor and critic models"""
    
    for i_episode in range(1, n_episodes+1):
        (prev_action, hidden_state), ep_reward, done = env.reset(), 0, False
        
        ep_env_reward = 0
        ep_KL_penalty = 0
        
        for step_i in range(1, env.max_steps+1):
            action, hidden_state, KL_error = select_action(input_state=prev_action, input_hidden_state=hidden_state,
                                                           actor_model=actor_model, critic_model=critic_model,
                                                           teacher_model=teacher_model, K=hp.K, use_MLE=use_MLE,
                                                           MCTS_thresh=MCTS_thresh)
                        
            (prev_action, hidden_state), env_reward, done, _ = env.step(action, hidden_state)
            
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
                if args.use_adversarial_training:
                    adversary_model.pred_pairs.append([env.source_sentence, env.pred_sentence()])
                break
        
        if update_RL_models:
            if critic_model != None:
                actor_critic_update(actor_model, actor_optimizer, critic_model, critic_optimizer,
                                only_update_critic=only_update_critic)
            else:
                REINFORCE_update(actor_model, actor_optimizer)
        
        if use_policy_distillation:
            saved_RL_model_results.env_rewards.append(ep_env_reward)
            saved_RL_model_results.KL_penalty.append(ep_KL_penalty)
            
            if args.verbose and (i_episode % hp.print_every == 0):
                avg_env_reward = np.mean(saved_RL_model_results.env_rewards[-hp.print_every:])
                avg_KL_penalty = np.mean(saved_RL_model_results.KL_penalty[-hp.print_every:])
                
                print('Episode {} | Avg env reward: {:.2f} | Avg KL penalty: {:.2f} | Lambda value: {:.2f}'.format(
                      i_episode, avg_env_reward, avg_KL_penalty, lambda_value))
        
            early_stopping_value = np.mean(saved_RL_model_results.env_rewards[-hp.early_stopping_n_mean:]) \
                    if len(saved_RL_model_results.env_rewards) > hp.early_stopping_n_mean else 0
            if (early_stopping_value >= hp.early_stopping_reward_thresh) or (i_episode == n_episodes-1):
                if args.save_models:
                    saved_RL_model_results.save_top_models(actor_model, 'actor_{:.3f}.pt'.format(early_stopping_value))
                    if args.init_critic:
                        saved_RL_model_results.save_top_models(critic_model, 'critic_{:.3f}.pt'.format(early_stopping_value))
                    saved_RL_model_results.export_rewards('model_performance.txt')
                    
                    if args.use_adversarial_training:
                        model_name = 'adversary_model{}_{:.3}.pt'.format(
                            adversary_model.update_iter,
                            adversary_model.training_accuracy[adversary_model.update_iter][-1])
                        data.save_model(adversary_model.model,
                                        os.path.join(saved_RL_model_results.folder_path, model_name))
                break
            
        else:
            saved_RL_model_results.env_rewards.append(ep_reward)
            
            if args.verbose and (i_episode % hp.print_every == 0):
                avg_env_reward = np.mean(saved_RL_model_results.env_rewards[-hp.print_every:])
                print('Episode {} | Average reward: {:.2f}'.format(i_episode, avg_env_reward))
            
            early_stopping_value = np.mean(saved_RL_model_results.env_rewards[-hp.early_stopping_n_mean:]) \
                    if len(saved_RL_model_results.env_rewards) > hp.early_stopping_n_mean else 0
            
            if args.save_models and (i_episode % args.checkpoint_n_episodes == 0):
                saved_RL_model_results.save_top_models(actor_model, 'actor_iter{}_{:.3f}.pt'.format(
                        i_episode, early_stopping_value))
                if args.init_critic:
                    saved_RL_model_results.save_top_models(
                            critic_model, 'critic_iter{}_{:.3f}.pt'.format(i_episode, early_stopping_value))
            
            if (early_stopping_value >= hp.early_stopping_reward_thresh) or (i_episode == n_episodes-1):
                if args.save_models:
                    saved_RL_model_results.save_top_models(actor_model, 'actor_{:.3f}.pt'.format(early_stopping_value))
                    if args.init_critic:
                        saved_RL_model_results.save_top_models(critic_model, 'critic_{:.3f}.pt'.format(early_stopping_value))
                    saved_RL_model_results.export_rewards('model_performance.txt')
                    
                    if args.use_adversarial_training:
                        model_name = 'adversary_model{}_{:.3}.pt'.format(
                            adversary_model.update_iter,
                            adversary_model.training_accuracy[adversary_model.update_iter][-1])
                        data.save_model(adversary_model.model,
                                        os.path.join(saved_RL_model_results.folder_path, model_name))
                break
            
        if (i_episode % hp.update_adversary_every == 0) and args.use_adversarial_training:
            n_target_samples = len(adversary_model.pred_pairs) / 0.7 - len(adversary_model.pred_pairs)
            adversary_model.target_pairs = data.sample_list(env.sentence_pairs, n_samples=int(n_target_samples))
            adversary_model.update_model()
            
            env.ESIM_model = adversary_model.model
            
class HyperParams(object):
    """Sets the experiment hyperparameters"""
    def __init__(self, print_every=10, early_stopping_reward_thresh=0.50):
        self.print_every = print_every
        self.K = 5
        self.early_stopping_reward_thresh = early_stopping_reward_thresh
        self.early_stopping_n_mean = 50
        
        self.beta = 10000
        self.sm_baseline_reward = 0.25
        self.distillation_n_mean = 50
        
        self.update_adversary_every = 6000

#%%

if args.train_models:
    """Instantiates models and environment, trains and evaluates the model"""
    
    # Instantiate models and environment 
    supervised_encoder, supervised_decoder = sm.load_supervised_models(
            args.SM_FOLDER, encoder_file_name=args.SM_ENCODER_FILE_NAME, decoder_file_name=args.SM_DECODER_FILE_NAME)
    actor_model, critic_model, actor_optimizer, critic_optimizer = init_actor_critic_models(
            supervised_decoder, init_critic=args.init_critic, transfer_weights=args.transfer_weights)
    
    # Create folder if saving models
    if args.save_models:
        saved_RL_model_results.init_folder(args, actor_model, critic_model)
    
    # Load pretrained critic
    if args.use_pretrained_critic and critic_model is not None:
        data.load_model(critic_model, os.path.join(config.saved_RL_model_path, args.env_name,
                                                   args.reward_function, args.PRETRAINED_CRITIC))
    # Optionally load trained models
    if args.load_models:
        actor_model, critic_model = data.load_RL_models(
                args.env_name, args.load_model_folder_name, actor_model, critic_model,
                actor_file_name='best', critic_file_name='best')

    # Instantiate teacher model if using policy distillation
    if args.use_policy_distillation:
        teacher_model = TeacherRNN(embedding_size=supervised_decoder.embedding_size,
                                   hidden_size=supervised_decoder.hidden_size,
                                   output_size=vocab_index.n_words).to(DEVICE)
        teacher_model.load_state_dict(supervised_decoder.state_dict())
    else:
        teacher_model = None

    # Load adversarial model if use adversarial training
    if args.use_adversarial_training:
        adversary_model = RLAdversary('ESIM_noisy_3')
    
    # Instantiate the environment and hyperparameters
    input_sentence = train_pairs[0]
    env = paraphrasee_env.ParaPhraseeEnvironment(
            source_sentence=input_sentence[0], target_sentence=input_sentence[1],
            supervised_encoder=supervised_encoder, reward_function=args.reward_function,
            similarity_model_name=args.similarity_model_name, sentence_pairs=train_pairs)
    hp = HyperParams(print_every=10, early_stopping_reward_thresh=args.early_stopping_reward_thresh)
    
    # Train models and save checkpoint if error occurs
    try:
        if args.pretrain_critic_n_episodes > 0:
            train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                            supervised_encoder, teacher_model, 
                            use_policy_distillation=False, update_RL_models=True, 
                            only_update_critic=True, use_MLE=False, MCTS_thresh=0,
                            n_episodes=args.pretrain_critic_n_episodes)
        else:
            train_RL_models(actor_model, critic_model, actor_optimizer, critic_optimizer,
                                supervised_encoder, teacher_model,
                                use_policy_distillation=args.use_policy_distillation, 
                                update_RL_models=args.update_RL_models,
                                only_update_critic=False, use_MLE=args.use_MLE,
                                MCTS_thresh=args.MCTS_thresh, n_episodes=args.n_episodes)
    except:
        if args.save_models:
            saved_RL_model_results.save_top_models(actor_model, 'actor_CHECKPOINT.pt')
            if args.init_critic:
                saved_RL_model_results.save_top_models(critic_model, 'critic_CHECKPOINT.pt')
            saved_RL_model_results.export_rewards('model_performance.txt')
            
            if args.use_adversarial_training:
                model_name = 'adversary_model{}_{:.3}.pt'.format(
                    adversary_model.update_iter,
                    adversary_model.training_accuracy[adversary_model.update_iter][-1])
                data.save_model(adversary_model.model,
                                os.path.join(saved_RL_model_results.folder_path, model_name))
            
        logging.error("Exception occurred", exc_info=True)

#%%

def validation_perf(input_folder_name, val_pairs, n_episodes, reward_metric='BLEU1', similarity_model_name='BERT',
                    use_MLE=True, MCTS_thresh=0, set_ESIM_model=None, verbose=False):
    """Instantiates and loads trained models and environment in order to test the model performance"""
    
    supervised_encoder, supervised_decoder = sm.load_supervised_models(
            args.SM_FOLDER, encoder_file_name=args.SM_ENCODER_FILE_NAME, decoder_file_name=args.SM_DECODER_FILE_NAME)
    
    actor_model, critic_model, _, _ = init_actor_critic_models(
            supervised_decoder, init_critic=1, transfer_weights=0)
    
    actor_model, critic_model = data.load_RL_models(
                args.env_name, input_folder_name, actor_model, critic_model,
                actor_file_name='best', critic_file_name='best')
    
    teacher_model = None
    
    input_sentence = train_pairs[0]
    env = paraphrasee_env.ParaPhraseeEnvironment(
            source_sentence=input_sentence[0], target_sentence=input_sentence[1],
            supervised_encoder=supervised_encoder, reward_function=reward_metric,
            similarity_model_name=similarity_model_name, sentence_pairs=val_pairs)
    if set_ESIM_model is not None:
        env.ESIM_model = set_ESIM_model
    
    hp = HyperParams(print_every=10, early_stopping_reward_thresh=args.early_stopping_reward_thresh)
    
    validation_performance = []
    validation_sentences = []
    
    with torch.no_grad():
        for i_episode in range(1, n_episodes+1):
            (prev_action, hidden_state), ep_reward, done = env.reset(), 0, False
            
            for step_i in range(1, env.max_steps+1):
                action, hidden_state, KL_error = select_action(input_state=prev_action, input_hidden_state=hidden_state,
                                                               actor_model=actor_model, critic_model=critic_model,
                                                               teacher_model=teacher_model, K=hp.K, use_MLE=use_MLE,
                                                               MCTS_thresh=MCTS_thresh)
                
                (prev_action, hidden_state), env_reward, done, _ = env.step(action, hidden_state)
        
                reward = env_reward
                ep_reward += reward
                
                actor_model.rewards.append(reward)
                
                if done:
                    break
            
            validation_performance.append(ep_reward)
            validation_sentences.append([env.source_sentence, env.target_sentence, env.pred_sentence()])
            
            if verbose:
                print('Source sentence: ', env.source_sentence)
                print('Target sentence: ', env.target_sentence)
                print()
                print('Supervised model prediction: ', env.supervised_baseline(supervised_decoder))
                print('RL model prediction: ', env.pred_sentence(), ep_reward)
                print()
               
    return np.array(validation_sentences), np.mean(validation_performance), validation_performance

if args.test_MCTS:
    """Designed as one-off to evaluate performance of MCTS"""
    input_folder_name=args.folder_name
    val_pairs=test_pairs
    n_episodes=args.n_episodes
    reward_metric='ESIM'
    similarity_model_name='BERT'
    use_MLE=False
    MCTS_thresh=0.80
    set_ESIM_model=load_ESIM_model(folder_name='ESIM_adv_30k1', file_name='ESIM_0.755.pt')
    verbose=False
    
    supervised_encoder, supervised_decoder = sm.load_supervised_models(
            args.SM_FOLDER, encoder_file_name=args.SM_ENCODER_FILE_NAME, decoder_file_name=args.SM_DECODER_FILE_NAME)
    
    actor_model, critic_model, _, _ = init_actor_critic_models(
            supervised_decoder, init_critic=1, transfer_weights=0)
    
    actor_model, critic_model = data.load_RL_models(
                args.env_name, input_folder_name, actor_model, critic_model,
                actor_file_name='actor_0.391.pt', critic_file_name='critic_0.391.pt')
    
    teacher_model = None
    
    input_sentence = train_pairs[0]
    env = paraphrasee_env.ParaPhraseeEnvironment(
            source_sentence=input_sentence[0], target_sentence=input_sentence[1],
            supervised_encoder=supervised_encoder, reward_function=reward_metric,
            similarity_model_name=similarity_model_name, sentence_pairs=val_pairs)
    if set_ESIM_model is not None:
        env.ESIM_model = set_ESIM_model
    
    hp = HyperParams(print_every=10, early_stopping_reward_thresh=args.early_stopping_reward_thresh)
    
    validation_performance = []
    validation_sentences = []
    
    try:
        with torch.no_grad():
            for i_episode in range(1, n_episodes+1):
                (prev_action, hidden_state), ep_reward, done = env.reset(), 0, False
                
                for step_i in range(1, env.max_steps+1):
                    action, hidden_state, KL_error = select_action(input_state=prev_action, input_hidden_state=hidden_state,
                                                                   actor_model=actor_model, critic_model=critic_model,
                                                                   teacher_model=teacher_model, K=hp.K, use_MLE=use_MLE,
                                                                   MCTS_thresh=MCTS_thresh)
                    
                    (prev_action, hidden_state), env_reward, done, _ = env.step(action, hidden_state)
            
                    reward = env_reward
                    ep_reward += reward
                    
                    actor_model.rewards.append(reward)
                    
                    if done:
                        break
                
                validation_performance.append(ep_reward)
                validation_sentences.append([env.source_sentence, env.target_sentence, env.pred_sentence()])
                
                if verbose:
                    print('Source sentence: ', env.source_sentence)
                    print('Target sentence: ', env.target_sentence)
                    print()
                    print('Supervised model prediction: ', env.supervised_baseline(supervised_decoder))
                    print('RL model prediction: ', env.pred_sentence(), ep_reward)
                    print()
        
        data.save_np_data(validation_sentences, os.path.join(
                config.saved_RL_text_path, 'MCTS/', reward_metric+'_MCTS.npy'))
        
    except:
        print(validation_sentences)
        data.save_np_data(validation_sentences, os.path.join(
                config.saved_RL_text_path, 'MCTS/', reward_metric+'_MCTS.npy'))
