"""Defines environment dynamics for paraphrase generation task as RL problem"""

import torch

import numpy as np
import random
import os

import config
import data
import model_evaluation
import supervised_model as sm
from train_ESIM import load_ESIM_model

DEVICE = config.DEVICE

MAX_LENGTH = config.MAX_LENGTH

SOS_token = config.SOS_token
EOS_token = config.EOS_token

train_pairs = data.TRAIN_PAIRS
val_pairs = data.VAL_PAIRS
test_pairs = data.TEST_PAIRS
vocab_index = data.VOCAB_INDEX

#%%

class ParaPhraseeEnvironment(object):
    """Define the paraphrase generation task in the style of OpenAI Gym"""
    def __init__(self, source_sentence, target_sentence, supervised_encoder, 
                 reward_function, similarity_model_name, sentence_pairs):
        self.name = 'ParaPhrasee'
        self.source_sentence = source_sentence # Stored as string
        self.target_sentence = target_sentence # Stored as string
        self.predicted_words = []
        self.reward_function = reward_function # String ex. BLEU
        self.similarity_model_name = similarity_model_name
        self.ESIM_model_name = 'ESIM_noisy_3'
        self.similarity_model, self.fluency_model, self.ESIM_model, \
        self.logr_model, self.std_scaler, \
        self.similarity_dist, self.fluency_dist, self.ESIM_dist = model_evaluation.init_eval_models(
                reward_function=self.reward_function, similarity_model_name=self.similarity_model_name,
                ESIM_model_name=self.ESIM_model_name)
        self.sentence_pairs = sentence_pairs
        self.supervised_encoder = supervised_encoder
        
        self.max_length = MAX_LENGTH
        self.max_steps = self.max_length
        self.done = 0
        self.ep_reward = 0
        self.gamma = 0.999
        self.changing_input = True
        
        
        self.action_tensor = torch.tensor([[SOS_token]], device=DEVICE) 
        self.encoder_outputs = torch.zeros(MAX_LENGTH, supervised_encoder.hidden_size, device=DEVICE)
        self.context, _, _ = sm.embed_input_sentence([self.source_sentence, self.target_sentence], supervised_encoder,
                                               max_length=self.max_length)
        self.state = (self.action_tensor, self.context)
        self.action_space = vocab_index.n_words
        
    def pred_sentence(self):
        """Returns the sentence prediction from the environment"""
        output_sentence = ' '.join(self.predicted_words)
        return output_sentence
    
    def supervised_baseline(self, supervised_decoder):
        """Returns the supervised model prediction for the same sentence for comparative purposes"""
        supervised_decoder_pred, _, baseline_reward = sm.validationMetricPerformance(
                input_pairs=[(self.source_sentence, self.target_sentence)], encoder=self.supervised_encoder,
                decoder=supervised_decoder, similarity_model=self.similarity_model, fluency_model=self.fluency_model,
                ESIM_model=self.ESIM_model, logr_model=self.logr_model, std_scaler=self.std_scaler,
                similarity_dist=self.similarity_dist, fluency_dist=self.fluency_dist, ESIM_dist=self.ESIM_dist,
                vocab_index=vocab_index, verbose=False, metric=self.reward_function)
        
        supervised_decoder_pred = supervised_decoder_pred[0][1]
        
        return supervised_decoder_pred, np.around(baseline_reward, 3)
    
    def step(self, action, decoder_hidden):
        """Key function which represents the transition dynamics. 
        given an action (word choice) this returns the updated state FROM THE AGENT
        is effectively the decoder
        
        All this is effectively doing is checking if the episode is over and returning the 
        appropirate reward, else updating the state based on the decoder outputs"""

        # Check whether episode is over
        if (action == EOS_token) or (len(self.predicted_words)>= self.max_length):
            self.state = action, decoder_hidden
            RL_model_reward = model_evaluation.performance_metrics(
                    target_sentence=self.target_sentence, pred_sentence=self.pred_sentence(), 
                    similarity_model=self.similarity_model, fluency_model=self.fluency_model, ESIM_model=self.ESIM_model,
                    logr_model=self.logr_model, std_scaler=self.std_scaler,
                    similarity_dist=self.similarity_dist, fluency_dist=self.fluency_dist, ESIM_dist=self.ESIM_dist,
                    vocab_index=vocab_index, metric=self.reward_function)
            # Calculate relative reward
            self.ep_reward = np.around(RL_model_reward, 3)
            self.done = 1
        else:
            self.state = action, decoder_hidden
            
            # Add word to pred words
            self.predicted_words.append(vocab_index.index2word[action.item()])
            
        return self.state, self.ep_reward, self.done, None
    
    def reset(self):
        """Resets the environment to a random initial state through picking a random sentence from the 
            sentence input pairs"""
        if self.changing_input:
            sentence_pair = random.choice(self.sentence_pairs)
            self.source_sentence = sentence_pair[0] # Stored as string
            self.target_sentence = sentence_pair[1] # Stored as string
        
        self.predicted_words = []
        
        self.ep_reward = 0
        self.done = 0
        
        self.action_tensor = torch.tensor([[SOS_token]], device=DEVICE) 
        self.encoder_outputs = torch.zeros(MAX_LENGTH, self.supervised_encoder.hidden_size, device=DEVICE)
        self.context, _, _ = sm.embed_input_sentence([self.source_sentence, self.target_sentence],
                                                     self.supervised_encoder, max_length=self.max_length)
        self.state = (self.action_tensor, self.context)
        
        return self.state
