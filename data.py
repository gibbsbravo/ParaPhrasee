"""Imports raw data from various sources, preprocesses, creates train/test sets and vocab index
Also contains functions for saving and loading"""

import pandas as pd
import numpy as np
import os
import pickle
import torch

import json
import itertools
import re
from collections import defaultdict

import config

DEVICE = config.DEVICE
MAX_LENGTH = config.MAX_LENGTH

def create_coco_pairs(input_path):
    """Loads MS-COCO data, gets caption data and converts to caption pairs"""
    # Load data from json
    with open(input_path) as json_data:
        data = json.load(json_data)
    
    # Instantiate dictionary
    captions_dict = defaultdict(list)
    
    # Fill dictionary with captions
    for item in data['annotations']:
        captions_dict[item['image_id']].append(item['caption'])
    
    # Pair captions and convert to list instead of tuple
    coco_pairs = []
    
    for _, value in captions_dict.items():
        coco_pairs.extend(itertools.combinations(value,2))
    
    coco_pairs = np.array([list(pair) for pair in coco_pairs])
    
    return coco_pairs

def create_quora_pairs(input_path):
    """Loads Quora data and keeps only questions labelled as duplicates as paraphrase pairs"""
    # Load Quora duplicates dataset
    df = pd.read_csv(input_path)
    
    # Keep only duplicate questions
    df = df[df['is_duplicate'] == 1]
    
    df.drop(['id','qid1','qid2','is_duplicate'], axis=1, inplace=True)
    
    quora_pairs = np.array([np.array([df['question1'].iloc[i],
                                   df['question2'].iloc[i]]) for i in range(len(df))])
    return quora_pairs

def create_pred_twitter_pairs(input_path, sim_threshold=0.75):
    """Loads automated twitter data and keeps only pairs over specified model confidence threshold 
    as paraphrase pairs"""
    df = pd.read_csv(input_path, sep="\t", 
                 header=None, usecols = [0,1,2])
    df.columns = ['sim_score', 'sent1', 'sent2']
    
    # Keep only captions with similarity greater than the threshold
    df = df.loc[df['sim_score']>sim_threshold]
    
    sentence_pairs = [[a,b] for a,b in df[['sent1','sent2']].values]
    return sentence_pairs
    
def create_human_twitter_pairs(input_path, sim_threshold=0.5):
    """Loads human labelled twitter data and keeps only pairs over inter-rater agreement threshold 
    as paraphrase pairs"""
    df = pd.read_csv(input_path, sep="\t", header=None, usecols = [0,1,2])
    df.columns = ['sent1', 'sent2', 'sim_score']    

    # Convert rater agreement into score
    df['sim_score'] = [int(i[1])/int(i[3]) for i in df['sim_score']]
    
    # Keep only captions with similarity greater than the threshold
    df = df.loc[df['sim_score']>sim_threshold]
    
    sentence_pairs = [[a,b] for a,b in df[['sent1','sent2']].values]
    
    return sentence_pairs

class VocabIndex:
    """Class which converts sentences to a vocabulary and gives each word an index as well as a count"""
    def __init__(self):
        self.word2index = {'SOS':config.SOS_token, 'UNK':config.UNK_token, 'EOS':config.EOS_token}
        self.word2count = {'SOS':1, 'UNK':1, 'EOS':1}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS, and Unk

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def preprocess(input_text, remove_punct=True, lower_case=False):
    """Preprocesses raw text based on required preprocessing steps"""
    # Convert to String
    input_text = str(input_text)
    
    # Lower first character
    input_text = input_text[0].lower() + input_text[1:]
    
    if remove_punct:
        # Add space in front of key punctuation and remove other punctuation
        input_text = re.sub(r"([.!?])", r" \1", input_text)
        input_text = re.sub(r"[^a-zA-Z.!?-]+", r" ", input_text)
    
    if lower_case:
        # Convert to lower case
        input_text = input_text.lower()
    
    return input_text

def filterPairs(pairs, max_length=18):
    """Removes pairs where either sentence has more tokens than the maxmimum"""
    return np.array([pair for pair in pairs if 
            len(pair[0].split(' ')) < max_length and \
            len(pair[1].split(' ')) < max_length])
    
def sample_list(input_list, n_samples=5000, sample_by_prop=False, sample_prop=0.10):
    """Returns a random subset of an input list based on either number of samples or proportion"""
    if sample_by_prop:
        selected_indices = np.random.choice(len(input_list),
                                            int(sample_prop * len(input_list)), replace=False)
    else:
        selected_indices = np.random.choice(len(input_list), n_samples, replace=False)
    sampled_list = input_list[selected_indices]
    return sampled_list

def caption_processing_pipeline(input_pairs, n_samples, max_length=18, 
                                remove_punct=True, lower_case=False):
    """Applies filtering, preprocessing, and sampling to raw dataset"""
    filtered_pairs = filterPairs(input_pairs, max_length)
    for idx, pair in enumerate(filtered_pairs):
        filtered_pairs[idx][0] = preprocess(pair[0], remove_punct, lower_case)
        filtered_pairs[idx][1] = preprocess(pair[1], remove_punct, lower_case)

    refiltered_pairs = filterPairs(filtered_pairs, max_length)
    sampled_pairs = sample_list(refiltered_pairs, n_samples)
    return sampled_pairs

def convert_unk_terms(vocab_index, min_count=1):
    """Converts words below count level's index to UNK"""
    unk_words = [a for a,b in vocab_index.word2count.items() if b <= min_count]
    
    # Convert word to UNK index
    for word in unk_words:
        vocab_index.word2index[word] = config.UNK_token
   
    return unk_words, len(unk_words)

def get_pairs(dataset_size=20000, coco_prop=0.50, quora_prop=0.25, twitter_prop=0.25, 
                      remove_unk=True, max_length=18):
    """Loads data based on dataset proportions and applies processing pipeline,
        then fills vocab_index and optionally removes unknown words"""
    # Load dataframes
    print('Loading data...')
    coco_train_pairs = create_coco_pairs(config.coco_train_path)
    coco_val_pairs = create_coco_pairs(config.coco_val_path)
    coco_pairs = np.vstack([coco_train_pairs, coco_val_pairs])
    
    quora_pairs = create_quora_pairs(config.quora_path)
    
    twitter_pairs = create_pred_twitter_pairs(config.twitter_path)
    
    # Filter out sentences greater than specified length and downsample
    # Can also be used to remove data from sample through setting sample_prop to zero
    sampled_coco_pairs = caption_processing_pipeline(coco_pairs, int(dataset_size*coco_prop),
                                                      max_length, remove_punct=True, lower_case=True)
    sampled_quora_pairs = caption_processing_pipeline(quora_pairs, int(dataset_size*quora_prop), max_length)
    sampled_twitter_pairs = caption_processing_pipeline(twitter_pairs, int(dataset_size*twitter_prop),
                                                        max_length, remove_punct=True, lower_case=True)
    
    # Merge dataframes
    caption_pairs = np.vstack([sampled_coco_pairs, sampled_quora_pairs, sampled_twitter_pairs])

    # Initialize VocabIndex and populate
    caption_vocab_index = VocabIndex()
    
    for idx, pair in enumerate(caption_pairs):
        caption_vocab_index.addSentence(pair[0])
        caption_vocab_index.addSentence(pair[1])
        
    print('Dataframe successfully created:')    
    print('Total Samples: {}'.format(len(caption_pairs)))
    print('    - COCO Image Captioning Samples: {} ({:.1%})'.format(len(sampled_coco_pairs),
          len(sampled_coco_pairs) / len(caption_pairs)))
    print('    - Quora Duplicate Question Samples: {} ({:.1%})'.format(len(sampled_quora_pairs),
          len(sampled_quora_pairs) / len(caption_pairs)))
    print('    - Twitter Share URL Samples: {} ({:.1%})'.format(len(sampled_twitter_pairs),
          len(sampled_twitter_pairs) / len(caption_pairs)))
    
    # Convert terms only occurring n times to unk in dict (DOES NOT IMPACT ACTUAL TEXT)
    if remove_unk:
        _, n_unk_words = convert_unk_terms(caption_vocab_index, min_count=1)
        
        print('Total vocabulary size: {}'.format(caption_vocab_index.n_words))
        print('{} UNK words'.format(n_unk_words))
    
    return caption_pairs, caption_vocab_index

def train_test_split(input_array, splits=(0.65,0.25,0.10)):
    """Creates a random train, val, test split for a given data input array"""
    np.random.seed(config.SEED)
    np.random.shuffle(input_array)
    n_dataset = len(input_array)
    train_split, val_split, test_split = splits
    
    train_idx = int(train_split * n_dataset)
    val_idx = int(train_split * n_dataset)+int(val_split * n_dataset)
    
    train_set = input_array[:train_idx]
    val_set = input_array[train_idx:val_idx]
    test_set = input_array[val_idx:]
    
    assert len(train_set) + len(val_set) + len(test_set) == len(input_array), "Some data has been lost"
    
    return train_set, val_set, test_set

def create_data(load_data=True, pairs_input_path='Data/pairs_data.npy',
                index_input_path='Data/vocab_index.pickle', dataset_size=20000):
    """Creates a train / test split and vocab_index by loading or 
    creating the dataset based on the file path or specified dataset proportions / requirements"""
    if load_data:
        print('Loading saved dataset....')
        pairs = load_np_data(pairs_input_path)
        vocab_index = load_vocab_index(index_input_path)
        print('Dataset loaded.')
        print('    Total number of sentence pairs: {}'.format(len(pairs)))
        print('    Total vocabulary size: {}'.format(vocab_index.n_words))
    
    else:
        # Create dataset from corpora
        pairs, vocab_index = \
        get_pairs(dataset_size=dataset_size, coco_prop=1,
                                            quora_prop=0, twitter_prop=0,
                                            remove_unk=False, max_length=MAX_LENGTH)
        # Shuffle dataset and ensure 
        np.random.seed(config.SEED)
        np.random.shuffle(pairs)
        assert max([max(len(a.split()),len(b.split())) for a,b in pairs]) < MAX_LENGTH, "Pairs exceed MAX_LENGTH"

    train_pairs, val_pairs, test_pairs = train_test_split(pairs, splits=(0.65,0.25,0.10))
    return train_pairs, val_pairs, test_pairs, vocab_index, pairs

def instantiate_vocab_idx(input_path):
    """Instantiates a vocab_index and fills it with the caption pairs"""
    caption_pairs = load_np_data(input_path)
    caption_vocab_index = VocabIndex()
    
    for idx, pair in enumerate(caption_pairs):
        caption_vocab_index.addSentence(pair[0])
        caption_vocab_index.addSentence(pair[1])
    return caption_vocab_index

# %% Saving and Loading Data

def save_np_data(input_data, file_name):
    """Only designed for saving Numpy arrays therefore name must include .npy extension"""
    if os.path.isfile(file_name):
        print('Error: File already exists - please change name or remove conflicting file')
    else:
        assert '.npy' in file_name, 'Please ensure .npy extension is included in file_name'
        np.save(file_name, input_data)

def load_np_data(file_name):
    """Only designed for loading Numpy arrays"""
    assert '.npy' in file_name, 'Please ensure file is .npy filetype'
    return np.load(file_name)

def save_np_to_text(input_data, file_name):
    """Designed for saving txt files therefore name must include .txt extension"""
    assert '.txt' in file_name, 'Please ensure .txt extension is included in file_name'
    with open(file_name, 'a') as file:
        np.savetxt(file, input_data, fmt='%1.2f')

def save_dict(input_dict, file_name):
    """Only designed for saving dicts to JSON arrays therefore name must include .json extension"""
    if os.path.isfile(file_name):
        print('Error: File already exists - please change name or remove conflicting file')
    else:
        assert '.json' in file_name, 'Please ensure .json extension is included in file_name'
        with open(file_name, 'w') as fp:
            json.dump(input_dict, fp)

def load_dict(file_name):
    """Used to load JSON dicts"""
    with open(file_name) as json_data:
        data = json.load(json_data)
    return data

def save_vocab_index(vocab_index, file_name):
    """Only designed for pickling the vocab idx therefore name must include .pickle extension"""
    if os.path.isfile(file_name):
        print('Error: File already exists - please change name or remove conflicting file')
    else:
        assert '.pickle' in file_name, 'Please ensure .pickle extension is included in file_name'
        pickle_out = open(file_name, "wb")
        pickle.dump(vocab_index, pickle_out)
        pickle_out.close()

def load_vocab_index(file_name):
    """Only designed for loading pickle files"""
    assert '.pickle' in file_name, 'Please ensure file is .pickle filetype'
    with open(file_name, 'rb') as handle:
        vocab_index = pickle.load(handle)
    return vocab_index


#%% Saving and Loading Models 

def save_model(model, file_name):
    """Only designed for saving PyTorch model weights therefore must include .pt extension"""
    if os.path.isfile(file_name):
        print('Error: File already exists - please change name or remove conflicting file')
    else:
        assert '.pt' in file_name, 'Please ensure .pt extension is included in file_name'
        torch.save(model.state_dict(), file_name)

def load_model(model, file_name, device=DEVICE):
    """Only designed for loading PyTorch model weights therefore must ensure model has an
    identical structure to saved version"""
    if DEVICE.type == 'cuda':
        model.load_state_dict(torch.load(file_name))
        model.to(device)
    else:
        model.load_state_dict(torch.load(file_name, map_location=device))

def save_exp_args(exp_args, file_name):
    """Saves experiment input arguments from model runs"""
    args_dict = dict(vars(exp_args))
    save_dict(args_dict, file_name)
    
def save_model_args(input_model, file_name):
    """Save model arguments for each experiment"""
    try:
        model_dict = dict(vars(input_model)['_modules'])
        model_dict['model_name'] = input_model.name
        save_dict(str(model_dict), file_name)
    except:
        print("Unable to save model args")
        
def extract_model_number(input_text, start_symbol='_', end_symbol='.pt'):
    """Returns the model number for a given saved model"""
    m = re.search(start_symbol+'(.+?)'+end_symbol, input_text)
    if m:
        found = m.group(1)
        
    return float(found)

def get_top_n_models(input_path, model_type='decoder', n=1, descending=False):
    """Returns the the top n saved models by performance"""
    folder_files = os.listdir(input_path)
    loss_value = [extract_model_number(file) for file in folder_files if ('.pt' in file) \
                      and (model_type in file)]
    loss_value.sort(reverse=descending)
    n_values = loss_value[:n]
    return n_values

def load_RL_models(env_name, folder_name, actor_model, critic_model,
                   actor_file_name='best', critic_file_name='best'):
    """Loads saved RL models"""
    if actor_file_name == 'best':
        actor_file_name = 'actor_{:.3f}.pt'.format(
                get_top_n_models(
                        os.path.join(config.saved_RL_model_path, env_name, folder_name), 'actor', n=1, descending=True)[0])
    
    load_model(actor_model, os.path.join(config.saved_RL_model_path, env_name,
                                              folder_name, actor_file_name))
    
    if critic_model is not None:
        if critic_file_name == 'best':
            critic_file_name = 'critic_{:.3f}.pt'.format(
                    get_top_n_models(
                            os.path.join(config.saved_RL_model_path, env_name, folder_name), 'critic', n=1, descending=True)[0])
        
        load_model(critic_model, os.path.join(config.saved_RL_model_path, env_name,
                                                   folder_name, critic_file_name))
    
        return actor_model, critic_model
    
    else:
        return actor_model, None

class SaveSupervisedModelResults(object):
    """Object for storing supervised model results as the experiment is being run"""
    def __init__(self, folder_name):
        self.path = config.saved_supervised_model_path
        self.folder_name = folder_name
        self.folder_path = os.path.join(self.path, self.folder_name)
        
        self.track_loss = True
        self.train_loss = []
        self.val_loss = []
        self.val_loss_thresh = 3.70
        
    def check_folder_exists(self):
        if os.path.isdir(self.folder_path):
            raise Exception('This experiment folder already exists')
    
    def init_folder(self, exp_args, encoder_model=None, decoder_model=None):
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            pass
        save_exp_args(exp_args, os.path.join(self.folder_path, 'exp_args.json'))
        save_model_args(decoder_model, os.path.join(self.folder_path, 'decoder_args.json'))
        
        if encoder_model is not None:
            save_model_args(encoder_model, os.path.join(self.folder_path, 'encoder_args.json'))
    
    def export_loss(self, train_file_name, val_file_name):
        save_np_to_text(self.train_loss, os.path.join(self.folder_path, train_file_name))
        save_np_to_text(self.val_loss, os.path.join(self.folder_path, val_file_name))
        self.reset()
    
    def save_top_models(self, input_model, file_name):
        save_model(input_model, os.path.join(self.folder_path, file_name))
    
    def reset(self):
        self.train_loss = []
        self.val_loss = []
        
class SaveRLModelResults(object):
    """Object for storing RL model results as the experiment is being run"""
    def __init__(self, env_name, folder_name):
        self.path = config.saved_RL_model_path
        self.folder_name = folder_name
        self.env_name = env_name
        self.folder_path = os.path.join(self.path, self.env_name, self.folder_name)
        
        self.env_rewards = []
        self.KL_penalty = []
    
    def check_folder_exists(self):
        if os.path.isdir(self.folder_path):
            raise Exception('This experiment folder already exists')
    
    def init_folder(self, exp_args, actor_model=None, critic_model=None):
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            pass
        save_exp_args(exp_args, os.path.join(self.folder_path, 'exp_args.json'))
        save_model_args(actor_model, os.path.join(self.folder_path, 'actor_args.json'))
        
        if critic_model is not None:
            save_model_args(critic_model, os.path.join(self.folder_path, 'critic_args.json'))
    
    def export_rewards(self, file_name):
        if len(self.KL_penalty) > 0:
            combined_rewards = np.array([[reward, penalty] for (reward, penalty) in 
                                      zip(self.env_rewards, self.KL_penalty)])
            save_np_to_text(combined_rewards, os.path.join(self.folder_path, file_name))
        
        else:
            save_np_to_text(self.env_rewards, os.path.join(self.folder_path, file_name))
        
        self.reset()
    
    def save_top_models(self, input_model, file_name):
        save_model(input_model, os.path.join(self.folder_path, file_name))
    
    def reset(self):
        self.env_rewards = []
        self.KL_penalty = []    

def load_loss_data(folder_name):
    """Loads saved loss data for supervised models"""
    training_loss = pd.read_csv(os.path.join(config.saved_supervised_model_path, folder_name, 'training_loss.txt'),
                                sep=" ", header=None, names = ['train_loss'], dtype = {'train_loss':np.float32})
    val_loss = pd.read_csv(os.path.join(config.saved_supervised_model_path, folder_name, 'val_loss.txt'),
                           sep=" ", header=None, names = ['val_loss'], dtype = {'val_loss':np.float32})
    
    n_iterations = int(len(training_loss)/len(val_loss))
    
    df = training_loss.copy()
    df['val_loss'] = np.repeat(val_loss['val_loss'].values, n_iterations)
    
    return df

def load_rewards_data(env_name, folder_name):
    """Loads saved rewards data for RL models"""
    rewards_df = pd.read_csv(os.path.join(config.saved_RL_model_path, env_name, folder_name,
                                          'model_performance.txt'), sep=" ", header=None)
    if len(rewards_df.columns) == 2:
        rewards_df.columns = ['env_rewards', 'KL_penalty']
        rewards_df['total_rewards'] = rewards_df['env_rewards'] + rewards_df['KL_penalty']
        
    elif len(rewards_df.columns) == 1:
        rewards_df.columns = ['env_rewards']
    
    return rewards_df

#%% ------------------------------ Load data ------------------------------
#Create data for use across all other modules. Loads the same saved data each time.

TRAIN_PAIRS, VAL_PAIRS, TEST_PAIRS, VOCAB_INDEX, _ = \
create_data(load_data=True, pairs_input_path=config.pairs_path,
                index_input_path=config.vocab_index_path, dataset_size=20000)

# Ensures that manually entered terms are in vocab_index
if 'SOS' not in VOCAB_INDEX.word2index:
    VOCAB_INDEX.word2index['SOS'] = config.SOS_token
    VOCAB_INDEX.word2index['UNK'] = config.UNK_token
    VOCAB_INDEX.word2index['EOS'] = config.EOS_token
    
    VOCAB_INDEX.word2count['SOS'] = 1
    VOCAB_INDEX.word2count['UNK'] = 1
    VOCAB_INDEX.word2count['EOS'] = 1


# Save data and index
#data.save_np_data(pairs, 'Data/pairs_data_100k.npy')
#data.save_vocab_index(vocab_index, 'Data/vocab_index_100k.pickle')

#%% --------------------------------------ARCHIVE----------------------------------------------

#def create_wiki_df():
#    """Creates dataframe for Wiki Answers dataset"""
#    
#    # Duplicate questions after lemmatization
#    df = pd.read_csv('D:/Paraphrase_Datasets/WikiAnswers_Duplicates/word_alignments.txt',
#                     sep="\t", nrows=100000, header=None, usecols = [0,1])
#    
#    # Finds the original question from the lemmatized version
#    question_df = pd.read_csv('D:/Paraphrase_Datasets/WikiAnswers_Duplicates/questions.txt',
#                     sep="\t",   header=None, usecols = [0,3])
#    
#    search_text = 'how Dose the stellar'
#    original_question = question_df.iloc[question_df[3].loc[[search_text in str(q) for q in question_df[3]]].index][0]
#    print(original_question.item())
#    
#    return df
