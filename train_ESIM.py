"""Defines and trains an ESIM model"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os

from ESIM.ESIM import ESIM

import config
from encoder_models import create_vocab_tensors
import data
from utils import tensorsFromPair

DEVICE = config.DEVICE
GPU_ENABLED = config.GPU_ENABLED
MAX_LENGTH = config.MAX_LENGTH

SOS_token = config.SOS_token
EOS_token = config.EOS_token

# Load Data 
if __name__ == '__main__':
    train_pairs = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_train_pairs.npy'))
    y_train = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_train_labels.npy'))
    y_train = torch.from_numpy(y_train).to(DEVICE)
    
    val_pairs = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_val_pairs.npy'))
    y_val = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_val_labels.npy'))
    y_val = torch.from_numpy(y_val).to(DEVICE)
    
    test_pairs = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_test_pairs.npy'))
    y_test = data.load_np_data(os.path.join(config.saved_ESIM_model_path, 'aug_test_labels.npy'))
    y_test = torch.from_numpy(y_test).to(DEVICE)

vocab_index = data.VOCAB_INDEX

parser = argparse.ArgumentParser(description='Train_Supervised_Model')
parser.add_argument('--train_models', action='store_true',
                    help='enable training of models')
parser.add_argument('--folder_name', type=str,
                    help='Brief description of experiment (no spaces)')

parser.add_argument('--n_epochs', type=int, default=1,
                    help='number of epochs to train online in each loop (default: 1)')
parser.add_argument('--verbose_training', action='store_true',
                    help='print results during training')

parser.add_argument('--load_models', action='store_true', help='Load pretrained model from prior point')
parser.add_argument('--load_model_folder_name', type=str,
                    help='folder which contains the saved models to be used')

if __name__ == '__main__':
    args = parser.parse_args()
    args.save_models = 0
    
    if args.train_models:
        args.save_models = 1
        saved_ESIM_model_results = data.SaveSupervisedModelResults(args.folder_name)
        saved_ESIM_model_results.check_folder_exists()

pretrained_emb_file = 'pretrained_emb_100k.npy'


#%% Manual commands for testing

#args.train_models = 1
#args.verbose_training = 1
#saved_ESIM_model_results = data.SaveSupervisedModelResults('ESIM')

#%%

def mask_batch(input_batch_pairs):
    """Convert batch of sentence pairs to tensors and masks for ESIM model"""
    input_tensor = torch.zeros((MAX_LENGTH, len(input_batch_pairs)), dtype=torch.long, device=DEVICE)
    target_tensor = torch.zeros((MAX_LENGTH,len(input_batch_pairs)), dtype=torch.long, device=DEVICE)
    
    for idx, pair in enumerate(input_batch_pairs):
        encoded_input, encoded_target = tensorsFromPair(pair)
        input_tensor[:len(encoded_input), idx], target_tensor[:len(encoded_target), idx] = \
            encoded_input.view(-1), encoded_target.view(-1)
    
    input_tensor_mask, target_tensor_mask = input_tensor != 0, target_tensor != 0
    input_tensor_mask, target_tensor_mask = input_tensor_mask.float(), target_tensor_mask.float()
    
    return input_tensor, input_tensor_mask, target_tensor, target_tensor_mask

def ESIM_pred(input_pairs, model, temperature=1):
    """Returns probability that sentences are paraphrases from trained ESIM model"""
    model.eval()
    
    input_tensor, input_tensor_mask, target_tensor, target_tensor_mask = mask_batch(input_pairs)
    
    with torch.no_grad():
        output = model(input_tensor, input_tensor_mask, target_tensor, target_tensor_mask)
        probs = F.softmax(output / temperature, dim=1)
        
    return probs[:,1]

def validation_error(val_pairs, y_val, model, temperature=1, batch_size=32, verbose=True):
    """Evalutes the error on a set of input pairs in terms of loss. 
    Is intended to be used on a validation or test set to evaluate performance"""
    model.eval()
    total_val_loss = 0
    val_sents_scanned = 0
    val_num_correct = 0
    batch_counter = 0
    batch_size = min(len(val_pairs), batch_size)
    
    output_probs = torch.zeros((len(val_pairs),2), device=DEVICE)
    
    for idx in range(len(val_pairs) // batch_size):
        input_tensor, input_tensor_mask, target_tensor, target_tensor_mask = mask_batch(
                val_pairs[idx*batch_size:(idx+1)*batch_size])
        batch_labels = y_val[idx*batch_size:(idx+1)*batch_size]
        
        with torch.no_grad():
            output = model(input_tensor, input_tensor_mask, target_tensor, target_tensor_mask)
            probs = F.softmax(output / temperature, dim=1)
        
        loss = criterion(output, batch_labels)
        
        output_probs[idx*batch_size:(idx+1)*batch_size,:] = probs
        
        result = output.detach().cpu().numpy()
        a = np.argmax(result, axis=1)
        b = batch_labels.data.cpu().numpy()
        
        val_num_correct += np.sum(a == b)
        val_sents_scanned += len(batch_labels)
        
        batch_counter += 1
        
        batch_loss = loss.data.item()
        total_val_loss += batch_loss
    
    val_loss = total_val_loss / batch_counter
    val_accuracy = (val_num_correct / val_sents_scanned)
    
    if verbose:
        print('{} batches | validation loss: {:.3} | validation accuracy: {:.3}'.format(
                    batch_counter, val_loss, val_accuracy))
        
    return val_loss, val_accuracy, output_probs

def model_pipeline(model, criterion, optimizer, batch_size=32, num_epochs=1,
                   report_interval=10, early_stopping_interval=100, verbose=True):
    """Model pipeline which trains model and also generates examples while training and evaluation 
    on the validation set for potential early stopping"""
    batch_counter = 0
    
    print('start training...')
    model.train()
    
    for epoch in range(num_epochs):
        model.train()
        print('--' * 20)
        train_sents_scanned = 0
        train_num_correct = 0
        batch_counter = 0
        
        for idx in range(len(train_pairs) // batch_size):
            input_tensor, input_tensor_mask, target_tensor, target_tensor_mask = mask_batch(
                    train_pairs[idx*batch_size:(idx+1)*batch_size])
            batch_labels = y_train[idx*batch_size:(idx+1)*batch_size]
            
            optimizer.zero_grad()
            output = model(input_tensor, input_tensor_mask, target_tensor, target_tensor_mask)
            loss = criterion(output, batch_labels)
            loss.backward()
            
            result = output.detach().cpu().numpy()
            a = np.argmax(result, axis=1)
            b = batch_labels.data.cpu().numpy()
            
            train_num_correct += np.sum(a == b)
            train_sents_scanned += len(batch_labels)
    
            optimizer.step()
            training_loss = loss.detach().item()
            batch_counter += 1
            
            saved_ESIM_model_results.train_loss.append(np.around(training_loss, 4))
            
            if batch_counter % report_interval == 0 and verbose == True:
                print('{} epochs, {} batches | training batch loss: {:.3} | train accuracy: {:.3}'.format(
                        epoch, batch_counter, training_loss, train_num_correct / train_sents_scanned))
                
            if batch_counter % early_stopping_interval == 0:
                val_prop = int(0.05 * len(val_pairs))
                random_idx = np.random.choice(val_prop, val_prop, replace=False)
                sample_val_pairs, sample_y_train = val_pairs[random_idx], y_val[random_idx]
                
                val_loss, val_accuracy, _ = validation_error(
                        sample_val_pairs, sample_y_train, model,
                        temperature=1, batch_size=32, verbose=verbose)
                model.train()
                                
                saved_ESIM_model_results.val_loss.append(np.around(val_accuracy, 4))
                

        saved_ESIM_model_results.save_top_models(model, 'ESIM_{:.3f}.pt'.format(
                val_accuracy))
    saved_ESIM_model_results.export_loss('training_loss.txt', 'val_loss.txt')
    
class HyperParams(object):
    """Sets the experiment hyperparameters"""
    def __init__(self, print_every=10):
        self.print_every = print_every
        self.early_stopping_interval = 150
        self.dim_word = 300
        self.batch_size = 32
        self.n_words = vocab_index.n_words
        self.n_classes = 2

def load_pretrained_emb(input_path=None):
    """Loads word embeddings for vocabulary or creates new vocabulary"""
    if input_path is not None:
        return data.load_np_data(input_path)
    else:
        return create_vocab_tensors(vocab_index)[0].cpu().numpy()

def load_ESIM_model(folder_name, file_name='best', path_override=None):
    """Instantiates and loads ESIM model"""
    hp = HyperParams()
    
    pretrained_emb = load_pretrained_emb(os.path.join(config.saved_ESIM_model_path, pretrained_emb_file))
    ESIM_model = ESIM(hp.dim_word, hp.n_classes, hp.n_words, hp.dim_word, pretrained_emb).to(DEVICE)
    
    
    if path_override is not None:
        data.load_model(ESIM_model, os.path.join(path_override, file_name))  
    
    else:
        if file_name == 'best':
            file_name = 'ESIM_{:.3f}.pt'.format(data.get_top_n_models(
                    os.path.join(config.saved_ESIM_model_path, folder_name), 'ESIM', n=1, descending=True)[0])
            
        data.load_model(ESIM_model, os.path.join(config.saved_ESIM_model_path, folder_name, file_name))
        
    return ESIM_model

#%%

class RLAdversary():
    """Defines RL adversary model for use as reward function"""
    def __init__(self, folder_name, file_name='best'):
        super(RLAdversary, self).__init__()
        self.name = 'ESIM RL Adversary'
        self.folder_name = folder_name
        self.file_name = file_name
        self.model, self.criterion, self.optimizer = self.init_model()
        
        self.pred_pairs = []
        self.target_pairs = []
        
        self.batch_size = 32
        self.num_epochs = 1
        
        self.update_iter = 0
        self.training_accuracy = {}
        
    def init_model(self):
        model = load_ESIM_model(self.folder_name, self.file_name)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        return model, criterion, optimizer
    
    def create_update_data(self):
        update_pairs = np.concatenate([self.pred_pairs, self.target_pairs])
        y_update = torch.zeros((len(update_pairs),), dtype=torch.long, device=DEVICE)
        y_update[len(self.pred_pairs):] = 1
        
        random_idx = np.random.choice(len(y_update), len(y_update), replace=False)
        update_pairs, y_update = update_pairs[random_idx], y_update[random_idx]
        return update_pairs, y_update
    
    def update_model(self):
        self.update_iter += 1
        self.training_accuracy[self.update_iter] = []
        
        update_pairs, y_update = self.create_update_data()
        
        self.model.train()
        
        batch_size = min(len(update_pairs), self.batch_size)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_sents_scanned = 0
            train_num_correct = 0
            
            for idx in range(len(update_pairs) // batch_size):
                input_tensor, input_tensor_mask, target_tensor, target_tensor_mask = mask_batch(
                        update_pairs[idx*batch_size:(idx+1)*batch_size])
                batch_labels = y_update[idx*batch_size:(idx+1)*batch_size]
                
                self.optimizer.zero_grad()
                output = self.model(input_tensor, input_tensor_mask, target_tensor, target_tensor_mask)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                
                self.optimizer.step()
                
                result = output.detach().cpu().numpy()
                a = np.argmax(result, axis=1)
                b = batch_labels.data.cpu().numpy()
                
                train_num_correct += np.sum(a == b)
                train_sents_scanned += len(batch_labels)
                
                self.training_accuracy[self.update_iter].append(train_num_correct / train_sents_scanned)
        
        self.pred_pairs = []
        self.target_pairs = []

    def reset(self):
        self.model, self.criterion, self.optimizer = self.init_model()
        
        self.pred_pairs = []
        self.target_pairs = []
        
        self.batch_size=32
        self.num_epochs=1
        
        self.update_iter = 0
        self.training_accuracy = {}

#%%

if (__name__ == '__main__') and args.train_models:
    """Initializes models subject to cmd line args and then trains and evaluates performance"""
    
    # Set the hyperparameters
    hp = HyperParams()
    
    # Load pretrained embeddings and model
    pretrained_emb = load_pretrained_emb(os.path.join(config.saved_ESIM_model_path, pretrained_emb_file))
    model = ESIM(hp.dim_word, hp.n_classes, hp.n_words, hp.dim_word, pretrained_emb).to(DEVICE)
    
    # Set criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create folder if saving 
    if args.save_models:
        saved_ESIM_model_results.init_folder(args, None, None)
        
    # Train model
    model_pipeline(model=model, criterion=criterion, optimizer=optimizer, batch_size=hp.batch_size,
                   num_epochs=args.n_epochs, report_interval=hp.print_every,
                   early_stopping_interval = hp.early_stopping_interval, verbose=args.verbose_training)

