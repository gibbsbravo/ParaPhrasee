"""Defines, trains, and evaluates the defined supervised model with MLE. 
    Includes modifications for teacher forcing and attention."""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

import time
import math
import regex as re
import operator
from queue import PriorityQueue
from collections import deque

import os

import config
import data
import model_evaluation
import encoder_models
import utils

# Configure hyperparameters
DEVICE = config.DEVICE
GPU_ENABLED = config.GPU_ENABLED

MAX_LENGTH = config.MAX_LENGTH
SOS_token = config.SOS_token
EOS_token = config.EOS_token
UNK_token = config.UNK_token

#Load data 
train_pairs = data.TRAIN_PAIRS
val_pairs = data.VAL_PAIRS
test_pairs = data.TEST_PAIRS
vocab_index = data.VOCAB_INDEX

# Set experiment args from command line
parser = argparse.ArgumentParser(description='Train_Supervised_Model')
parser.add_argument('--train_models', action='store_true',
                    help='enable training of models')
parser.add_argument('--folder_name', type=str,
                    help='Brief description of experiment (no spaces)')

parser.add_argument('--n_iterations', type=int, default=15,
                    help='number of iterations to run the training loop (default: 15)')
parser.add_argument('--n_epochs', type=int, default=5000,
                    help='number of epochs to train online in each loop (default: 5000)')
parser.add_argument('--early_stoppage_holdoff', type=int, default=5,
                    help='number of iterations where validation loss can be higher than min (default: 2)')

parser.add_argument('--start_tf_ratio', type=float, default=0.90,
                    help='ratio of teacher forcing at start (default: 0.90)')
parser.add_argument('--end_tf_ratio', type=float, default=0.85,
                    help='ratio of teacher forcing at end (default: 0.85)')
parser.add_argument('--tf_decay_iters', type=int, default=5,
                    help='number of iterations it takes for teacher forcing to decay to end value (default: 5)')

parser.add_argument('--encoder_model', type=str,
                    choices=['VanillaEncoder']+encoder_models.pretrained_models_list,
                    default='VanillaEncoder', help='select encoder model (default: VanillaEncoder)')
parser.add_argument('--decoder_model', type=str, 
                    choices=['VanillaDecoder', 'AttnDecoder'],
                    default='VanillaDecoder', help='select decoder model (default: VanillaEncoder)')
parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Switch'], default='SGD',
                    help='optimizer used in training (default: SGD)')

parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden nodes in encoder and decoder (default: 256)')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='number of hidden nodes in encoder and decoder embedding layers (default: 256)')

parser.add_argument('--load_models', action='store_true', help='Load pretrained model from prior point')
parser.add_argument('--load_model_folder_name', type=str,
                    help='folder which contains the saved models to be used')

if __name__ == '__main__':
    args = parser.parse_args()
    args.save_models = 0
    
    if args.train_models:
        args.save_models = 1
        saved_supervised_model_results = data.SaveSupervisedModelResults(args.folder_name)
        saved_supervised_model_results.check_folder_exists()

#%% Manual Testing - turn train models on while keeping save models off then modify as you like

#args.train_models = 1
#saved_supervised_model_results = data.SaveSupervisedModelResults('test')
#
#args.encoder_model = 'VanillaEncoder'
#args.decoder_model = 'AttnDecoder'
#
#args.n_iterations = 3
#args.n_epochs = 100

#%% Define models and training / evaluation functions

class DecoderRNN(nn.Module):
    """Vanilla decoder which decodes based on single context vector"""
    def __init__(self, embedding_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.name = 'VanillaDecoderRNN'
        self.uses_attention = False
        self.is_agent = False
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class AttnDecoderRNN(nn.Module):
    """Attention decoder which decodes based on trained weightings over INPUT word context vectors. 
    Does not currently attend over generated text while decoding"""
    def __init__(self, embedding_size, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH+1):
        super(AttnDecoderRNN, self).__init__()
        self.name = 'AttentionDecoderRNN'
        self.uses_attention = True
        self.is_agent = False
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class Optimizers(nn.Module):
    """Defines optimizer object used in training, 
        particularly relevant when using optimizer switching"""
    def __init__(self, encoder, decoder, switch_thresh=0.05):
        super(Optimizers, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if encoder.trainable_model:
            self.encoder_optimizer = optim.Adam(encoder.parameters())
            self.encoder_opt_name = 'Adam'
        else:
            self.encoder_optimizer = 'Null'
            self.encoder_opt_name = 'Null'
        self.decoder_optimizer = optim.Adam(decoder.parameters())
        self.decoder_opt_name = 'Adam'
        self.training_loss = deque(maxlen=5)
        self.switch_thresh = switch_thresh
        self.enable_switch = True
    
    def optimizer_switch(self, force_switch_opt=False):
        """Designed to switch between Adam and SGD optimizers after initial reduction in performance"""
        # Change the optimizer to SGD if the difference between the min and mean is below threshold
        threshold_condition = False
        if len(self.training_loss) == self.training_loss.maxlen:
            threshold = (np.min(self.training_loss) / np.mean(self.training_loss)) -1
            threshold_condition = threshold < self.switch_thresh
            
        if threshold_condition or force_switch_opt:
            if self.encoder.trainable_model:
                self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=0.01)
                self.encoder_opt_name = 'SGD'
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=0.01)
            self.decoder_opt_name = 'SGD'
            self.enable_switch = False

class Teacher_Forcing_Ratio(object):
    """Defines the teacher forcing ratio which decays when update method is called"""
    def __init__(self, start_teacher_forcing_ratio, end_teacher_forcing_ratio, n_iterations):
        self.start_teacher_forcing_ratio = start_teacher_forcing_ratio
        self.end_teacher_forcing_ratio = end_teacher_forcing_ratio
        self.teacher_decay = (self.start_teacher_forcing_ratio - self.end_teacher_forcing_ratio) / n_iterations
        self.teacher_forcing_ratio = self.start_teacher_forcing_ratio
    
    def update_teacher_forcing_ratio(self):
        self.teacher_forcing_ratio = np.max([self.teacher_forcing_ratio - self.teacher_decay,
                               self.end_teacher_forcing_ratio])

class BeamSearchNode(object):
    """Beam node used in beam decoding implementation"""
    """ Notebook source: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py """
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(input_pair, encoder, decoder, beam_width=5, n_output_sentences=1, encoder_outputs=None):
    """Implements beam search decoding using specified encoder, decoder, and beam length"""
    """ Notebook source: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py """
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and
    T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, 
    [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    assert beam_width > 1, 'Beam width must be greater than 1'
    
    if encoder.trainable_model:
        input_tensor, _ = utils.tensorsFromPair(input_pair)
        
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(MAX_LENGTH+1, encoder.hidden_size, device=DEVICE)
    
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
    
        decoder_hidden = encoder_hidden
    
    else:
        decoder_hidden = encoder.sentence_embedding(input_pair[0])
        decoder_hidden = layer_normalize(decoder_hidden)
    
    topk = n_output_sentences  # how many sentence do you want to generate
    
    # Start with the start of the sentence token
    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    
    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    for _ in range(2000):
        # give up when decoding takes too long
        if qsize > 1000: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == EOS_token and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        if decoder.uses_attention:
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

        # do actual beam search
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()
            
            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, next_node = nextnodes[i]
            nodes.put((score, next_node))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.append(utterance)
    
    output_sentences = []
    for sentence in utterances:
        output_words = [vocab_index.index2word[word_idx.item()] for word_idx in sentence]
        output_sentences.append(' '.join(output_words[1:-1]))
    
    return output_sentences

def asMinutes(s):
    """Converts time to minutes for print output"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """Tracks time change for print output"""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showAttention(input_sentence, output_words, attentions):
    """Helper function which creates graph for plotting attention alignments table"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_pair, encoder, decoder):
    """Creates output which shows the attention alignments between the generated text and input"""
    output_words, attentions = evaluate(input_pair, encoder, decoder, 
                                           max_length=MAX_LENGTH)
    print('input =', input_pair[0])
    print('output =', ' '.join(output_words))
    showAttention(input_pair[0], output_words, attentions)

def n_model_parameters(model):
    """Returns the number of trainable parameters for a given model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class LayerNorm(nn.Module):
    """Handles layer normalization used to normalize features between RNN layers"""
    """ Source: https://github.com/pytorch/pytorch/issues/1959"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def layer_normalize(input_features, output_shape = (1,1,-1)):
    """Applies layer normalization used to normalize features between RNN layers"""
    # Initialize layernorm object
    layer_norm = LayerNorm(input_features.squeeze().shape).to(DEVICE)
    
    # Normalize features and reshape
    normalized_features = layer_norm(input_features.squeeze().float())
    normalized_features = normalized_features.view(output_shape).detach()
    return normalized_features

def train_encoder(input_tensor, encoder, encoder_optimizer, max_length=MAX_LENGTH):
    """Initializes encoder model and generates encoder preds for model training"""
    # Initialize empty hidden layer 
    encoder_hidden = encoder.initHidden()

    # Clear the gradients from the optimizers
    encoder_optimizer.zero_grad()
    
    # Initialize arrays for vector length of input and target
    input_length = input_tensor.size(0)
    
    # Initialize encoder outputs and instantiate the loss (includes padding room)
    encoder_outputs = torch.zeros(max_length+1, encoder.hidden_size, device=DEVICE)
    
    # Encodes input tensor into [len x hidden layer] sentence embedding matrix
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        
        encoder_outputs[ei] = encoder_output[0, 0]
    
    return encoder_hidden, encoder_outputs, encoder_optimizer

#%%
def train(input_pair, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):
    """Model training logic, initializes graph, creates encoder outputs matrix for attention model,
    applies teacher forcing (randomly), calculates the loss and trains the models"""
    if encoder.trainable_model:
        # Encode sentences using encoder model
        input_tensor, target_tensor = utils.tensorsFromPair(input_pair)
        decoder_hidden, encoder_outputs, encoder_optimizer = train_encoder(
                    input_tensor, encoder, encoder_optimizer, max_length)
    else:
        # Encode sentences using pretrained encoder model
        target_tensor = utils.tensorFromSentence(vocab_index, input_pair[1])
        decoder_hidden = encoder.sentence_embedding(input_pair[0])
        decoder_hidden = layer_normalize(decoder_hidden)
    
    # Clear the gradients from the decoder optimizer
    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(0)
    
    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    loss = 0
    
    # Randomly apply teacher forcing subject to teacher forcing ratio
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if decoder.uses_attention:
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing: set next input to correct target

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if decoder.uses_attention:
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
            
    # Calculate the error and blackpropogate through the network 
    loss.backward()
    
    if encoder.trainable_model:
        encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(input_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               n_iters=5000, print_every=1000, teacher_forcing_ratio=0.9):
    """Training loop including setting optimizers and loss function, currently trains
    online using each example instead of batches"""
    start = time.time()
    
    print_loss_total = 0  # Reset every print_every
    
    criterion = nn.NLLLoss()
    
    # Sample n random pairs
    training_pairs = data.sample_list(input_pairs, n_iters)
    
    # For EACH pair train model to decrease loss
    for idx, pair in enumerate(training_pairs):
        loss = train(pair, encoder, decoder, encoder_optimizer, 
                     decoder_optimizer, criterion, teacher_forcing_ratio)
        print_loss_total += loss
        
        iter = idx+1
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            if opt.enable_switch:
                opt.training_loss.append(print_loss_avg)
            if args.save_models:
                saved_supervised_model_results.train_loss.append(np.around(print_loss_avg,2))
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

def embed_input_sentence(input_pair, encoder, max_length=MAX_LENGTH):
    """Embeds the input sentence using a trained encoder model"""
    with torch.no_grad():
        if encoder.trainable_model:
            input_tensor, target_tensor = utils.tensorsFromPair(input_pair)
            
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length+1, encoder.hidden_size, device=DEVICE)
    
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
    
            decoder_hidden = encoder_hidden
            
            return decoder_hidden, target_tensor, encoder_outputs
            
        else:
            target_tensor = utils.tensorFromSentence(vocab_index, input_pair[1])
            decoder_hidden = encoder.sentence_embedding(input_pair[0])
            decoder_hidden = layer_normalize(decoder_hidden)
        
            return decoder_hidden, target_tensor, None
        
def evaluate(input_pair, encoder, decoder, max_length=MAX_LENGTH):
    """Generates the supervised prediction for a given input sentence"""
    if encoder.trainable_model:
        decoder_hidden, target_tensor, encoder_outputs = embed_input_sentence(input_pair, encoder, max_length)
    else:
        decoder_hidden, target_tensor, _ = embed_input_sentence(input_pair, encoder, max_length)
    
    with torch.no_grad():
        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS

        decoded_words = []
        decoder_attentions = torch.zeros(max_length+1, max_length+1)

        for di in range(max_length-1):
            if decoder.uses_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.data.topk(1)
            
            decoder_input = topi.squeeze().detach()  # AL # detach from history as input
            
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab_index.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        
        if decoder.uses_attention:
            return decoded_words, decoder_attentions[:di + 1]
        else:
            return decoded_words, 'None'
        
def evaluateError(input_pair, encoder, decoder, max_length=MAX_LENGTH):
    """Generates the predictions as well as the error using teacher forcing"""
    criterion = nn.NLLLoss()
    
    if encoder.trainable_model:
        decoder_hidden, target_tensor, encoder_outputs = embed_input_sentence(input_pair, encoder, max_length)
    else:
        decoder_hidden, target_tensor, _ = embed_input_sentence(input_pair, encoder, max_length)
    
    target_length = target_tensor.size(0)
    with torch.no_grad():   
        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS

        decoded_words = []
        loss = 0
        
        use_teacher_forcing = True if random.random() < tf_ratio.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                if decoder.uses_attention:
                        decoder_output, decoder_hidden, _ = decoder(
                            decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = decoder(
                            decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di]) # AL
                decoder_input = target_tensor[di]
                decoded_words.append(vocab_index.index2word[target_tensor[di].item()])
        else:
            for di in range(max_length):
                if decoder.uses_attention:
                    decoder_output, decoder_hidden, _ = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                
                topv, topi = decoder_output.data.topk(1)
                
                decoder_input = topi.squeeze().detach()  # AL # detach from history as input
                
                if di < len(target_tensor):
                    loss += criterion(decoder_output, target_tensor[di]) # AL
                
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(vocab_index.index2word[topi.item()])
    
        return decoded_words, loss.item() / target_length

def generateSentences(input_pairs, encoder, decoder, n=10):
    """Generates sentences based on encoder and decoder models given an input"""
    for i in range(n):
        pair = random.choice(input_pairs)
        print('>', pair[0])
        print('=', pair[1])
        
        output_words = evaluate(pair, encoder, decoder)[0]
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def validationError(input_pairs, encoder, decoder, verbose=True):
    """Evalutes the error on a set of input pairs in terms of loss. 
    Is intended to be used on a validation or test set to evaluate performance"""
    loss = 0
    
    for pair in input_pairs:
        _, pair_loss = evaluateError(pair, encoder, decoder)
        loss += pair_loss
        
    avg_loss = loss / len(input_pairs)
    
    if verbose:
        print('The average validation loss is {:.3} based on {} samples'.format(avg_loss, len(input_pairs)))
    return avg_loss

def predict_sentences(input_pairs, encoder, decoder):
    """Predicts the generated outputs for a set of input sentences"""
    pred_pairs = []
    
    for pair in input_pairs:
        output_words = evaluate(pair, encoder, decoder)[0]
        output_sentence = ' '.join(output_words)
        output_sentence = re.sub(r" <EOS>", r"", output_sentence)
        pred_pairs.append([pair[1], output_sentence])
    return np.array(pred_pairs)

def validationMetricPerformance(input_pairs, encoder, decoder, similarity_model=None, fluency_model=None,
                                ESIM_model=None, logr_model=None, std_scaler=None, 
                                similarity_dist=None, fluency_dist=None, ESIM_dist=None,
                                vocab_index=vocab_index, verbose=True, metric='BLEU1'):
    """Returns the model performance on a specified reward metric"""
    pred_pairs = predict_sentences(input_pairs, encoder, decoder)
    
    metrics_perf = np.array([model_evaluation.performance_metrics(
            target_sent, pred_sent, similarity_model=similarity_model, fluency_model=fluency_model,
            ESIM_model=ESIM_model, logr_model=logr_model, std_scaler=std_scaler,
            similarity_dist=similarity_dist, fluency_dist=fluency_dist, 
            ESIM_dist=ESIM_dist, vocab_index=vocab_index,
            metric=metric) for (target_sent, pred_sent) in pred_pairs])
    
    mean_performance = metrics_perf.mean()
    
    if verbose:
        print('Average metric performance: {}'.format(mean_performance))
    
    return pred_pairs, metrics_perf, mean_performance

def model_pipeline(n_iterations, encoder, decoder, encoder_optimizer, decoder_optimizer, n_epochs=5000):
    """Model pipeline which trains model and also generates examples while training and evaluation 
    on the validation set for potential early stopping. The teacher forcing ratio can also be 
    adjusted stepwise as desired"""
    
    for i in range(n_iterations):
        print('-------------------- Iteration {}: --------------------'.format(i+1))
        print('    Teacher Forcing Ratio: {:.2} | Optimizer: {}'.format(
                tf_ratio.teacher_forcing_ratio, opt.decoder_opt_name))
        if opt.enable_switch:
            opt.optimizer_switch()
        trainIters(train_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, n_epochs,
                   print_every=1000, teacher_forcing_ratio=tf_ratio.teacher_forcing_ratio)
        generateSentences(val_pairs, encoder, decoder, n=3)
        avg_val_loss = validationError(data.sample_list(val_pairs, sample_by_prop=True, sample_prop=0.20),
                                       encoder, decoder)
        tf_ratio.update_teacher_forcing_ratio()
        
        saved_supervised_model_results.val_loss.append(np.around(avg_val_loss, 2))
        
        if args.save_models:
            saved_supervised_model_results.export_loss('training_loss.txt', 'val_loss.txt')
            
            if avg_val_loss <= saved_supervised_model_results.val_loss_thresh:
                if encoder.trainable_model:
                    saved_supervised_model_results.save_top_models(encoder, 'encoder_{:.3f}.pt'.format(avg_val_loss))
                    saved_supervised_model_results.save_top_models(decoder, 'decoder_{:.3f}.pt'.format(avg_val_loss))
                else:
                    saved_supervised_model_results.save_top_models(decoder, 'decoder_{:.3f}.pt'.format(avg_val_loss))
                    
        if (len(saved_supervised_model_results.val_loss[:-1]) > 0) and \
        (avg_val_loss > min(saved_supervised_model_results.val_loss[:-1])):
            args.early_stoppage_holdoff -= 1
            if args.early_stoppage_holdoff <= 0:
                break

def define_encoder(encoder_name='VanillaEncoder'):
    """Initializes the encoder model"""
    # Instantiate encoder and decoder models
    if encoder_name == 'VanillaEncoder':
        return encoder_models.EncoderRNN(input_size=vocab_index.n_words,
                                               embedding_size=args.embedding_size,
                                               hidden_size=args.hidden_size).to(DEVICE)
    
    elif encoder_name == 'InitializedEncoder':
        return encoder_models.InitializedEncoderRNN(
                input_size=vocab_index.n_words, embedding_size=300, hidden_size=args.hidden_size, 
                caption_vocab_index=vocab_index, freeze_weights=True).to(DEVICE)
        
    elif encoder_name == 'GloveEncoder':
        return encoder_models.GloveEncoder()
    
    elif encoder_name == 'InferSentEncoder':
        return encoder_models.InferSentEncoder()
    
    elif encoder_name == 'BERTEncoder':
        return encoder_models.BERTEncoder()
        
    else:
        print("Please specify one of the following encoders: {}".format(
                ['VanillaEncoder']+encoder_models.pretrained_models_list))

def define_decoder(encoder, decoder_name='VanillaDecoder'):
    """Initializes the decoder model"""
    hidden_size = encoder.hidden_size
    decoder_embedding_size = args.embedding_size

    if decoder_name == 'VanillaDecoder':
        return DecoderRNN(embedding_size=decoder_embedding_size, hidden_size=hidden_size, 
                              output_size=vocab_index.n_words).to(DEVICE)
    elif decoder_name == 'AttnDecoder':
        return AttnDecoderRNN(embedding_size=decoder_embedding_size, hidden_size=hidden_size, 
                                       output_size=vocab_index.n_words, dropout_p=0.1).to(DEVICE)
    else:
        print("""Please specify one of the following decoders: 
            ['VanillaDecoder', 'AttnDecoder']""")

def load_supervised_models(folder_name, encoder_file_name='best', decoder_file_name='best'):
    """Initializes the encoder and decoder models and loads the weights from the trained models"""
    hidden_size=256; embedding_size=256;
    if encoder_file_name in encoder_models.pretrained_models_list:
        encoder = define_encoder(encoder_file_name)
        decoder = DecoderRNN(embedding_size=embedding_size, hidden_size=encoder.hidden_size,
                                 output_size=vocab_index.n_words).to(DEVICE)
        if decoder_file_name == 'best':
            decoder_file_name = 'decoder_{:.3f}.pt'.format(data.get_top_n_models(
                    os.path.join(config.saved_supervised_model_path, folder_name), 'decoder', n=1, descending=False)[0])
            
        data.load_model(decoder, os.path.join(config.saved_supervised_model_path, folder_name, decoder_file_name))
        return encoder, decoder
        
    else:
        if folder_name in ['Baseline_Test', 'VanillaEncoder', 'VanillaEncoder_Adam',
                           'VanillaEncoder_Switch', 'VanillaEncoder_TF']:
            pass
        
        elif folder_name in ['Attention_SGD', 'Attention_Adam']:
            decoder = AttnDecoderRNN(embedding_size=embedding_size, hidden_size=hidden_size,
                                     output_size=vocab_index.n_words).to(DEVICE)
        
        else:
            raise SystemExit('Please correct test folder name')
        
        if 'encoder' not in locals():
            encoder = encoder_models.EncoderRNN(input_size=vocab_index.n_words,
                                               embedding_size=embedding_size, hidden_size=hidden_size).to(DEVICE)    
        if 'decoder' not in locals():
            decoder = DecoderRNN(embedding_size=embedding_size, hidden_size=hidden_size,
                                 output_size=vocab_index.n_words).to(DEVICE)
        
        if encoder_file_name == 'best':
            encoder_file_name = 'encoder_{:.3f}.pt'.format(data.get_top_n_models(
                    os.path.join(config.saved_supervised_model_path, folder_name), 'encoder', n=1, descending=False)[0])
        
        if decoder_file_name == 'best':
            decoder_file_name = 'decoder_{:.3f}.pt'.format(data.get_top_n_models(
                    os.path.join(config.saved_supervised_model_path, folder_name), 'decoder', n=1, descending=False)[0])
        
        data.load_model(encoder, os.path.join(config.saved_supervised_model_path, folder_name, encoder_file_name))
        data.load_model(decoder, os.path.join(config.saved_supervised_model_path, folder_name, decoder_file_name))
        
        return encoder, decoder

def test_sentence_evaluation(input_pair, encoder, decoder):
    """Produces a predicted sentence using both MLE and beam search"""
    print('Input Sentence: ', input_pair[0])
    print('Target Sentence: ', input_pair[1])
    print()
    print('MLE Model Output:',' '.join(evaluate(input_pair, encoder, decoder)[0]))
    print('Beam Search Model Output:', beam_decode(input_pair=input_pair, encoder=encoder, decoder=decoder, 
                beam_width=4, n_output_sentences=1, encoder_outputs=None))
    
#    if decoder.uses_attention:
#        evaluateAndShowAttention(input_pair, encoder, decoder)

#%% Train and Evaluate Model

if (__name__ == '__main__') and args.train_models:
    """Initializes models subject to cmd line args and then trains and evaluates performance"""
    
    # Initialize encoder and decoder models
    encoder = define_encoder(encoder_name=args.encoder_model)
    decoder = define_decoder(encoder, decoder_name=args.decoder_model)
    
    # Load saved weights from specified model
    if args.load_models:
        if args.encoder_model in encoder_models.pretrained_models_list:
            encoder, decoder = load_supervised_models(
                    args.load_model_folder_name, encoder_file_name=args.encoder_model,
                    decoder_file_name='best')
        else:
            encoder, decoder = load_supervised_models(
                    args.load_model_folder_name, encoder_file_name='best',
                    decoder_file_name='best')
    
    # Initialize folder if saving models
    if args.save_models:
        saved_supervised_model_results.init_folder(args, encoder, decoder)
    
    
    # Specify numberr of iterations and the optimizer used
    n_iterations = args.n_iterations
    
    opt = Optimizers(encoder, decoder, switch_thresh=0.05)
    if args.optimizer == 'SGD':
        opt.optimizer_switch(force_switch_opt=True)
    elif args.optimizer == 'Adam':
        opt.enable_switch = False   
    else:
        pass
    
    # Initialize teacher forcing object as well as key args
    tf_ratio = Teacher_Forcing_Ratio(start_teacher_forcing_ratio=args.start_tf_ratio,
                                     end_teacher_forcing_ratio=args.end_tf_ratio,
                                     n_iterations=args.tf_decay_iters)
    
    # Train model subject to args and save if error
    try:
        model_pipeline(n_iterations=n_iterations, encoder=encoder, decoder=decoder,
                       encoder_optimizer=opt.encoder_optimizer, decoder_optimizer=opt.decoder_optimizer,
                       n_epochs=args.n_epochs)
    except:
        if args.save_models:
            saved_supervised_model_results.export_loss('training_loss.txt', 'val_loss.txt')
            
            try:
                saved_supervised_model_results.save_top_models(encoder, 'encoder_CHECKPOINT.pt')
                saved_supervised_model_results.save_top_models(decoder, 'decoder_CHECKPOINT.pt')
            except:
                saved_supervised_model_results.save_top_models(decoder, 'decoder_CHECKPOINT.pt')
