"""Contains wrappers for different evaluation functions to be used primarily as 
    reward functions for RL model"""

import numpy as np

import os
import config
import utils
import data
from train_ESIM import ESIM_pred, load_ESIM_model
import encoder_models

#%% 

from eval_metrics.bleu.bleu_scorer import BleuScorer
from eval_metrics.rouge.rouge import Rouge
from eval_metrics.cider.cider_scorer import CiderScorer
from eval_metrics.meteor.meteor import Meteor

# Define conventional automatic metrics warppers from eval_metrics package
def BLEU_score(target_sentence, pred_sentence, n_tokens=1):
    """Returns BLEU score at specified n-gram level for a given target and predicted sentence pair"""
    try:    
        # Set n to BLEU score level
        bleu_scorer = BleuScorer(n=n_tokens)
        bleu_scorer += (pred_sentence[0], target_sentence)
        BLEU_score, _ = bleu_scorer.compute_score()
        return np.around(BLEU_score[n_tokens-1], 4)
    except:
        print('rejected sentence: ', pred_sentence)

def ROUGE_score(target_sentence, pred_sentence):
    """Returns ROUGE score for a given target and predicted sentence pair"""
    try:
        rouge = Rouge()
        ROUGE_score = rouge.calc_score(pred_sentence, target_sentence)
        return np.around(ROUGE_score, 4)
    except:
        print('rejected sentence: ', pred_sentence)
    
def CIDER_score(target_sentence, pred_sentence):
    """Returns CIDER score for a given target and predicted sentence pair"""
    try:
        cider_scorer = CiderScorer(idf_terms_path=os.path.join(config.PATH, 'eval_metrics/cider/idf_terms'))
        cider_scorer += (pred_sentence[0], target_sentence)
        CIDER_score, _ = cider_scorer.compute_score()
        return np.around(CIDER_score, 4)
    except:
        print('rejected sentence: ', pred_sentence)

def METEOR_score(target_sentence, pred_sentence):
    """Returns METEOR score for a given target and predicted sentence pair"""
    try:
        meteor = Meteor()
        METEOR_score, _ = meteor.compute_score(pred_sentence, 
                            target_sentence)
        return np.around(METEOR_score, 4)
    except:
        print("Java did not execute properly.")

#%% 

def tokenize(input_sentence):
    """Converts an input sentence to a set of tokens after applying preprocessing"""
    preprocessed_sentence = data.preprocess(input_sentence, remove_punct=True, lower_case=True)
    tokens = preprocessed_sentence.split()
    return tokens

def sentence_similarity(target, pred, similarity_model):
    """Calculates the cosine similarity between the sentence embeddings of a target and predicted pair 
        using the embedding model specified"""
    try:
        cosine_sim = utils.cosine_similarity(similarity_model.sentence_embedding(target).view(1,-1),
                                       similarity_model.sentence_embedding(pred).view(1,-1))
        return np.around(cosine_sim.item(),4)
    except:
        print('similarity rejected sentence: ', pred)
        return 0.01

def sentence_length(input_sentence, min_value=6, max_value=12):
    """Returns the sentence length score for use as an auxiliary reward function"""
    input_value = np.clip(len(input_sentence.split()), min_value, max_value)
    return 1 - ((input_value - min_value) / (max_value - min_value))

def avg_word_frequency(input_sentence, vocab_index, total_word_count):
    """Returns the average word frequency for use as an auxiliary reward function"""
    return np.mean([vocab_index.word2count[word] for word in input_sentence.split()]) / total_word_count
    
def rare_word_prop(input_sentence, vocab_index, rare_thresh=10):
    """Returns the proportion of rare words in a sentence for use as an auxiliary reward function"""
    try:
        assert len(input_sentence) > 0, 'Sentence is empty'
        input_words = input_sentence.split()
        n_rare_words = 0
        for word in input_words:
            if vocab_index.word2count[word] <= rare_thresh:
                n_rare_words += 1
            else:
                pass
        return n_rare_words / len(input_words)
    except:
        print('rejected sentence: ', input_sentence)
        return 0.01

def word_syllable_count(input_word):
    """Returns the syllable count for a word"""
    """source: https://stackoverflow.com/questions/46759492/syllable-count-in-python"""
    vowels = "aeiouyAEIOUY"
    count = 0
    prior_letter = None
    
    try:
        for idx, letter in enumerate(input_word):
            if (idx == 0) and (letter in vowels):
                count += 1
            elif (letter in vowels) and (prior_letter not in vowels):
                count += 1
            prior_letter = letter
            
        if input_word.endswith("e"):
            count -= 1
        return max(1, count)
    except:
        print('rejected word: ', input_word)
        return 1

def scaled_sent_syllable_count(input_sentence, max_avg_n_syllables=2):
    """Returns the average syllable count for a sentence for use as an auxiliary reward function"""
    try:
        assert len(input_sentence) > 0, 'Sentence is empty'
        avg_syllable_count = np.mean([word_syllable_count(word) for word in input_sentence.split()])
        return np.min([(avg_syllable_count / max_avg_n_syllables), 1])
    except:
        print('rejected sentence: ', input_sentence)
        return 0.01

def libertarian_pred(input_sentence, BERT_model, logr_model, std_scaler):
    """Returns the probability a given sentence is a comment from a Libertarian subreddit rather than 
        an Anarchist / Socialist subreddit based on a trained model for use as an auxiliary reward function"""
    try:
        encoded_sent = BERT_model.sentence_embedding(input_sentence).reshape(1, -1)
        standardized_sent = std_scaler.transform(encoded_sent)
        probs = logr_model.predict_proba(standardized_sent)
        return probs[0][1]
    
    except:
        print('rejected political sentence: ', input_sentence)
        return 0.01

def performance_metrics(target_sentence, pred_sentence, similarity_model=None, fluency_model=None,
                        ESIM_model=None, logr_model=None, std_scaler=None, similarity_dist=None, fluency_dist=None,
                        ESIM_dist=None, vocab_index=None, metric='BLEU1'):
    """The main pipeline which handles applying the appropriate reward function and handles 
        the relevant models / data inputs required"""
    if metric == 'BLEU1':
        return BLEU_score(target_sentence=[target_sentence], pred_sentence=[pred_sentence], n_tokens=1)
    
    if metric == 'BLEU2':
        return BLEU_score(target_sentence=[target_sentence], pred_sentence=[pred_sentence], n_tokens=2)
    
    elif metric == 'ROUGE':
        return ROUGE_score(target_sentence=[target_sentence], pred_sentence=[pred_sentence])
    
    elif metric == 'CIDER':
        return CIDER_score(target_sentence=[target_sentence], pred_sentence=[pred_sentence])
    
    elif metric == 'METEOR':
        return METEOR_score(target_sentence=[target_sentence], pred_sentence=[pred_sentence])
    
    elif metric == 'FLUENCY':
        return fluency_model.fluency_score(pred_sentence)
    
    elif metric == 'PARA':
        return sentence_similarity(target_sentence, pred_sentence, similarity_model)
    
    elif metric == 'PARA_F':
        similarity_score = sentence_similarity(target_sentence, pred_sentence, similarity_model)
        fluency_score = fluency_model.fluency_score(pred_sentence)
        
        scaled_similarity_score = utils.cdf_score(similarity_dist, similarity_score)
        scaled_fluency_score = utils.cdf_score(fluency_dist, fluency_score)
        
        return np.mean([scaled_similarity_score, scaled_fluency_score])
    
    elif metric == 'PARASIM':
        similarity_score = sentence_similarity(target_sentence, pred_sentence, similarity_model)
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        
        scaled_similarity_score = utils.cdf_score(similarity_dist, similarity_score)
        scaled_ESIM_score = utils.cdf_score(ESIM_dist, ESIM_score)
        
        return np.mean([scaled_similarity_score, scaled_ESIM_score])
    
    elif metric == 'PARASIM_F':
        similarity_score = sentence_similarity(target_sentence, pred_sentence, similarity_model)
        fluency_score = fluency_model.fluency_score(pred_sentence)
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        
        scaled_similarity_score = utils.cdf_score(similarity_dist, similarity_score)
        scaled_fluency_score = utils.cdf_score(fluency_dist, fluency_score)
        scaled_ESIM_score = utils.cdf_score(ESIM_dist, ESIM_score)
        
        return np.mean([scaled_similarity_score, scaled_fluency_score, scaled_ESIM_score])

    elif metric == 'ESIM':
        return ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
    
    elif metric == 'ESIM_short':
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        length_score = sentence_length(pred_sentence)

        return 0.6 * ESIM_score + 0.4 * length_score
    
    elif metric == 'ESIM_syllables':
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        syllable_score = scaled_sent_syllable_count(pred_sentence, max_avg_n_syllables=2)

        return 0.6 * ESIM_score + 0.4 * syllable_score
    
    elif metric == 'ESIM_rare':
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        rare_score = rare_word_prop(pred_sentence, vocab_index, rare_thresh=2500)

        return 0.6 * ESIM_score + 0.4 * rare_score
    
    elif metric == 'ESIM_F':
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        fluency_score = fluency_model.fluency_score(pred_sentence)
        
        scaled_ESIM_score = utils.cdf_score(ESIM_dist, ESIM_score)
        scaled_fluency_score = utils.cdf_score(fluency_dist, fluency_score)
        
        return np.mean([scaled_ESIM_score, scaled_fluency_score])
    
    elif metric == 'ESIM_libertarian':
        ESIM_score = ESIM_pred([[target_sentence, pred_sentence]], ESIM_model, temperature=2).item()
        libertarian_score = libertarian_pred(pred_sentence, similarity_model, logr_model, std_scaler)
        
        return 0.6 * ESIM_score + 0.4 * libertarian_score

def init_eval_models(reward_function='BLEU1', similarity_model_name='BERT', ESIM_model_name='ESIM_noisy_3'):
    """Initializes the appropriate models / data for use in the performance metrics evaluation. 
        The following fields are being initialized: 
        similarity_model, fluency_model, ESIM_model, \
        logr_model, std_scaler, \
        similarity_dist, fluency_dist, ESIM_dist"""
    if reward_function == 'FLUENCY':
        return None, encoder_models.GPTLanguageModel(), None, \
                            None, None, \
                            None, None, None
    
    if reward_function == 'PARA':
        if similarity_model_name =='BERT':
            return encoder_models.BERTEncoder(), None, None, \
                            None, None, \
                            None, None, None
            
        elif similarity_model_name =='InferSent':
            return encoder_models.InferSentEncoder(), None, None, \
                    None, None, \
                    None, None, None
    
    elif reward_function == 'PARA_F':
        if similarity_model_name =='BERT':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'BERT_dist.npy'))
            fluency_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'fluency_dist.npy'))
            return encoder_models.BERTEncoder(), encoder_models.GPTLanguageModel(), None, \
                    None, None, \
                    similarity_dist, fluency_dist, None
        
        elif similarity_model_name =='InferSent':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'InferSent_dist.npy'))
            fluency_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'fluency_dist.npy'))
            return encoder_models.InferSentEncoder(), encoder_models.GPTLanguageModel(), None, \
                    None, None, \
                    similarity_dist, fluency_dist, None
    
    elif reward_function == 'PARASIM':
        if similarity_model_name =='BERT':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'BERT_dist.npy'))
            ESIM_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'ESIM_dist.npy'))
            return encoder_models.BERTEncoder(), None, load_ESIM_model(ESIM_model_name), \
                    None, None, \
                    similarity_dist, None, ESIM_dist
        
        elif similarity_model_name =='InferSent':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'InferSent_dist.npy'))
            ESIM_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'ESIM_dist.npy'))
            return encoder_models.InferSentEncoder(), None, load_ESIM_model(ESIM_model_name), \
                    None, None, \
                    similarity_dist, None, ESIM_dist
    
    elif reward_function == 'PARASIM_F':
        if similarity_model_name =='BERT':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'BERT_dist.npy'))
            fluency_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'fluency_dist.npy'))
            ESIM_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'ESIM_dist.npy'))
            return encoder_models.BERTEncoder(), encoder_models.GPTLanguageModel(), load_ESIM_model(ESIM_model_name), \
                    None, None, \
                    similarity_dist, fluency_dist, ESIM_dist
        
        elif similarity_model_name =='InferSent':
            similarity_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'InferSent_dist.npy'))
            fluency_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'fluency_dist.npy'))
            ESIM_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'ESIM_dist.npy'))
            return encoder_models.InferSentEncoder(), encoder_models.GPTLanguageModel(), load_ESIM_model(ESIM_model_name), \
                    None, None, \
                    similarity_dist, fluency_dist, ESIM_dist
    
    elif reward_function == 'ESIM':
        return None, None, load_ESIM_model(ESIM_model_name), \
                None, None, \
                None, None, None
    
    elif reward_function == 'ESIM_short':
        return None, None, load_ESIM_model(ESIM_model_name), \
                None, None, \
                None, None, None
    
    elif reward_function == 'ESIM_rare':
        return None, None, load_ESIM_model(ESIM_model_name), \
                None, None, \
                None, None, None
    
    elif reward_function == 'ESIM_F':
        fluency_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'fluency_dist.npy'))
        ESIM_dist = data.load_np_data(os.path.join(config.saved_SM_dist_path, 'ESIM_dist.npy'))
        
        return None, encoder_models.GPTLanguageModel(), load_ESIM_model(ESIM_model_name), \
                None, None, \
                None, fluency_dist, ESIM_dist
    
    elif reward_function == 'ESIM_syllables':
        return None, None, load_ESIM_model(ESIM_model_name), \
                None, None, \
                None, None, None
    
    elif reward_function == 'ESIM_libertarian':
        logr_model = data.load_vocab_index(os.path.join(
                config.saved_reddit_model_path, 'anarsoc_libertar_logr.pickle'))
        
        std_scaler = data.load_vocab_index(os.path.join(
                config.saved_reddit_model_path, 'std_scaler.pickle'))
        
        return encoder_models.BERTEncoder(), None, load_ESIM_model(ESIM_model_name), \
                logr_model, std_scaler, \
                None, None, None

    else:
        return None, None, None, None, None, None, None, None

#%% ------------------------------- ARCHIVE -------------------------------

# Prior defined versions of BLEU and ROUGE

#from collections import Counter
#from rouge.rouge import rouge_n_sentence_level
#from nltk import bigrams
#
#def BLEU_score(target_tokens, pred_tokens, ngram = 'unigram'):
#    if ngram == 'bigram':
#        target_tokens = list(bigrams(target_tokens))
#        pred_tokens = list(bigrams(pred_tokens))
#    
#    word_counts = Counter(target_tokens)
#    score = 0
#    
#    for token in pred_tokens:
#        if word_counts[token] > 0:
#           word_counts[token] -=1 
#           score += 1
#    score /= len(target_tokens)
#    return score
#
#def ROUGE_score(target, pred, ngram = 'unigram'):
#    """ Note: rouge n_sentence_level has hypothesis and reference positions swapped"""
#    if ngram == 'unigram':
#        _, _, score = rouge_n_sentence_level(pred, target, 1)
#        return score
#    if ngram == 'bigram':
#        _, _, score = rouge_n_sentence_level(pred, target, 2)
#        return score
#    else:
#        print("Please select either: 'unigram' or 'bigram'")


