# ParaPhrasee - Paraphrase Generation Using Deep Reinforcement Learning

The thesis and repo associated with the article Paraphrase Generation Using Deep Reinforcement Learning.

The code is not intended to run end-to-end for new applications and is instead meant to be used as starter code or for taking code snippets. 

On page 29 of the thesis there is a full list of the different modules and a brief description. Please send me an email at andrewgbravo@gmail.com if you would like one of the specific modules not contained in the repo.

The key modules included are:
- data: Imports raw data from various sources, preprocesses, creates train/validation/test sets and vocab index. Also contains functions for saving and loading
- encoder_models: Defines classes for each encoder: GloVe, BERT, InferSent, Vanilla and GPT along with related code for retrieving embeddings
- MCTS: Monte Carlo Tree Search implementation for both FrozenLake and ParaPhrasee environments
- model_evaluation: Contains wrappers for different evaluation functions to be used primarily as reward functions for RL model
- paraphrasee_env: Defines environment dynamics for paraphrase generation task as RL problem
- RL_model: Defines and trains RL models for ParaPhrasee environment
- supervised_model: Defines, trains, and evaluates the defined supervised model with MLE and beam search. Includes modifications for teacher-forcing and attention
- toy_RL_pipeline: Defines and trains RL models for either CartPole or FrozenLake environments
- train_ESIM: Defines and trains an ESIM model for use as the discriminator / adversary
