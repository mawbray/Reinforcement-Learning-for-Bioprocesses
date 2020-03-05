# Reinforcement-Learning-for-Bioprocesses
A study of reinforcement learning algorithms for continuous bioprocessesing

The following repository examines the implementation of various reinforcement learning algorithms (model-free) for online optimisation of continuous bioprocessing. The initial case study considers a simple fermenation model with reward allocated via backallocation from the terminal state and the associated satisfaction of some objective function.

The study considers:

 - Monte Carlo Learning 
 - SARSA
 - Q Learning
 
It aims to explore the importance of temporal credit assignment in a discrete environment, the efficacy of multiple agent reinforcement learning  (MARL), different MIMO, MISO and SIMO arrangements as well as looking towards continuous time operation. Currently, the repository only explores credit assignment within an SIMO environment.

ExpDesign.py file calls all other .py files and functions to train and validate the agents - it is curretly set up to only train and validate the Monte Carlo Learner, but this can be adjusted by altering the initial import within ExpDesign.py

Please see the google colab workbook for further details of implementation 
