# Importing (only for Monte Carlo Learner)

import numpy as np
import pandas as pd
import scipy.integrate as scp
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
eps  = np.finfo(float).eps
import csv

from Experiment_online import Experiment as ExpTrain
from QLearnerOn import greedye_QL, Greedye_QLearned
from Environment_Cont_x2_SIMO import Model_env
from ValidationExperiment import Experiment_Done as ExpValid

####################### -- Defining Parameters and Designing Experiement -- #########################
#investigating steps_, alpha and discount factor for 1,000,000 epochs of training 
# Model definitions: parameters, steps, tf, x0  
      
p        = {'u_m' : 0.0923*0.62, 'K_N' : 393.10, 'u_d' : 0.01, 'Y_nx' : 504.49}                 # model parameter definitions
steps_   = np.array([40])                                                                       # number of control interactions
tf       = 16.*24                                                 
x0       = np.array([0.5,150.0])                                                                # initial conditions of environment
modulus  = np.array([0.025, 5])                                                                 # line distance of state space
state_UB = np.array([5, 1000])                                                                  # state space upper bound

# Agent definitions: num_actions, eps_prob, alpha, discount
num_actions = 15                                                                                # number (range) of actions available to agent
alpha = 0.1
disc1 = np.array([0.55])                                                                        # discount factor in back-allocation
disc2 = np.array([0.99])                                                                        # discount factor in agent learning
xi_ = np.array([3])                                                                             # Epsilon greedy definition (from experiment)

# Experiment defintions: env, agent, controls, episodes
controls = np.linspace(0,7,num_actions)                                                         # defining range of controls
episodes_train = 1000000                                                                        # number of training epochs
episodes_valid = 1000                                                                           # numeber of validation epochs
reward_training = np.zeros((episodes_train, xi_.shape[0], disc1.shape[0], disc2.shape[0]))      # memory allocation 
reward_validation = np.zeros((episodes_valid, xi_.shape[0], disc1.shape[0], disc2.shape[0]))    # memory allocation 
bracket = int(1000)

def EpochNoMean(data, bracket):
    nrows = int(data.shape[0]/bracket)
    plot_prep_mean = np.zeros((int(nrows)))
    for f in range(0,nrows):
        x = data[f*bracket:f*bracket+ bracket-1]
        y = np.mean(x,0)
        plot_prep_mean[f] = y
    return plot_prep_mean

#plot of 1000 epoch mean throughout training
def Plotting(data, bracket, pNo_mean):
    plt.figure(figsize =(20,16))
    plt.scatter(np.linspace(0, len(data), len(data)), data, label= 'Mean R over 1000 epochs')
    plt.xlabel('Training epochs (1e3)',  fontsize=28)
    plt.ylabel('Mean reward over ' + str(bracket)+ ' epochs', fontsize=28)
    #plt.title('Mean Reward over ' + str(bracket) + ' epochs with training')
    plt.tick_params(labelsize=24)
    plt.savefig('D:\\Documents_Data\\UoM\\Projects\\First Year\\RL - Bioprocesses\\WorkBooks\\Reinforcement-Learning-for-Bioprocesses\\Accumulated_reward_' + str(pNo_mean) + 'QLCAP_1_d055_d099_40steps_1e6.png')

# running experiement
for i in range(0, xi_.shape[0]):
  for j in range(0, disc1.shape[0]):    
    for k in range(0,disc2.shape[0]): 
      #run training 
      env = Model_env(p, steps_, tf, x0, modulus)                                               # calling environment
      agent = greedye_QL(num_actions, modulus, state_UB, alpha, disc1[j], disc2[k], steps_)           # calling agent
      experiment = ExpTrain(env, agent, controls, episodes_train, xi_[i])                       # calling training experiment
      reward_training[:,i,j,k], d = experiment.simulation()                                     # running training experiment
      agent = Greedye_QLearned(num_actions, d, steps_)                                         # calling learned agent
      exp_done = ExpValid(env, agent, controls, episodes_valid, 0)                              # calling validation experiment
      reward_validation[:,i,j,k] = exp_done.simulation()                                        # running validation experiment
      reward_train_mean = EpochNoMean(reward_training[:,i,j,k],bracket)
      x_o = Plotting(reward_train_mean, bracket, "rule_allocation")

