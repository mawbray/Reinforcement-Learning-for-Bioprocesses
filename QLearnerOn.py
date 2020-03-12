import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
eps  = np.finfo(float).eps

############# Defining Agent To Be Trained ##################
class greedye_QL:
    # defining training agent  
    def __init__(self, num_actions, modulus, state_UB, alpha, disc1, disc2, movements):
      self.name = "e-Greedy"
      self.num_actions, self.modulus, self.UB, self.movements  = num_actions, modulus, state_UB, movements
      self.alpha, self.disc1, self.disc2                        = alpha, disc1, disc2
      self.X_s = np.linspace(0, int(self.UB[0]), int(self.UB[0]/self.modulus[0] + 1), dtype = np.float64)
      self.N_s = np.linspace(0, int(self.UB[1]), int(self.UB[1]/self.modulus[1] + 1), dtype = np.float64)
      f1, f2 = str(self.modulus[0]), str(self.modulus[1])
      decimal1, decimal2 = f1[::-1].find('.'), f2[::-1].find('.') 
      self.sitevisit = {(X_si, N_si, ti): np.zeros((1)) for X_si in np.round(self.X_s,decimal1) for N_si in np.round(self.N_s,decimal2) for ti in np.arange(1,self.movements+1, dtype ="int")}
      self.d = {(X_si, N_si, ti): np.random.randn(num_actions) for X_si in np.round(self.X_s,decimal1) for N_si in np.round(self.N_s,decimal2) for ti in np.arange(1,self.movements+1, dtype ="int")}
     
    def act(self,state, eps_prob, s):                                      
      #e-greedy definition
      self.eps_prob = eps_prob
      time_to_term = int(self.movements - s)  
      if np.random.uniform(0,1) <= self.eps_prob:
        action = np.random.randint(0,self.num_actions)
      else: action = np.argmax(self.d[(state[0],state[1], time_to_term)])
      return action

    def Learn(self, state, action, s, reward=None):
        self.reward = reward
        XT, NT = state[-1,0], state[-1,1] 
        #takes some state and attributes discounted reward to action via QL update 
 
        for i in range(0,1):                 #attributing value to actions (hence shape-1)
          indx = action
          time_to_term = int(self.movements - s)
          if s < self.movements -1:
              opt_action = np.argmax(self.d[(state[i+1,0], state[i+1,1], int(time_to_term - 1))])
              opt_future = self.d[(state[i+1,0], state[i+1,1], int(time_to_term - 1))][opt_action]
          else: opt_future = 0
          #input of variational derivative
          dXdt, dNdt = (state[i+1,0] - state[i,0]),  (state[i+1,1] - state[i,1])
          dRdX, dRdN = 100, -1
          varderivX, varderivN = dXdt * dRdX * 1/(XT+1), dNdt * dRdN * 1/(NT+1)
          Rtp1 = (self.disc1**(time_to_term)) * (varderivX + varderivN)
          if Rtp1 < -1:
              Rtp1 = -1 
          elif Rtp1 > 1:
              Rtp1 = 1
          #print(indx)
          self.d[(state[i,0],state[i,1],time_to_term)][int(indx)] = self.d[(state[i,0],state[i,1], time_to_term)][int(indx)] * (1-self.alpha) + self.alpha * (Rtp1 + opt_future*self.disc2)
          return   
        
    def learned(self):
      dictionary = self.d
      return dictionary 
  
############ Defining learned Agent #################
class Greedye_QLearned:
  # defining trained agent
    def __init__(self, num_actions, d, movements):
      self.name = "e-Greedy"
      self.num_actions = num_actions
      self.d, self.movements = d, movements
     
    def act(self,state, eps_prob, s):                                      
      # e-greedy definition
      self.eps_prob = eps_prob
      time_to_term = int(self.movements - s)  
      if np.random.uniform(0,1) <= self.eps_prob:
        action = np.random.randint(0,self.num_actions)
      else: action = np.argmax(self.d[(state[0],state[1], time_to_term)])
      return action
