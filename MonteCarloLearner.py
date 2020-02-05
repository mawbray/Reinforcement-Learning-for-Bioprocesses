import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
eps  = np.finfo(float).eps



############# Defining Agent To Be Trained ##################
class greedye_MCL:
    def __init__(self, num_actions, modulus, state_UB, disc1, disc2, movements):
      self.name = "e-Greedy"
      # defining number of actions available to agent, the line distance of the state space, the upper bound of the state space and the number of s-a pairs observed per sequence (from t = 0...T)
      self.num_actions, self.modulus, self.UB, self.movements  = num_actions, modulus, state_UB, movements    
      self.disc1, self.disc2 = disc1, disc2                                                                       # defining discount factors
      self.X_s = np.linspace(0, int(self.UB[0]), int(self.UB[0]/self.modulus[0] + 1), dtype = np.float64)
      self.N_s = np.linspace(0, int(self.UB[1]), int(self.UB[1]/self.modulus[1] + 1), dtype = np.float64)
      f1, f2 = str(self.modulus[0]), str(self.modulus[1])
      decimal1, decimal2 = f1[::-1].find('.'), f2[::-1].find('.')                                           
      # defining state-action value dictionary based on grid definition from X_s and N_s
      self.d = {(X_si, N_si, ti): np.random.randn(num_actions) for X_si in np.round(self.X_s,decimal1) for N_si in np.round(self.N_s,decimal2) for ti in np.arange(1,self.movements+1, dtype ="int")}
      # initialising count for observation of each state-action pair
      self.dcount = {(X_si, N_si, ti): np.zeros(num_actions) for X_si in np.round(self.X_s,decimal1) for N_si in np.round(self.N_s,decimal2) for ti in np.arange(1,self.movements+1, dtype ="int")}
     
    def act(self,state, eps_prob, s):                                      
      # e-greedy definition
      self.eps_prob = eps_prob
      time_to_term = int(self.movements - s)                          
      if np.random.uniform(0,1) <= self.eps_prob:
        action = np.random.randint(0,self.num_actions)
      else: action = np.argmax(self.d[(state[0],state[1], time_to_term)])
      return action

    def Learn(self, state, action, reward):
        self.reward = reward
        XT, NT = state[-1,0], state[-1,1] 
        # takes some state and attributes discounted reward to action via QL update 
        for i in range(0,action.shape[0]):                 # attributing value to actions
          indx = action[i]
          time_to_term = int(self.movements - i)           # finding time index
          Gt = 0                                           # return at time t = 0
          W = 1
          self.dcount[(state[i,0],state[i,1],time_to_term)][int(indx)] = self.dcount[(state[i,0],state[i,1],time_to_term)][int(indx)] + 1      #updatingcount
          if i < action.shape[0]-1:
              # finding return from time,t=i in sequence
              for j in range(i, action.shape[0]):
                  time_to_ter = int(self.movements - j)                                         # defining time to termination (time index)
                  dXdt, dNdt = (state[j+1,0] - state[j,0]),  (state[j+1,1] - state[j,1])
                  dRdX, dRdN = 100, -1
                  varderivX, varderivN = dXdt * dRdX, dNdt * dRdN         # calculating variational derivative of action at time t = i
                  Rtp1 =  (self.disc1**time_to_ter)*(varderivX + varderivN)     # allocating reward based on some backallocation function discounted to time and implementation of variational derivative
                  Gt += Rtp1*self.disc2**(j-i)                                                  # updating observed return and discouting via \gamma_{2}
          else:
              dXdt, dNdt = (state[j+1,0] - state[j,0]),  (state[j+1,1] - state[j,1])
              dRdX, dRdN = 100, -1
              varderivX, varderivN = dXdt * dRdX , dNdt * dRdN 
              Rtp1 = (self.disc1**time_to_term)*(varderivX + varderivN)
              Gt += Rtp1                                                                        # for last action in sequence expected return is equivalent to reward observed
          alpha = W / self.dcount[(state[i,0],state[i,1],time_to_term)][int(indx)]              # updating learning parameter based on number s-a pair count
          self.d[(state[i,0],state[i,1],time_to_term)][int(indx)] = self.d[(state[i,0],state[i,1], time_to_term)][int(indx)] * (1-alpha) + alpha * (Gt)          
          #print(Gt)
          return   
          
    def learned(self):
      dictionary = self.d
      return dictionary 
  
############ Defining learned Agent #################
class Greedye_MCLearned:
  # defining trained agent
    def __init__(self, num_actions, d, movements):
      self.name = "e-Greedy"
      self.num_actions = num_actions
      self.d, self.movements = d, movements
     
    def act(self,state, eps_prob, s):                                      
      #e-greedy definition
      self.eps_prob = eps_prob
      time_to_term = int(self.movements - s)  
      if np.random.uniform(0,1) <= self.eps_prob:
        action = np.random.randint(0,self.num_actions)
      else: action = np.argmax(self.d[(state[0],state[1], time_to_term)])
      return action
