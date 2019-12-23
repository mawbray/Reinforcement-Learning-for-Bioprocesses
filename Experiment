# Import 
import numpy as np
import scipy.integrate as scp

import numpy.random as rnd
eps  = np.finfo(float).eps


################# --- Training Agent --- #####################
class Experiment(object):
    def __init__(self, env, agent, controls, episodes,xi):
      self.env , self.agent,                     = env, agent 
      self.controls, self.episodes, self.xi      = controls, episodes, xi

    def eps_prob(self,ei,episodes):
        if self.xi == int(1):
            F = 0.1
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(2):
            F = 0.2
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(3):
            F = 0.3
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(4):
            F = 0.4
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(5):
            F = 0.5
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(6):
            F = 0.05
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        elif self.xi == int(7):
            F = 0.01
            G = -np.log(0.1)*F*episodes     # =no of episodes until behave =0.1
            if ei < G:
                behave = np.exp(-ei/(episodes*F))
            else:
                behave = 0.1
        else: behave = 1                    # behave randomly all the time
        return behave      
            
       
    def simulation(self):
      # Simulation takes environment, imparts control action from e-greedy policy and simulates, observes next state to the end of the sequence and outputs reward
      # internal definitions
      discrete_env = self.env.discrete_env
      dt, movements, x0   = self.env.dt, int(self.env.tf/float(self.env.dt)), self.env.x0
      model, ctrls = self.env.model, self.controls       #takes set of control options
      episodes = self.episodes

      # compile state and control trajectories
      xt = np.zeros((movements+1, x0.shape[0], episodes))
      tt = np.zeros((movements+1))
      c_hist = np.zeros((movements, episodes))
      ctrl = np.zeros((movements, episodes))
      reward = np.zeros((episodes))

      for ei in range(episodes):
        # initialize simulation
        current_state = x0
        xt[0,:,ei]  = current_state
        tt[0]    = 0.
        
        
        # define e greedy policy exploration
        eps_prob = self.eps_prob(ei,episodes)
      
        # simulation
        for s in range(movements):
            action_indx  = self.agent.act(current_state, eps_prob, s)        # select control for this step from that possible
            ctrl[s,ei]   =  ctrls[action_indx]                               # find control action relevant to index from agent.act
            c_hist[s,ei] = action_indx                                       # storing control history for each epoch
            ode          = scp.ode(self.env.model)                           # define ode
            ode.set_integrator('lsoda', nsteps=3000)                         # define integrator
            ode.set_initial_value(current_state,dt)                          # set initial value
            ode.set_f_params(ctrl[s,ei])                                     # set control action
            current_state = list(ode.integrate(ode.t + dt))                  # integrate system
            current_state = discrete_env(np.array(current_state))
            xt[s+1,:,ei]  = current_state                                    # add current state Note: here we can add randomnes as: + RandomNormal noise
            tt[s+1]       = (s+1)*dt
        
        for i in [0, 0.2, 0.4, 0.6, 0.8]:
            if i == ei/episodes:
                print('Simulation is', i*100 , ' percent complete')

        reward[ei] = self.env.reward(xt[:,:,ei])
        self.agent.Learn(xt[:,:,ei], c_hist[:,ei], reward[ei])
            
      d = self.agent.learned()
      return reward, d 
  
