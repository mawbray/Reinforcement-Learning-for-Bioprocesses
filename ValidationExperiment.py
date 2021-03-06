import numpy as np
import pandas as pd
import scipy.integrate as scp
import matplotlib.pyplot as plt
import numpy.random as rnd
eps  = np.finfo(float).eps

#################### --- Validation Simulation --- #####################
class Experiment_Done(object):
    def __init__(self, env, agent, controls, episodes, eps):
      self.env , self.agent,         = env, agent 
      self.controls, self.episodes   = controls, episodes
      self.eps                       =  eps
       
    def simulation(self):
      # Simulation takes environment and simulates, next iteration and outputs reward
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
      plot_m = np.array([episodes-1])

      for ei in range(episodes):
        # initialize simulation
        current_state = x0
        xt[0,:,ei]  = current_state
        tt[0]    = 0.

        # define e greedy policy exploration
        eps_prob = self.eps        #act greedily
      
        # simulation
        for s in range(movements):
            action_indx = self.agent.act(current_state, eps_prob, s)        # select control for this step from that possible
            ctrl[s,ei]   =  ctrls[action_indx]                              # find control action relevant to index from agent.act
            c_hist[s,ei] = action_indx                                      # storing control history for each epoch
            ode       = scp.ode(self.env.model)                             # define ode
            ode.set_integrator('lsoda', nsteps=3000)                        # define integrator
            ode.set_initial_value(current_state,dt)                         # set initial value
            ode.set_f_params(ctrl[s,ei])                                    # set control action
            current_state = list(ode.integrate(ode.t + dt))                 # integrate system
            current_state = discrete_env(np.array(current_state))
            xt[s+1,:,ei]     = current_state                                # add current state Note: here we can add randomnes as: + RandomNormal noise
            tt[s+1]       = (s+1)*dt
        
        
        reward[ei] = self.env.reward(xt[:,:,ei])
        
      return reward  
      
