# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:34:05 2019
@author: Max Mowbray - University of Manchester, Manchester, United Kingdom
"""

import numpy as np

############ Defining Environment ##############

class Model_env: 
    
    # --- initializing model --- #
    def __init__(self, parameters, steps, tf, x0, modulus):
        
        # Object variable definitions
        self.parameters, self.steps = parameters, steps
        self.x0, self.dt, self.tf      = x0, tf/steps, tf                
        self.modulus                   = modulus          # two column array [biomass nitrate ]
        
    # --- dynamic model definition --- #    
    # model takes state and action of previous time step and integrates -- definition of ODE system at time, t
    def model(self, t, state, control):
        # internal definitions
        params = self.parameters
        FCn   = control
                
        # state vector
        Cx  = state[0]
        Cn  = state[1]
        
        # parameters
        u_m  = params['u_m']; K_N  = params['K_N'];
        u_d  = params['u_d']; Y_nx = params['Y_nx'];
        
        # algebraic equations
        
        # variable rate equations
        dev_Cx  = u_m * Cx * Cn/(Cn+K_N) - u_d*Cx**2
        dev_Cn  = - Y_nx * u_m * Cx * Cn/(Cn+K_N) + FCn
        
        return np.array([dev_Cx, dev_Cn],dtype='float64')
    
    def discrete_env(self, state):
        # discretisation of the system, with introduction of stochasticity in terms of modulus
        modulus = self.modulus
        
            
        resid = state % modulus
        resid = resid/modulus
        UB = 1 - resid
        draw =  np.random.uniform(0,1,2)

        for i in range(state.shape[0]):
            if draw[i] < UB[i]:
              state[i] = state[i] - resid[i] * modulus[i]
            else:
              state[i] = state[i] - resid[i] * modulus[i] + modulus[i]
        
        # fixes for representation 
        # Nitrate fix
        if state[1] < 0:
          state[1] = 0
        elif state[0] < 0:
            state[0] = 0
        
        # Biomass fix
        f = str(self.modulus[0])
        decimal = f[::-1].find('.')  
        state[0] = np.round(state[0], decimal)
        f1 = str(self.modulus[1])
        decimal1 = f1[::-1].find('.')  
        state[0] = np.round(state[0], decimal1)

        if state[0] == eps:
            state[0] = 0
        if state[1] == eps:
            state[1] = 0
        
        return state

    def reward(self, state):
      reward = 100*state[-1][0] - state[-1][1]              # objective function 1
      return reward

        

