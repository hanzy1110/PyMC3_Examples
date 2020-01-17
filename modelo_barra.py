#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:33:53 2020

@author: ubuntu
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

size = 1000

#Beam dimensions:
L, b, h = 5, 0.15, 0.3 #en [m]

#Measured deflection
V_Medido = np.array([12.84, 13.12, 12.13, 12.19, 12.67])*1e-3 
V_Medido = np.concatenate((np.mean(V_Medido)*np.random.randn(size), V_Medido))


#Statistical moments:
mu_E = 30000 #Media del Modulo de Young [MPa]
mu_p = 0.012 #Media de la Carga en [kN/m]

sigma_E = 4500 #Desvio de la Media logarítmica [MPa]
sigma_p = 0.05 * 0.012 #Desvio de la carga [kN/m]

Modelo_Barra = pm.Model()

with Modelo_Barra:
    
    E = pm.Normal('E',mu = mu_E, sigma = sigma_E)        

    #Deflection statistical Moments
    sigma_V = pm.Normal('sigma_V', sigma = 1)
    mu_V = (5/32) * (mu_p * L**4)/(mu_E * b * h**3)

    V_obs = pm.Normal('V_obs', mu = mu_V, sigma = sigma_V, observed = V_Medido)

map_estimate = pm.find_MAP(model = Modelo_Barra)

print(map_estimate)

with Modelo_Barra:

    # instantiate sampler
    step = pm.Slice()

    # draw 5000 posterior samples
    trace = pm.sample(5000, step=step)

pm.traceplot(trace)