#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

def gp(XX,YY,XX_ran,d):
    #ker= RBF()
    #Fit an GP model
    ker = ConstantKernel(1, (0.01, 100)) * RBF(length_scale=0.5 * np.ones(d,), length_scale_bounds=(0.05, 2.0)) +WhiteKernel(1e-4,(1e-6, 1e-2))
    
    model = GaussianProcessRegressor(kernel=ker,random_state=0,n_restarts_optimizer=5)
    model.fit(XX.to_numpy(),YY.to_numpy())
    
    y_hat = model.predict(XX_ran.to_numpy())

    return y_hat


# def gp(XX,YY,XX_ran,d):
#     ker= RBF()
#     #Fit an GP model
    
#     model = GaussianProcessRegressor(kernel=ker,random_state=0)
#     model.fit(XX.to_numpy(),YY.to_numpy())
    
#     y_hat = model.predict(XX_ran.to_numpy())

#     return y_hat