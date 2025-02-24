#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
import os 
import pandas as pd 
import numpy as np
import time 
import pickle 

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import gpytorch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.sampling import SobolQMCNormalSampler, SobolEngine #.samplers

## import acquistion functions 

from botorch.acquisition import qKnowledgeGradient,qExpectedImprovement,qProbabilityOfImprovement
from botorch.acquisition import qUpperConfidenceBound,qMaxValueEntropy, qLowerBoundMaxValueEntropy
from botorch.optim import optimize_acqf_discrete


# Define the GP Regression Model
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, d):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        # Mean function
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Base RBF Kernel with ARD
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        self.base_kernel.lengthscale = 0.5 * torch.ones(d)
        lengthscale_constraint = gpytorch.constraints.Interval(0.05, 2.0)
        self.base_kernel.register_constraint("raw_lengthscale", lengthscale_constraint)
        
        # Scale Kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.covar_module.outputscale = 1.0
        outputscale_constraint = gpytorch.constraints.Interval(0.01, 100.0)
        self.covar_module.register_constraint("raw_outputscale", outputscale_constraint)
        
        # Likelihood noise
        self.likelihood.noise = torch.tensor(1e-4)
        noise_constraint = gpytorch.constraints.Interval(1e-6, 1e-2)
        self.likelihood.register_constraint("raw_noise", noise_constraint)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
def levy(x):
    import math
    d=len(x)
    w= 1 + (x - 1)/4
    pi=math.pi
    sin=np.sin
    p2=np.exp2
    term1=(sin(pi*w[0]))**2
    a1=(w[d-1]-1)**2
    a2=1+(sin(2*pi*w[d-1]))**2
    term3=a1*a2
    wi=w[0:(d-1)]
    b1=(wi-1)**2
    b2=1+10*(sin((pi*wi)+1)**2)
    summ=sum(b1*b2)
    y=term1+summ +term3
    
    return -y

def bayopt(acq='EI',func='Levy',budget=50,pool_size=1000,batch_size=4,initial_size=14,d=6):


    wd=os.getcwd()
    
    idxpool=np.arange(1,11) #31
    
    if(func=='Levy'):
        lb=-5
        up=5
        b=torch.ones([2,d])
        b[0]=b[0]*lb
        b[1]=b[1]*up
        bounds=b
        #print('bounds:',bounds)
        
    print('*** acquisition function is: ',acq,'***')    
    for i in idxpool:
        start=time.time()
        B=budget-initial_size
        poolname='pool_'+str(i) #+'.csv'
        print('****Initialization: ',poolname,' ****')
        path1 = wd+'/Levy-6d/'+str(poolname) # 

        train_X=pd.read_csv(path1, index_col=0, header=0).round(3).iloc[:,0:d].to_numpy()
        train_Y = levy(train_X.T).reshape(-1,1)
        train_X=torch.from_numpy(train_X)
        train_Y=torch.from_numpy(train_Y)
        
        bestf = train_Y.max()
        yvals=[-bestf]* initial_size
        times=[0] * initial_size
        t=0
        while (0<B):
            t=t+1
            print('Remained Budget',B)
            print('Labeled Set Size',train_X.shape)
            best_f = train_Y.max()
            # Initialize the model
            model = SingleTaskGP(train_X, train_Y)

            # Set the mean module to ConstantMean (optional)
            model.mean_module = gpytorch.means.ConstantMean()

            # Access and modify the base RBF Kernel with ARD
            model.covar_module.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=d)

            # Set initial lengthscale
            model.covar_module.base_kernel.lengthscale = 0.5 * torch.ones(d)
            lengthscale_constraint = gpytorch.constraints.Interval(0.05, 2.0)
            model.covar_module.base_kernel.register_constraint("raw_lengthscale", lengthscale_constraint)

            # Set initial outputscale
            model.covar_module.outputscale = 1.0
            outputscale_constraint = gpytorch.constraints.Interval(0.01, 100.0)
            model.covar_module.register_constraint("raw_outputscale", outputscale_constraint)

            # Set the likelihood noise
            model.likelihood.noise_covar.noise = torch.tensor(1e-4)

            # Define the noise constraint
            noise_constraint = gpytorch.constraints.Interval(1e-6, 1e-2)

            # Register the constraint on the raw_noise parameter within noise_covar
            model.likelihood.noise_covar.register_constraint("raw_noise", noise_constraint)


            # Define the Marginal Log Likelihood
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Fit the model
            with gpytorch.settings.cholesky_jitter(1e-3):
                fit_gpytorch_model(mll)
                
                
            # Define the Sobol sampler for the acquisition function
            sampler = SobolQMCNormalSampler(1024)

            # Generate candidate set using Sobol sequences
            sobol_engine = SobolEngine(dimension=bounds.size(1), scramble=True)
            candidate_set = sobol_engine.draw(1000).to(bounds.device, bounds.dtype)
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

            # Define the acquisition function
            
            if(acq=='EI'):
                aq = qExpectedImprovement(model,best_f,sampler) 
            elif(acq=='PI'):
                aq = qProbabilityOfImprovement(model,best_f,sampler)
            elif(acq=='UCB'):
                delta = 0.1
                beta = 2 * np.log((t ** 2 * np.pi ** 2) / (6 * delta))
                aq = qUpperConfidenceBound(model,beta,sampler)
            elif(acq=='Gibbon'):
                aq = qLowerBoundMaxValueEntropy(model, candidate_set)   
            elif(acq=='KG'): #discrete optimization is not supported here 
                #aq= qKnowledgeGradient(model)
                print('Not supported')
            else:
                print('No Valid Acquisition Function Selected')
                break
    
            # Optimize the acquisition function over the candidate set
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=aq,
                q=batch_size,
                choices=candidate_set,
                unique=True,
            )

            
            print('New candidate values',levy(candidates.T))
            ##update the labeled pool
            train_X=torch.cat((train_X,candidates),0)
            train_Y=torch.cat((train_Y,levy(candidates.T).reshape(-1,1)),0)
            #print('Minimum Found So Far',train_Y.max())
            
            best_f = train_Y.max()
            print('bestf',-best_f)
            yvals.extend([-best_f]*batch_size)
            B=B-batch_size
            T=time.time()
            tt=T-start 
            times.extend([tt]*batch_size)
            print('**********')
        end=time.time()
        duration=end-start
        print('Timing of run is ',duration,' seconds')
        train_Y = train_Y.numpy()
        train_X =train_X.numpy()
        yvals = np.array(yvals).reshape(train_X.shape[0], 1)
        times = np.array(times).reshape(train_X.shape[0], 1)        
#         print(yvals.shape,times.shape)
#         print(train_Y.shape)
#         print(train_X.shape)     
        combined = np.concatenate([train_X, train_Y, yvals, times], axis=1)
        df= pd.DataFrame(combined)
        directory = 'result'+'_'+acq+'_'+func+'_'+str(d)+'_'+str(batch_size)
        os.makedirs(directory, exist_ok=True)
        # Convert the NumPy array to a Pandas DataFrame
        df.to_csv(f'{directory}/res_{i}.csv', index=False)                     
        print('\n ***********************\n ')

bayopt(acq='EI',func='Levy',budget=400,pool_size=1000,batch_size=4,initial_size=14,d=6)
bayopt(acq='PI',func='Levy',budget=400,pool_size=1000,batch_size=4,initial_size=14,d=6)
bayopt(acq='UCB',func='Levy',budget=400,pool_size=1000,batch_size=4,initial_size=14,d=6)
bayopt(acq='Gibbon',func='Levy',budget=400,pool_size=1000,batch_size=4,initial_size=14,d=6)

# bayopt(acq='EI',func='Levy',budget=400,pool_size=1000,batch_size=20,initial_size=14,d=6)
# bayopt(acq='PI',func='Levy',budget=400,pool_size=1000,batch_size=20,initial_size=14,d=6)
# bayopt(acq='UCB',func='Levy',budget=400,pool_size=1000,batch_size=20,initial_size=14,d=6)
# bayopt(acq='Gibbon',func='Levy',budget=400,pool_size=1000,batch_size=20,initial_size=14,d=6)

# bayopt(acq='EI',func='Levy',budget=400,pool_size=1000,batch_size=50,initial_size=14,d=6)
# bayopt(acq='PI',func='Levy',budget=400,pool_size=1000,batch_size=50,initial_size=14,d=6)
# bayopt(acq='UCB',func='Levy',budget=400,pool_size=1000,batch_size=50,initial_size=14,d=6)
#bayopt(acq='Gibbon',func='Levy',budget=400,pool_size=1000,batch_size=50,initial_size=14,d=6)



# In[ ]:




