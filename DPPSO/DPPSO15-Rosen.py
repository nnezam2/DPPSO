#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import csv
import math
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from DPP_Sampling import DPP_pure
from DPP_Sampling import DPP_scored
from DPP_Sampling import DPP_scoredcond
from DPP_Sampling import DPP_scoredcond2
from DPP_Sampling import DPP_scoredcond3
from sklearn.metrics import r2_score
from TestProblems import *
import numpy as np 
import pandas as pd 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import warnings
import os
import time 
warnings.filterwarnings('ignore') 
wd=os.getcwd()

#rosenbrock,rastrigin,levy,Michalewicz,Zakharov,Perm,Ackley #1:7 
#Sphere1,Ellipsoid2,Rastrigin3,BuecheRastrigin4, LinearSlope5 #10: 
#AttractiveSector6, StepEllipsoid7, Rosenbrock8, RosenbrockRotated9 #15: 
#EllipsoidRotated10, Discus11, BentCigar12, SharpRidge13, DifferentPowers14 #19: 
#RastriginRotated15, Weierstrass16, Schaffers17, Schaffers18, GriewankRosenBrock19 #24: 
#Schwefel20, Gallagher21, Gallagher22, Katsuura23, LunacekBiRastrigin24 #29:

# 1 and 3 

def gp(XX,YY,d):
    #ker= RBF()
    #Fit an GP model
    ker = ConstantKernel(1, (0.01, 100)) * RBF(length_scale=0.5 * np.ones(d,), length_scale_bounds=(0.05, 2.0)) +WhiteKernel(1e-4,(1e-6, 1e-2))
    
    model = GaussianProcessRegressor(kernel=ker,random_state=0,n_restarts_optimizer=5)
    model.fit(XX,YY)
    
    #y_hat = model.predict(XX_ran.to_numpy())

    return model

    
class My_BO:
    def __init__(self,B,d,func,Strategy,Init_X,Init_Y,Cand_size,batch_size,lb,up):
        self.Budget=B
        self.dimension=d
        self.Strategy=Strategy
        self.func=func 
        self.x=Init_X
        self.y=Init_Y
        self.Cand_size=Cand_size
        self.batch_size=batch_size
        self.t=0
        self.lb=lb
        self.up=up
        self.impr=0
        self.newsol=0
        self.r=0.2
        self.cimp=0
        self.tre=0.05
        self.alpha=1
        self.beta=1
        self.times=[0]* (self.x.shape[0]-1)
        self.yvals=[self.y.min()]* self.x.shape[0]
        #self.det=[]
        #self.hvol=[]
        
    def optimize(self):
        import time 
        start=time.time()
        self.t=self.t+1
        print('Iteration Number ', str(self.t))
        print('Remained Budget',self.Budget)
        print('Labeled Set Size',self.x.shape)

        if(self.Strategy=='DPPSO1'): #correct one (LCB) 
            
            best_f = self.y.min()
            #self.yvals.append(best_f)
            print('bestf', best_f)
            print('log(bestf)',np.log(best_f))
            
            XX=pd.DataFrame(self.x)
            YY=pd.DataFrame(self.y)
            #print('ys:',YY)
            
            ## Pool Generation 
            num1=min(XX.shape[0],1*self.batch_size)
            nk=[]
            ids=[]
            EEPAdist=1
            num2=num1
            
            print('Current impr:',self.impr)
            if(self.impr<=1):
                self.impr=0
            else:    
                self.impr=2
            self.alpha += self.impr
            self.beta += 2-self.impr
            rnd_sample=np.random.beta(self.alpha, self.beta)
            self.r = 0.1 + rnd_sample * 0.9 #based on radius range
            
            print('Current radius r value is:',self.r)
            
            df=XX.copy()
            df2=YY.copy()   
            #print('Numbers:',num1,num2)
            new_ids,nn1=DPP_scoredcond3(self.x,df,df2,func,d,EEPAdist,ids,nk,num2,self.t)
            nn1=nn1.drop_duplicates()
            cp=nn1.reset_index(drop=True).iloc[:,0:self.dimension]
            #print(cp)
            cp['y']=evaluate(cp,self.func).iloc[:,self.dimension]
            
            centers=cp.sort_values(by=['y']).head(num2).iloc[:,0:self.dimension].reset_index(drop=True)    
            c_pt=[]
            cc=[]
            pp=[]
            num_cand=int((self.Cand_size)/centers.shape[0])
            total=int(num_cand*centers.shape[0])
            for i in range(centers.shape[0]):
                points=[]
                for j in range(self.dimension):
                    c=centers.iloc[i,j]
                    v=np.random.normal(c,self.r,num_cand)
                    points.append(v.T)           
                cc.append(np.array(points).T)

            pool2=np.array(cc)
            pool2=pool2.reshape(total,self.dimension)
            pool2=pd.DataFrame(pool2)
            print('final pool size',pool2.shape[0])
            sp=pool2 #dynamic_pool(centers,num_cen,d,func,num_can,iteration)
            sp=sp.drop_duplicates().reset_index(drop=True)
            XX_ran=sp.copy()
            #print('Pool',XX_ran)
            
            ## Modeling 
            model=gp(self.x,self.y,self.dimension)
            mean_j,var_j=model.predict(np.array(XX_ran),return_std=True)
            std = np.sqrt(var_j)
            Yfit_random=pd.DataFrame(mean_j)   
            new=evaluate(sp,self.func)
            new_ids,nn1=DPP_scoredcond2(XX,XX_ran,Yfit_random,func,d,EEPAdist,ids,new,self.batch_size,self.t)
            nn1=nn1.reset_index(drop=True)
            
            nn2=np.around(np.array(nn1.iloc[:,0:self.dimension]),decimals=3)
            self.x=np.append(self.x,nn2,axis=0)
            newy=evaluate(nn1.iloc[:,0:self.dimension],self.func).iloc[:,self.dimension].to_numpy()
            print('newy',newy)
            self.y=np.append(self.y,newy) #change 
            self.Budget=self.Budget-self.batch_size
            sol=self.y.min()
            self.impr=self.newsol-sol       
            self.newsol=sol 
            
            if(self.impr>self.tre): self.cimp=0
            else: self.cimp=self.cimp+1
                
            ##time 
            T=time.time()
            tt=T-self.start  
            self.times.extend([tt]*batch_size)
            ## best values
            best_f = self.y.min()
            self.yvals.extend([best_f]*batch_size)            
#         self.hvol=np.append(self.hvol,0)
#         self.det=np.append(self.det,0) 
        print('*****************')
             
    def run(self):
        self.start=time.time()
        self.times.append(self.start)
        while(self.Budget>0):
            self.optimize()  
            
        end=time.time()
        duration=end-self.start
        print('Timing of run is ',duration,' seconds')  
    
        return self.x,self.y,self.yvals,self.times 


#Input:
N_rep=10 
Budget=600
d=15
Cand_size=1500
batch_size=20
Name='rosenbrock_15_20' #problem_dim
func=1 
initial_size=2*(d+1)
lb=-2.048
up=2.048

strategies=np.array(['DPPSO1'])

for ss in strategies:
    
    Strategy=ss

    for i in np.arange(1,N_rep+1):

        B=Budget-initial_size
        #path='./pool_'+str(d)+'d'+'/pool_'+str(i)
        wd=os.getcwd()
        poolname='pool_'+str(i)
        path = wd+'/Rosen-15d/'+str(poolname)
        
        folder_name = 'result_'+Strategy+'_'+Name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            print(f"The folder '{folder_name}' already exists.")

        Init_X=pd.read_csv(path, index_col=0, header=0).round(3).iloc[:,0:d]#.to_numpy()
        Init_Y = evaluate(Init_X,func).iloc[:,d]#.reshape(-1,1)
        
        bo=My_BO(B,d,func,Strategy,Init_X.to_numpy(),Init_Y.to_numpy(),Cand_size,batch_size,lb,up)
        x,y,yvals,times=bo.run()
        
        print('Best solution found:',y.min())
        ans=pd.DataFrame(x)
        ans[d]=pd.DataFrame(y)
        ans[d+1]=yvals
        ans[d+2]=times 
        ans.to_csv('./'+folder_name+'/'+Strategy+'_'+Name+'_'+str(i)+'.csv')

#Input:
N_rep=10 
Budget=600
d=15
Cand_size=1500
batch_size=50
Name='rosenbrock_15_50' #problem_dim
func=1 
initial_size=2*(d+1)
lb=-2.048
up=2.048

strategies=np.array(['DPPSO1'])

for ss in strategies:
    
    Strategy=ss

    for i in np.arange(1,N_rep+1):

        B=Budget-initial_size
        #path='./pool_'+str(d)+'d'+'/pool_'+str(i)
        wd=os.getcwd()
        poolname='pool_'+str(i)
        path = wd+'/Rosen-15d/'+str(poolname)
        
        folder_name = 'result_'+Strategy+'_'+Name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            print(f"The folder '{folder_name}' already exists.")

        Init_X=pd.read_csv(path, index_col=0, header=0).round(3).iloc[:,0:d]#.to_numpy()
        Init_Y = evaluate(Init_X,func).iloc[:,d]#.reshape(-1,1)
        
        bo=My_BO(B,d,func,Strategy,Init_X.to_numpy(),Init_Y.to_numpy(),Cand_size,batch_size,lb,up)
        x,y,yvals,times=bo.run()
        
        print('Best solution found:',y.min())
        ans=pd.DataFrame(x)
        ans[d]=pd.DataFrame(y)
        ans[d+1]=yvals
        ans[d+2]=times 
        ans.to_csv('./'+folder_name+'/'+Strategy+'_'+Name+'_'+str(i)+'.csv')
        



# In[ ]:





# In[ ]:




