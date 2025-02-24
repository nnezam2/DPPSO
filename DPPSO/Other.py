#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
import math
    
def R_rule(imp,r):
    
    if(imp>0.01):
        r=r*2
    else:
        r=r/2

    if(r>4):
        r=4
    elif(r<0.1 and imp>0.01):
        r=0.1
    elif(r<0.1 and imp<0.01):
        r=1
        
    return r 

def func_reader(func,d,wd,poolname):
    if(func==1 and d==30):
        print('Function is Rosenbrock')
        lb=-2.048
        up=2.048
        path1 = wd+'/pool_fracmodel30/'+str(poolname) # 
        
    elif(func==2 and d==30):
        print('Function is Rastrigin')
        lb=-5.12
        up=5.12
        path1 = wd+'/pool_fracmodel31/'+str(poolname) #
        
    elif(func==3 and d==30):
        print('Function is Levy')
        lb=-5
        up=5
        path1 = wd+'/pool_fracmodel32/'+str(poolname) #
        
    elif(func==3 and d==6):
        print('Function is Levy')
        lb=-5
        up=5
        path1 = wd+'/Levy_6d/'+str(poolname) #
        
    elif(func==2 and d==5):
        print('Function is Rastrigin')
        lb=-5
        up=5
        path1 = wd+'/Rastrigin_5d/'+str(poolname) #
    elif(func==2 and d==10):
        print('Function is Rastrigin')
        lb=-5
        up=5
        path1 = wd+'/Rast_10d/'+str(poolname) #
        
    elif(func==2 and d==15):
        print('Function is Rastrigin')
        lb=-5
        up=5
        path1 = wd+'/Rastrigin_15d/'+str(poolname) #
        
    elif(func==2 and d==50):
        print('Function is Rastrigin')
        lb=-5
        up=5
        path1 = wd+'/Rastrigin_50d/'+str(poolname) #
        
    elif(func==4 and d==30):
        print('Function is Michalewicz')
        lb= 0 
        up= math.pi 
        path1= wd+'/Michalewicz_30d/'+str(poolname)
    elif(func==4 and d==5):
        print('Function is Michalewicz')
        lb= 0 
        up= math.pi 
        path1= wd+'/Michalewicz_5d/'+str(poolname)
    elif(func==4 and d==10):
        print('Function is Michalewicz')
        lb= 0 
        up= math.pi 
        path1= wd+'/Michalewicz_10d/'+str(poolname)
    elif(func==4 and d==15):
        print('Function is Michalewicz')
        lb= 0 
        up= math.pi 
        path1= wd+'/Michalewicz_15d/'+str(poolname)
    elif(func==4 and d==50):
        print('Function is Michalewicz')
        lb= 0 
        up= math.pi 
        path1= wd+'/Michalewicz_50d/'+str(poolname)  
        
    elif(func==5 and d==30):
        print('Function is Zakharov')
        lb= -5 
        up= 10 
        path1= wd+'/Zakharov_30d/'+str(poolname)
    elif(func==6 and d==30):
        print('Function is Perm')
        lb= -2 
        up=  2
        path1= wd+'/Perm_30d/'+str(poolname)
        
    elif(func==7 and d==5):
        print('Function is Ackley')
        lb= -15 
        up=  20
        path1= wd+'/Ackley_5d/'+str(poolname)
        
    elif(func==77 and d==15):
        print('Function is Ackley')
        lb= -15 
        up=  20
        path1= wd+'/Ackley_15d/'+str(poolname)
        
    elif(func==8):
        print('Function is Weierstrass')
        lb= -5 
        up= 5
        path1= wd+'/Weierstrass_30d/'+str(poolname)   
    
    elif(func==9):
        print('Function is Schwefel')
        lb= -5 
        up= 5
        path1= wd+'/Sch_30d/'+str(poolname) 
        
    elif(func>=10 and d==30):
        print('Function is BBOB problem ',str(func-9))
        lb= -5 
        up= 5
        path1= wd+'/SharpR_30d/'+str(poolname) 
        
    elif(func==80 and d==60):
        print('Function is Rover 60d test problem')
        lb=0
        up=1
        path1= wd+'/Rover_60d/'+str(poolname)
    print('************************************\n')   
    
    return path1,lb,up

