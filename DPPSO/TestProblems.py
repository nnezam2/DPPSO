#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
import math
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from BBOB_problems import *
#from Robotpush import PushReward

def rosenbrock(x):
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return total
    
def rastrigin(x):
    d=len(x)
    total=10 * d + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    return total

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
    
    return y

def Michalewicz(x):
    d=len(x)
    y=-np.sum(np.sin(x) * (np.sin(((1 + np.arange(d)) * x ** 2) / np.pi)) ** 20)
    return y

def Zakharov(x):
    d=len(x)
    y=np.sum(x ** 2) + np.sum(0.5 * (1 + np.arange(d)) * x) ** 2 
    total=y + np.sum(0.5 * (1 + np.arange(d)) * x) ** 4
    return total 

def Perm(x):
    d=len(x)
    beta = 10.0
    outer = 0.0
    for ii in range(d):
        inner = 0.0
        for jj in range(d):
            xj = x[jj]
            inner += ((jj + 1) + beta) * (xj ** (ii + 1) - (1.0 / (jj + 1)) ** (ii + 1))
        outer += inner ** 2
    return outer        

def Ackley(x):
    d=len(x)
    term1=np.exp(-0.2 * np.sqrt(sum(x ** 2) / d))
    term2=np.exp(sum(np.cos(2.0 * np.pi * x)) / d)
    total=-20.0 * term1-term2+ 20+ np.exp(1)
    #print(total)
    return total

def branin(x):
    x1=x[0]
    x2=x[1]
    #print(x)
    #print(x1,x2)
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    result = term1 + term2 + s
    
    return result

# def goldstein_price(x):
#     x1 = x[0]
#     x2 = x[1]
    
#     term1 = 1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
#     term2 = 30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    
#     return term1 * term2 

def goldstein_price(xx):
    x1bar = 4*xx[0] - 2
    x2bar = 4*xx[1] - 2

    fact1a = (x1bar + x2bar + 1)**2
    fact1b = 19 - 14*x1bar + 3*x1bar**2 - 14*x2bar + 6*x1bar*x2bar + 3*x2bar**2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2*x1bar - 3*x2bar)**2
    fact2b = 18 - 32*x1bar + 12*x1bar**2 + 48*x2bar - 36*x1bar*x2bar + 27*x2bar**2
    fact2 = 30 + fact2a * fact2b

    prod = fact1 * fact2

    y = (np.log(prod) - 8.693) / 2.427
    
    return y

def six_hump_camel(x):
    x1=x[0]
    x2=x[1]
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * (x1 ** 2)
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * (x2 ** 2)
    return term1 + term2 + term3


def hartmann_6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])

    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

    xxmat = np.tile(x, (4, 1))
    inner = np.sum(A[:, :6] * (xxmat - P[:, :6]) ** 2, axis=1)
    outer = np.sum(alpha * np.exp(-inner))
    
    return -outer

# Example usage:
xx = np.random.rand(6)  # Random 6-dimensional vector
result = hartmann_6(xx)
print("Result:", result)


def evaluate(x,func):
    xx=x.copy()
    n=x.shape[0]
    m=x.shape[1]
    #print('n is:',n)
    #print(x)
    ys=[]
    if (func==1):
        for i in range(n):
            ys.append(rosenbrock(x.iloc[i,:]))
    elif(func==2):
        for i in range(n):
            ys.append(rastrigin(x.iloc[i,:]))    
    elif(func==3):
        for i in range(n):
            ys.append(levy(x.iloc[i,:]))
    elif(func==4):
        for i in range(n):
            ys.append(Michalewicz(x.iloc[i,:]))
    elif(func==5):
        for i in range(n):
            ys.append(Zakharov(x.iloc[i,:]))
    elif(func==6):
        for i in range(n):
            ys.append(Perm(x.iloc[i,:]))
    elif(func==7 or func==77):
        for i in range(n):
            ys.append(Ackley(x.iloc[i,:]))
    elif(func==10):  
        for i in range(n):
            ys.append(sphere1(x.iloc[i,:]))               
    elif(func==11):  
        for i in range(n):
            ys.append(ellipsoid2(x.iloc[i,:]))               
    elif(func==12):  
        for i in range(n):
            ys.append(rastrigin3(x.iloc[i,:]))               
    elif(func==13):  
        for i in range(n):
            ys.append(buecheRastrigin4(x.iloc[i,:]))               
    elif(func==14):  
        for i in range(n):
            ys.append(linearSlope5(x.iloc[i,:]))               
    elif(func==15):  
        for i in range(n):
            ys.append(attractiveSector6(x.iloc[i,:]))               
    elif(func==16):  
        for i in range(n):
            ys.append(stepEllipsoid7(x.iloc[i,:]))               
    elif(func==17):  
        for i in range(n):
            ys.append(rosenbrock8(x.iloc[i,:]))               
    elif(func==18):  
        for i in range(n):
            ys.append(rosenbrockRotated9(x.iloc[i,:]))               
    elif(func==19):  
        for i in range(n):
            ys.append(ellipsoidRotated10(x.iloc[i,:]))               
    elif(func==20):  
        for i in range(n):
            ys.append(discus11(x.iloc[i,:]))               
    elif(func==21):  
        for i in range(n):
            ys.append(bentCigar12(x.iloc[i,:]))               
    elif(func==22):  
        for i in range(n):
            ys.append(sharpRidge13(x.iloc[i,:]))               
    elif(func==23):  
        for i in range(n):
            ys.append(differentPowers14(x.iloc[i,:]))               
    elif(func==24):  
        for i in range(n):
            ys.append(rastriginRotated15(x.iloc[i,:]))               
    elif(func==25):  
        for i in range(n):
            ys.append(weierstrass16(x.iloc[i,:]))               
    elif(func==26):  
        for i in range(n):
            ys.append(schaffers17(x.iloc[i,:]))               
    elif(func==27):  
        for i in range(n):
            ys.append(schaffers18(x.iloc[i,:]))               
    elif(func==28):  
        for i in range(n):
            ys.append(griewankRosenBrock19(x.iloc[i,:]))               
    elif(func==29):  
        for i in range(n):
            ys.append(schwefel20(x.iloc[i,:]))               
    elif(func==30):  
        for i in range(n):
            ys.append(gallagher21(x.iloc[i,:]))               
    elif(func==31):  
        for i in range(n):
            ys.append(gallagher22(x.iloc[i,:]))               
    elif(func==32):  
        for i in range(n):
            ys.append(katsuura23(x.iloc[i,:]))               
    elif(func==33):  
        for i in range(n):
            ys.append(lunacekBiRastrigin24(x.iloc[i,:]))                            
    elif(func==34):  
        for i in range(n):
            f=PushReward()
            #print(x.iloc[i,:])
            #print(f(x.iloc[i,:]))
            ys.append(f(x.iloc[i,:]))  
    elif(func==35):
        for i in range(n):
            #print(branin(x.iloc[i,:]))
            ys.append(branin(x.iloc[i,:]))
    elif(func==36):
        for i in range(n):
            ys.append(hartmann_6(x.iloc[i,:]))            
    elif(func==37):
        for i in range(n):
            ys.append(goldstein_price(x.iloc[i,:]))  
            
    yy=pd.DataFrame(ys)
    #print(yy,yy.shape)
    xx[m]=yy.copy()
    #xx[m+1]=yy.copy()
    return xx

#Sphere1,Ellipsoid2,Rastrigin3,BuecheRastrigin4, LinearSlope5
#AttractiveSector6, StepEllipsoid7, Rosenbrock8, RosenbrockRotated9
#EllipsoidRotated10, Discus11, BentCigar12, SharpRidge13, DifferentPowers14
#RastriginRotated15, Weierstrass16, Schaffers17, Schaffers18, GriewankRosenBrock19 
#Schwefel20, Gallagher21, Gallagher22, Katsuura23, LunacekBiRastrigin24