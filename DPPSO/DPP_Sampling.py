#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import smp 

def DPP_pure(XX,XX_ran,Yfit_random,func,d,EEPAdist,ids,k):
    
    from smp import sample_ids_mc    
    scores = np.ones(XX_ran.shape[0])
    #print('Inside',np.array(XX_ran))
    new_id=sample_ids_mc(np.array(XX_ran),np.ones(XX_ran.shape[0]),k,alpha=1, gamma=1, cond_ids=ids, steps=1000)
    XX_ran['index']=XX_ran.index
    selected_points=pd.DataFrame(XX_ran.iloc[new_id])
    
    return new_id,selected_points 

def DPP_scored(XX,space,ys,func,d,EEPAdist,ids,new,k,ite):
    
    XX_ran=space
    Yfit_random=ys
    from smp import sample_ids_mc

    if(ite<10):
        e=1
    elif(10<=ite<20):
        e=0.9
    elif(20<=ite<30):
        e=0.8
    elif(40<=ite<50):
        e=0.75
    elif(50<=ite<60):
        e=0.6
    elif(50<=ite<60):
        e=0.5
    elif(60<=ite<70):
        e=0.4          
    elif(70<=ite<80):
        e=0.25
    elif(80<=ite<90):
        e=0.10
    else:
        e=0.01 
        
    print('\nepsilon is:',e)
    scores1=np.array(Yfit_random)
    mini=scores1.min()
    scores=1/(scores1-mini+e)
   
    new_id=sample_ids_mc(np.array(XX_ran),scores,k,alpha=1, gamma=2**(ite/10), cond_ids=ids, steps=1000)

    XX_ran['index']=XX_ran.index
    selected_points=pd.DataFrame(XX_ran.iloc[new_id])
    
    print('\nYhat Range',np.array(Yfit_random).min(),np.array(Yfit_random).max())
    print('Score Range',scores.min(),scores.max())
    
    print('\nyhat of the selected points',Yfit_random[new_id])
    #print('Scores of the selected points',scores[new_id])
    return new_id,selected_points 

def DPP_scoredcond(XX,XX_ran,Yfit_random,func,d,EEPAdist,ids,new,k,ite):
    from smp import sample_ids_mc
    
    scores=np.ones(XX_ran.shape[0])
    new_id=sample_ids_mc(np.array(XX_ran),scores,k,alpha=1, gamma=2, cond_ids=ids, steps=1000)

    XX_ran['index']=XX_ran.index
    selected_points=pd.DataFrame(XX_ran.iloc[new_id])
    
    return new_id,selected_points 


def DPP_scoredcond2(XX,XX_ran,Yfit_random,func,d,EEPAdist,ids,new,k,ite):
    ids=[]
    from smp import sample_ids_mc
    
    if(ite<10):
        e=0.9
    elif(10<=ite<20):
        e=0.8
    elif(20<=ite<30):
        e=0.7
    elif(30<=ite<40):
        e=0.6
    elif(40<=ite<50):
        e=0.5
    elif(50<=ite<60):
        e=0.4
    elif(60<=ite<70):
        e=0.3         
    elif(70<=ite<80):
        e=0.2
    elif(80<=ite<90):
        e=0.1
    else:
        e=0.01
      
    print('\nepsilon is:',e)
    scores1=np.array(Yfit_random)
    mini=scores1.min()
    scores=1/(scores1-mini+e)
    new_id=sample_ids_mc(np.array(XX_ran),scores,k,alpha=1, gamma=10, cond_ids=ids, steps=1000)
    XX_ran['index']=XX_ran.index
    selected_points=pd.DataFrame(XX_ran.iloc[new_id])
    print('Score Range',scores.min(),scores.max())
    #print('Scores of the selected points',scores[new_id])
           
    return new_id,selected_points 



def DPP_scoredcond3(XX,XX_ran,Yfit_random,func,d,EEPAdist,ids,new,k,ite):
    ids=[]
    from smp import sample_ids_mc
    
    if(ite<10):
        e=0.01
    elif(10<=ite<20):
        e=0.1
    elif(20<=ite<30):
        e=0.2
    elif(30<=ite<40):
        e=0.3
    elif(40<=ite<50):
        e=0.4
    elif(50<=ite<60):
        e=0.5
    elif(60<=ite<70):
        e=0.6         
    elif(70<=ite<80):
        e=0.7
    elif(80<=ite<90):
        e=0.8
    else:
        e=0.9
      
    print('\nepsilon is:',e)
    scores1=np.array(Yfit_random)
    #print('Inside score:',scores1)
    mini=scores1.min()
    scores=1/(scores1-mini+e)
    #print('Inside',np.array(XX_ran))
    #print('Inside',scores)
    new_id=sample_ids_mc(np.array(XX_ran),scores,k,alpha=1, gamma=10, cond_ids=ids, steps=1000)
    XX_ran['index']=XX_ran.index
    selected_points=pd.DataFrame(XX_ran.iloc[new_id])
    print('Score Range',scores.min(),scores.max())
    #print('Scores of the selected points',scores[new_id].min(),scores[new_id].max())
        
    
    return new_id,selected_points 

