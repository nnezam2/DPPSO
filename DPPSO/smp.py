#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import pickle
import os.path
import time
class Kernel(object):
    # Returns the submatrix of the kernel indexed by ps and qs
    def getKernel(self, ps, qs):
        return np.squeeze(ps[:,None]==qs[None,:]).astype(float)
        # Readable version: return np.array([[1. if p==q else 0. for q in qs] for p in ps])
    def __getitem__(self, args):
        ps, qs = args
        return self.getKernel(ps, qs)

class RBFKernel(Kernel):
    def __init__(self, R):
        self.R = R
    def getKernel(self, ps, qs):
        D = cdist(ps,qs)**2
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        return D
    
class ScoredKernel(Kernel):
    def __init__(self, R, space, scores, alpha=2, gamma=1):
        self.R = R
        self.space = space
        if alpha > 0:
            self.scores = scores**(float(gamma)/alpha)
        else:
            self.scores = scores
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        ps = np.array(self.space)[p_ids]
        qs = np.array(self.space)[q_ids]
        D = cdist(ps,qs)**2
        
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        
        # I added the below line to have different sample scores
        D = ((D*self.scores[p_ids]).T * self.scores[q_ids]).T
        #print(D)
        return D
    
class ConditionedScoredKernel(Kernel):
    #print('entering kernel')
    def __init__(self, R, space, scores, cond_ids, alpha=2, gamma=1):
        self.R = R
        self.space = np.array(space)
        #print(self.space)
        #print(self.space.shape)
        self.cond_ids = np.sort(cond_ids)
        #print(self.cond_ids)
        if alpha > 0:
            self.scores = np.array(scores**(float(gamma)/alpha)).reshape(-1)
        else:
            self.scores = scores
        self.kernel = self.computeFullKernel()
        
        
    def computeFullKernel(self):
        import time 
        start_time = time.time()
#         D = cdist(self.space,self.space)+np.eye(self.space.shape[0])#**2   
#         D = np.exp(-D/(2*self.R**2))
#         D = ((D*self.scores).T * self.scores).T

        D=self.space.dot(self.space.T)
        #print('kernel before scoring',D)
        D = ((D*self.scores).T * self.scores).T
        
        #print('Distance part: ',D.shape)
        if len(self.cond_ids)>0:
            eye1 = np.eye(len(self.space))
            for id in self.cond_ids:
                eye1[id,id] = 0
            D = np.linalg.inv(D + eye1) #D + eye1
            #print('after inversion',D.shape)
            mask = np.full(len(self.space), True, dtype=bool)
            mask[self.cond_ids] = False
            D = D[mask,:]
            D = D[:,mask]
            D = np.linalg.inv(D) - np.eye(len(self.space)-len(self.cond_ids))
            #print('after formula',D.shape)
            t=(time.time() - start_time)
            #print('time inversion',t)
            start_time2 = time.time()
            #print('shape before',D.shape)
            #D2=D.copy()
            #for id in self.cond_ids: # cond_ids is sorted
                #D2= np.insert(D2, id, 0, axis=0)
                #D2= np.insert(D2, id, 0, axis=1)
            #t2=(time.time() - start_time2)
            #print('shape after',D.shape)
            #print('time inserting D',t2)
            #print(D2[3999:])
            #print(D2.shape)
            #import pandas as pd
            #D3=pd.DataFrame(D2)
            #print(D3.iloc[3500:,:])
            #D3.to_csv('D2.csv',index=False)
            start_time3 = time.time()
            #import pandas as pd 
            #D3=D.copy()
            #D3=pd.DataFrame(D3)
            #print(D3)
            #print(np.array(self.cond_ids))
            #sc=np.array(self.cond_ids)
            #print(sc)
            #print(sc.dtype)
            c=len(self.cond_ids)
            N=D.shape[0]
            b= np.zeros((N,N+c))
            b[:,:-c] = D
            k= np.zeros((N+c,N+c))
            k[:-c,:] = b
            import pandas as pd 
            #print(pd.DataFrame(k))
            D=k
            #D4=pd.DataFrame(D)
            #print(D4.iloc[3500:,:])
            #D4.to_csv('D.csv',index=False)
            t3=(time.time() - start_time3)
            #print('time inserting D',t3)
            
            print(D.shape)
            
        #print('final scored kernel',D)
        return D
    
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        D = self.kernel[p_ids,:]
        D = D[:,q_ids]
        
        #print('final',D)
        return D

class Sampler(object):
    def __init__(self, kernel, space, k, cond_ids=[]):
        self.kernel = kernel
        self.space = space
        self.k = k
        self.cond_ids = cond_ids
        # norms will hold the diagonals of the kernel
        self.norms = np.array([self.kernel[p_id, p_id][0][0] for p_id in range(len(self.space))])
        self.clear()
    def clear(self):
        # S will hold chosen set of k points
        self.S = list(self.cond_ids)
        # M will hold the inverse of the kernel on S
        if len(self.cond_ids) == 0:
            self.M = np.zeros(shape=(0, 0))
        else:
            #print('creating S and M to hold selected points')
            self.M = np.linalg.pinv(self.kernel[self.S, self.S])
            #print(self.S)
    def makeSane(self):
        self.M = np.linalg.pinv(self.kernel[self.S, self.S])
    def testSanity(self):
        eps = 1e-4
        assert self.M.shape == (len(self.S), len(self.S))
        if len(self.S)>0:
            #print('Test Sanity')
            diff = np.abs(np.dot(self.M, self.kernel[self.S, self.S])-np.eye(len(self.S)))
            assert np.all(np.abs(diff)<=eps)
    def append(self, ind):
        if len(self.S)==0:
            self.S = [ind]
            self.M = np.array([[1./self.norms[ind]]])
        else:
            #print('Appending')
            u = self.kernel[self.S, ind]
            # Compute Schur complement inverse
            v = np.dot(self.M, u)
            scInv = 1./(self.norms[ind]-np.dot(u.T, v))
            self.M = np.block([[self.M+scInv*np.outer(v, v), -scInv*v], [-scInv*v.T, scInv]])
            self.S.append(ind)
    def remove(self, i):
        if len(self.S)==1:
            self.S = []
            self.M = np.zeros(shape=(0, 0))
        else:
            #print('Removing')
            mask = [True]*len(self.S)
            mask[i] = False
            # Readable version: mask = [j!=i for j in range(len(self.S))]
            scInv = self.M[i, i]
            v = self.M[mask, i]
            self.M = self.M[mask, :][:, mask] - np.outer(v, v)/scInv
            self.S = self.S[:i] + self.S[i+1:]
            
    # Return array containing ratio of increase in kernel determinant after adding each point in space
    def ratios(self, item_ids=None):
        #print('Printing Ratios')
        if item_ids is None:
            item_ids = np.arange(len(self.space))
        if len(self.S)==0:
            #print('Length S=0: ',-np.sort(-self.norms[item_ids])[0:10])
            #ii=np.argmax(self.norms[item_ids])
            #self.ii=ii
            #print('id of point',ii)
            #print('point is',self.space[self.ii])
            return self.norms[item_ids]
        else:
            #print('Getting Ratio')
            U = self.kernel[item_ids, self.S]
            
            #print('Length S>0: ',-np.sort(-(self.norms[item_ids] - np.sum(np.dot(U, self.M)*U, axis=1)))[0:10])
            
            return self.norms[item_ids] - np.sum(np.dot(U, self.M)*U, axis=1)
    
    # Finds greedily the item to add to maximize the determinant of the kernel
    def addGreedy(self):
        self.append(np.argmax(self.ratios()))
    # Important step, because we need to start from a point whose probability is not too small
    def warmStart(self):
        self.clear()
        for i in range(self.k):
            self.addGreedy()
            
    def keepCurrentState(self):
        self.backup_S = self.S.copy()
        self.backup_M = self.M.copy()
    def restoreState(self):
        self.S = self.backup_S.copy()
        self.M = self.backup_M.copy()
        
    # Run one step of Markov chain
    def step(self, alpha=1.):
        temp = np.random.randint(len(self.cond_ids),len(self.S))
        remove_id = self.S[temp]
        self.remove(temp)
        
        add_id = np.random.randint(len(self.space))
        
        new_prob = np.maximum(self.ratios(add_id), 0.)**alpha
        old_prob = np.maximum(self.ratios(remove_id), 0.)**alpha
        
        if np.random.rand() < new_prob / old_prob:
            self.append(add_id)
            #print(add_id)
        else:
            self.append(remove_id)
            #print(remove_id)
        #print(self.S)
        
    def sample(self, alpha=2., steps=1000, makeSaneEvery=10):
        for iter in range(steps):
            self.step(alpha=alpha)
            if iter%makeSaneEvery==0:
                self.makeSane()
        return self.S

def setup_sampler(space, scores, k, alpha, gamma, cond_ids):
    sp = np.array(space)
    v = np.prod([np.max(sp[:,i])-np.min(sp[:,i]) for i in range(sp.shape[1])])
    d = sp.shape[1]
    R = np.exp(np.log(v)/d - np.log(2*(k+len(cond_ids)))/d) # 2 is an empirical factor
    #print('R is:',R)
    s = Sampler(ConditionedScoredKernel(R, space, scores,cond_ids, alpha, gamma), space, k, cond_ids)
    s.warmStart()
    return s

def sample_ids_mc(points, scores, k, alpha=2., gamma=1., cond_ids=[], steps=1000):
    points = [p for p in points]
    scores = np.array(scores).reshape(-1,1)
    s = setup_sampler(points, scores, k, alpha, gamma, cond_ids)
    x = s.sample(alpha=alpha, steps=steps)
    return x[len(cond_ids):]

def sample_mc(points, scores, k, alpha=2., gamma=1., cond_ids=[]):
    x = sample_ids_mc(points, scores, k, alpha, gamma)
    return np.array(points)[x]


# In[ ]:




