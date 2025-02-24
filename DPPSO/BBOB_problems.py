#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import math
import ioh

def sphere1(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Sphere", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

def ellipsoid2(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Ellipsoid", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

def rastrigin3(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Rastrigin", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)  


def buecheRastrigin4(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "BuecheRastrigin", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def linearSlope5(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "LinearSlope", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

 
def attractiveSector6(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "AttractiveSector", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

    
def stepEllipsoid7(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "StepEllipsoid", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)   
    
def rosenbrock8(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Rosenbrock", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)    
    
def rosenbrockRotated9(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "RosenbrockRotated", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)    

    
def ellipsoidRotated10(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "EllipsoidRotated", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans) 

def discus11(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Discus", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def bentCigar12(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "BentCigar", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def sharpRidge13(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "SharpRidge", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def differentPowers14(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "DifferentPowers", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def rastriginRotated15(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "RastriginRotated", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def weierstrass16(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Weierstrass", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def schaffers17(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Schaffers10", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def schaffers18(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Schaffers1000", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def griewankRosenBrock19(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "GriewankRosenBrock", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def schwefel20(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Schwefel", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def gallagher21(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Gallagher101", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def gallagher22(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Gallagher21", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)


def katsuura23(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "Katsuura", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

def lunacekBiRastrigin24(x):
    import ioh
    import numpy as np 
    d=len(x)
    problem = ioh.get_problem(
    "LunacekBiRastrigin", 
    instance=1,
    dimension=d
    )
    ans=problem(x)
    return np.array(ans)

