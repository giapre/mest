#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Drug set ups

def s1_inters_bif_line(s1p, s1x0): #intersection of the bifurcation with dz = 0
        s1z, s1x1 = s1p
        return (s1x1**3 + 2*s1x1**2 - 4.1 + s1z, s1x1 - s1x0 - s1z/4)
    
def s2_inters_bif_vert(s2p, s2z): #intersection of the bifurcation with the vertical line
    s2x1 = s2p
    return (s2x1**3 + 2*s2x1**2 - 4.1 + s2z)

def f1_find_the_delta():
    
    from scipy.optimize import fsolve
    
    f1x0 = np.arange(-3.025,-2.062,0.001)
    
    f1all_the_deltas = []
    for m in f1x0:
        f1z, f1x1 = fsolve(s1_inters_bif_line, (3, -1), m) 
        #print(f'The intersection between the bifurcation line and the nullcline of z for x0 = {m} is at point ({f1z};{f1x1})')
        
        f1inters1, f1inters2, f1inters3 = fsolve(s2_inters_bif_vert, (-2, -1, 1), f1z)
        #print(f'And the intersections between the vertical line of z = {f1z} are at x1 = {f1inters1, f1inters2, f1inters3}')
        #return (f1z, f1x1)
        
        f1delta = abs(f1inters1 - f1inters2)
        #print(f'And the gap to push x1 above the separatrix has to be more than {f1delta}')
        
        f1all_the_deltas.append(f1delta)
        
    plt.plot(f1x0, f1all_the_deltas)
        
    for ii, hh in enumerate(f1x0):
        f1x0[ii] = round(hh, 3)
    
    return(f1all_the_deltas, f1x0)


# In[3]:


def dose_effect(fdrug, fdose, fstarting_x0):
    
    fdose = fdose*12.3 #Conversion from human to mouse
    
    fall_the_deltas, fx0 = f1_find_the_delta()
    
    fmax_mest_effect = 29.9*np.log10(184.5) - 19.5 #I set the phenytoin as referral drug for max MEST effect by using
    #maximal phenytoin dose
    
    if fdrug == 'phenytoin':
        feffect = 29.9*np.log10(fdose) - 19.5 #mest curve for phenytoin
    if fdrug == 'carbamazepine':
        feffect = 15.8*np.log10(fdose) - 1.03 #mest curve for carbamazepine
    if fdrug == 'valproate':
        feffect = 9.97*np.log10(fdose) - 5.97 #mest curve for valproate
        
    frelative_drug_effect = feffect/fmax_mest_effect #relative % effect compared to maximal phenytoin
    
    fwhere_is_x0 = np.where(fx0 > fstarting_x0) #I seek for the starting x0 and corresponding i in the I-x0 relationship
    fstarting_x0_idx = fwhere_is_x0[0][0] -1 
    fstarting_i = fall_the_deltas[fstarting_x0_idx]
    
    feffect_ep_i = frelative_drug_effect*(2-fstarting_i) #I calculate the relative effect of the dose on the epileptor
    
    fwhere_is_i =np.where(fall_the_deltas < (feffect_ep_i + fstarting_i)) #I find the new x0 based on the relative effect
    ffinal_i_idx = fwhere_is_i[0][0] 
    ffinal_x0 = fx0[ffinal_i_idx]
    
    print(fstarting_x0, "became", ffinal_x0)
    return ffinal_x0


# In[ ]:




