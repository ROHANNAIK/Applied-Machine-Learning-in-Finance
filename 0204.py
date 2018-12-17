# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:23:22 2018

Kalman Filter: Preliminary (Toy Model)

@author: Yizhen Zhao 
"""

import numpy as np
import matplotlib.pyplot as plt
import xlrd 

T = 100
Y = np.random.normal(0,1,T)
"""xlrd.open_workbook("AMZN.xls")"""
 
S = Y.shape[0]
S = S + 1

"Initialize Params: "
Z = 1;
T = 0.5;
H = np.var(Y);
Q = 0.5*np.var(Y);

"Kalman Filter Starts:"
u_predict = np.zeros(S)
u_update = np.zeros(S)
P_predict = np.zeros(S)
P_update = np.zeros(S)
v = np.zeros(S)
F = np.zeros(S)
KF_Dens = np.zeros(S)
for s in range(1,S):
 if s == 1: 
    P_update[s] = 1000
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
 else: 
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
    v[s]=Y[s-1]-Z*u_predict[s-1]   
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s]; 
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    KF_Dens[s] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]      
    Likelihood = sum(KF_Dens[1:-1])
          
timevec = np.linspace(1,S,S)
plt.plot(timevec, u_update,'r',timevec[0:-1], Y,'b:')