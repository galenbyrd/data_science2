#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:18:30 2019

@author: GalenByrd
"""
import numpy as np
import math
import matplotlib.pyplot as plt

T = list(np.array(range(10,10001,10)))
overall = []
for t in T:
    oneRun = []
    X = []
    Y = []
    Z = []
    for i in range(t):
        x = np.random.binomial(size=1, n=1, p= 0.5)[0]
        X.append(int(x))
        y = np.random.binomial(size=1, n=1, p= (math.e**(4*x-2)/(1+math.e**(4*x-2))))[0]
        Y.append(int(y))
        z = np.random.binomial(size=1, n=1, p= (math.e**(2*(x+y)-2)/(1+math.e**(2*(x+y)-2))))[0]
        Z.append(int(z))
    oneRun.append(X)
    oneRun.append(Y)
    oneRun.append(Z)
    overall.append(oneRun)

probs = []
for i in overall:
    giveny=0.0
    zones=0.0
    for index,l in enumerate(i[1]):
        if (l==1):
            giveny+=1
            if (i[2][index]==1):
                zones+=1
    pr = zones/giveny
    probs.append(pr)

    
plt.plot(T,probs, label='Pr (Z = 1 | Y = 1)')
plt.plot(T,[.83]*len(T), label='Pr = 0.83')
plt.legend()
plt.xlabel("Duration of simulation")
plt.ylabel("Probability");   
    






#################### INTERVENTION #########################################
T = list(np.array(range(10,10001,10)))
overall = []
for t in T:
    oneRun = []
    X = []
    Y = list(np.ones(t))
    Z = []
    for i in range(t):
        x = np.random.binomial(size=1, n=1, p= 0.5)[0]
        X.append(int(x))
        z = np.random.binomial(size=1, n=1, p= (math.e**(2*(x+y)-2)/(1+math.e**(2*(x+y)-2))))[0]
        Z.append(int(z))
        Z.append(z)
    oneRun.append(X)
    oneRun.append(Y)
    oneRun.append(Z)
    overall.append(oneRun)

probs = []
for i in overall:
    zones=0.0
    for index,l in enumerate(i[1]):
        if (i[2][index]==1):
            zones+=1
    pr = zones/(index+1)
    probs.append(pr)

    
plt.plot(T,probs, label='Pr (Z = 1 | Y := 1)')
plt.plot(T, [.69]*len(T), label='Pr = 0.69')
plt.legend()
plt.xlabel("Duration of simulation")
plt.ylabel("Probability");   





















        
        if (x==0):
            y = np.random.binomial(size=1, n=1, p= (math.e**(-2)/(1+math.e**(-2))))[0]
        else:
            y = np.random.binomial(size=1, n=1, p= (math.e**(2)/(1+math.e**(2))))[0]
        Y.append(y)
        if (x==0 & y==0):
            z = np.random.binomial(size=1, n=1, p= (math.e**(-2)/(1+math.e**(-2))))[0]
        elif (x==1 & y==1):
            z = np.random.binomial(size=1, n=1, p= (math.e**(2)/(1+math.e**(2))))[0]
        else:
            z = np.random.binomial(size=1, n=1, p= (math.e**(0)/(1+math.e**(0))))[0]
        Z.append(z)



    
    
#y1 = np.random.binomial(size=5000, n=1, p= (math.e**(2)/(1+math.e**(2))))
#y2 = np.random.binomial(size=5000, n=1, p= (math.e**(-2)/(1+math.e**(-2))))
#z1 = np.random.binomial(size=2500, n=1, p= (math.e**(2)/(1+math.e**(2))))
#z2 = np.random.binomial(size=2500, n=1, p= (math.e**(-2)/(1+math.e**(-2))))
#z3 = np.random.binomial(size=5000, n=1, p= (math.e**(.5)/(1+math.e**(.5))))
#Y = list(y1)+list(y2)
#Z = list(z1)+list(z2)+list(z3)

#for x in X:
#    y = np.random.binomial(size=1, n=1, p= (math.e**(4x-2)/(1+(4x-2))))




X = bernoulli.rvs(size=10000,p=0.5)

#for x in X:
#    y = bernoulli.rvs(size=1,p=(math.e**(4x-2)/(1+(4x-2))))
    

