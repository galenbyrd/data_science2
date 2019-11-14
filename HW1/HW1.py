#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:30:31 2019

@author: GalenByrd
"""
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import random


################ QUESTION 1 ######################################################
# CODE SOURCE: https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb
# MK1 = A
# MK2 = B

mk1 = 26751
mk1Kills = 183
mk1p = mk1Kills/mk1
mk2 = 27079
mk2Kills = 222
mk2p = mk2Kills/mk2
observations_mk1 = [0]*(mk1-mk1Kills) + [1]*mk1Kills
random.shuffle(observations_mk1)
observations_mk2 = [0]*(mk2-mk2Kills) + [1]*mk2Kills
random.shuffle(observations_mk2)

with pm.Model() as model:
    mk1dist = pm.Uniform("mk1", 0, 1)
    mk2dist = pm.Uniform("mk2", 0, 1)
    delta = pm.Deterministic("delta", mk1dist - mk2dist)
    obs_mk1 = pm.Bernoulli("obs_mk1", mk1dist, observed=observations_mk1)
    obs_mk2 = pm.Bernoulli("obs_mk2", mk2dist, observed=observations_mk2)
    step = pm.Metropolis()
    trace = pm.sample(20000, step=step)
    burned_trace=trace[1000:]
p_mk1_samples = burned_trace["mk1"]
p_mk2_samples = burned_trace["mk2"]
delta_samples = burned_trace["delta"]

pm.traceplot(trace)

print("Probability MK2 is DEADLIER than MK1: %.3f" % np.mean(delta_samples < 0))


################ QUESTION 2 ######################################################
# CODE SOURCE: https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb
    
count_data = np.loadtxt('data/txtdata.csv')
n_count_data = len(count_data)

with pm.Model() as model:
    alpha = 1.0/count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
with model:
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data)
with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']

print(pm.gelman_rubin(trace))
pm.traceplot(trace)


def func(t, phi1, phi2,lambda1,lambda2):
    return (lambda1 / (1. + np.exp(phi1*t + phi2))+lambda2)

with pm.Model() as model:
    alpha = 1.0/count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    phi1 = pm.Normal("phi1", 0,1.5)
    phi2 = pm.Normal("phi2", 0,1.5)  
    lambda_new = pm.Deterministic("lambda_new",func(idx,phi1,phi2,lambda_1,lambda_2))
  
with model:
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(func(idx,phi1,phi2,lambda_1,lambda_2) > idx, lambda_1, lambda_2)
with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data)
with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
lambda_new = trace['lambda_new']
phi1_samples = trace['phi1']
phi2_samples = trace['phi2']

pm.traceplot(trace)
print(pm.gelman_rubin(trace))


N = lambda_new.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    ix = day < lambda_new[:,day]
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()+ lambda_2_samples[~ix].sum()) / N


low_CI=[]
upper_CI=[]
for value in expected_texts_per_day:
    low_CI.append(value-stats.sem(expected_texts_per_day)/math.sqrt(len(expected_texts_per_day)))
    upper_CI.append(value+stats.sem(expected_texts_per_day)/math.sqrt(len(expected_texts_per_day)))
    

plt.figure(figsize=(12.5,8.5))
plt.plot(range(n_count_data), expected_texts_per_day, lw=1, color="purple",label="expected number of text-messages received")
plt.fill_between(range(n_count_data), low_CI, upper_CI, color = 'red', alpha = .7, label = '95% CI')
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,label="observed texts per day")
plt.legend(loc="upper left");
    
    
    
    





 
# =============================================================================
# CODE GRAVE
#
#    
## priors for our logistic (see below):
#beta  = pm.Normal("beta",  0, 0.001, value=0)
#alpha = pm.Normal("alpha", 0, 0.001, value=0)
## Both have mean zero and **tolerance** = 1/variance = 0.001.
#
#
#
#
#with pm.Model() as model:
#    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
#    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
#    p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))
#
## connect the probabilities in `p` with our observations through a
## Bernoulli random variable.
#with model:
#    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
#    
#    # Mysterious code to be explained in Chapter 3
#    start = pm.find_MAP()
#    step = pm.Metropolis()
#    trace = pm.sample(120000, step=step, start=start)
#    burned_trace = trace[100000::2]
#
#alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
#beta_samples = burned_trace["beta"][:, None]
#
#pm.traceplot(trace)
#
#t = np.linspace(temperature.min() - 5, temperature.max()+5, 50)[:, None]
#p_t = logistic(t.T, beta_samples, alpha_samples)
#
#mean_prob_t = p_t.mean(axis=0)
#
#f = plt.figure(figsize=(12.5,10))
#ax = f.add_subplot(311)
#
#ax.set_xlim(0, .02)
#ax.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
#         label="posterior of $p_A$", color="#A60628", normed=True)
#ax.vlines(mk1p, 0, 1000, linestyle="--", label="true $p_A$ (unknown)")
#ax.legend(loc="upper right")
#ax.set_title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
#
#ax2 = f.add_subplot(312)
#
#ax2.set_xlim(0, .02)
#ax2.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
#         label="posterior of $p_B$", color="#467821", normed=True)
#ax2.vlines(mk2p, 0, 1000, linestyle="--", label="true $p_B$ (unknown)")
#ax2.legend(loc="upper right")
#
#ax3 = f.add_subplot(313)
#ax3.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
#         label="posterior of delta", color="#7A68A6", normed=True)
#ax3.vlines(mk1p - mk2p, 0, 1000, linestyle="--",
#           label="true delta (unknown)")
#ax3.vlines(0, 0, 1000, color="black", alpha=0.2)
#ax3.legend(loc="upper right");
#
#
#
#
#
#
#
#ax = plt.subplot(311)
##ax.set_autoscaley_on(False)
#
#plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
#         label="posterior of $\lambda_1$", color="#A60628", normed=True)
#plt.legend(loc="upper left")
#plt.title(r"""Posterior distributions of the variables
#    $\lambda_1,\;\lambda_2,\;\tau$""")
#plt.xlim([15, 30])
#plt.xlabel("$\lambda_1$ value")
#
#ax = plt.subplot(312)
#ax.set_autoscaley_on(False)
#plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
#         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
#plt.legend(loc="upper left")
#plt.xlim([15, 30])
#plt.xlabel("$\lambda_2$ value")
#
#plt.subplot(313)
#w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
#plt.hist(tau_samples, bins=n_count_data, alpha=1,
#         label=r"posterior of $\tau$",
#         color="#467821", weights=w, rwidth=2.)
#plt.xticks(np.arange(n_count_data))
#
#plt.legend(loc="upper left")
#plt.ylim([0, .75])
#plt.xlim([35, len(count_data)-20])
#plt.xlabel(r"$\tau$ (in days)")
#plt.ylabel("probability");
#
#
#
#     
# n_flips = 100
# n_heads = 60
# 
# data = [0]*(n_flips-n_heads) + [1]*n_heads
# 
# # this is our prior distribution for coin flip prob p:
# p = pm.Uniform("p", lower=0, upper=1)
# obs = pm.Bernoulli("obs", p, value=data, observed=True)
# berR = pm.Bernoulli("berR", p)
# berO = pm.Bernoulli("berO", p, value=1, observed=True)
# 
# model = pm.Model([obs, p])
# mcmc = pm.MCMC(model)
# mcmc.sample(300000, 50000, 3)
# p_samples = mcmc.trace("p")[:]
# 
# 
# plt.hist(p_samples, bins=100, histtype='stepfilled', alpha=0.85, color="#A60628", normed=True, edgecolor="none")
# plt.xlabel("$p$")
# plt.ylabel(r"Pr($p \mid$data)")
# plt.xlim(0.4,0.8)
# =============================================================================
