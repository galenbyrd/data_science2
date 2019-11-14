#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:18:39 2019

@author: GalenByrd
"""

import matplotlib.pyplot as plt
import random
import math
import scipy, scipy.stats
import numpy as np

######################## FUNCTIONS ########################
# RETURNS REWARD WHEN ARM i IS PLAYED
def play_arm(i):
    return arm2pull[i].rvs()   

def regret(time, reward):
    regret = time*max(arm2trueExpectedReward.values()) - reward
    return regret

def random_gamble(timesteps):
    totalBestArm = 0.0
    listRandomRewards = []
    for t in range(timesteps):
        pull = random.randint(0,4)
        listRandomRewards.append(play_arm(pull))
        if (pull==4):
            totalBestArm +=1
    return regret(timesteps,sum(listRandomRewards)),totalBestArm/timesteps
 
def naiveGreedy(timesteps):
    armEstimatedRewards = [0.5,0.5,0.5,0.5,0.5]
    armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
    totalRewards = 0.0
    for t in range(timesteps):
        pull = armEstimatedRewards.index(max(armEstimatedRewards))
        reward = play_arm(pull)
        totalRewards += reward
        armRewardHistory[pull].append(reward)
        armEstimatedRewards[pull]= np.mean(armRewardHistory[pull])
    return regret(timesteps,totalRewards),len(armRewardHistory[4])/timesteps  
        
def epsilonFirstGreedy(timesteps,m):
    armEstimatedRewards = [0,0,0,0,0]
    armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
    totalRewards = 0.0
    if (timesteps> 5*m):
        for l in range(m):
            for i in range(5):
                r = play_arm(i)
                totalRewards += r
                armRewardHistory[i].append(r)
                armEstimatedRewards[i]= np.mean(armRewardHistory[i])
        biggestEstimate = armEstimatedRewards.index(max(armEstimatedRewards))
        for t in range(timesteps-5*m):
            reward = play_arm(biggestEstimate)
            totalRewards += reward
            armRewardHistory[biggestEstimate].append(reward)
            armEstimatedRewards[biggestEstimate] = np.mean(armRewardHistory[biggestEstimate])
            #biggestEstimate = armEstimatedRewards.index(max(armEstimatedRewards))
    else:
        for l in range(m):
            for i in range(5):
                    if (timesteps>5*l+i):
                        r = play_arm(i)
                        totalRewards += r
                        armRewardHistory[i].append(r)
    return regret(timesteps,totalRewards),len(armRewardHistory[4])/timesteps

def epsilonGreedy(timesteps,eps):
    armEstimatedRewards = [1,1,1,1,1]
    armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
    totalRewards = 0.0
    for t in range(timesteps):
        # PLAY BIGEST ESTIMATE WITH PR(1-epsilon)
        if (random.randint(0,100)>(100-eps)):
            biggestEstimate = armEstimatedRewards.index(max(armEstimatedRewards))
            reward = play_arm(biggestEstimate)
            totalRewards += reward
            armRewardHistory[biggestEstimate].append(reward)
            armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
        else:
            arm = random.randint(0,4)
            reward = play_arm(arm)
            armRewardHistory[arm].append(reward)
            totalRewards += reward
            armEstimatedRewards[arm] = np.mean(armRewardHistory[arm]) #sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
    return regret(timesteps,totalRewards),len(armRewardHistory[4])/timesteps

def UCB1(timesteps):
    armEstimatedRewards = []
    armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
    totalRewards = 0.0
    for i in range(5):
        r = play_arm(i)
        armEstimatedRewards.append(r + math.sqrt(2*math.log(i+1)/1))
        armRewardHistory[i].append(r)
    for t in range(timesteps-5):
        biggestEstimate = armEstimatedRewards.index(max(armEstimatedRewards)) 
        reward = play_arm(biggestEstimate)
        armRewardHistory[biggestEstimate].append(reward)
        totalRewards += reward
        armEstimatedRewards[biggestEstimate] = np.mean(armRewardHistory[biggestEstimate]) + math.sqrt(2*math.log(t+1)/len(armRewardHistory[biggestEstimate])) #sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate]) #sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
    return regret(timesteps,totalRewards),len(armRewardHistory[4])/timesteps
    
#################### QUESTION 1 ###############################################
# SET UP BANDITS
arm2pull = {}
arm2trueExpectedReward = {}
arm = 0
for a in [2,3,4,5,6]:
    b = 8-a
    arm2pull[arm] = scipy.stats.beta(a,b)
    arm2trueExpectedReward[arm] = a/(a+b)
    arm += 1
arms = list(arm2pull.keys())


# SHOW REWARD DISTRIBUTIONS
x = np.linspace(0,1,100)
for arm in arms:
    p = arm2pull[arm]
    a,b = p.args
    plt.plot(x,p.pdf(x), label="arm {}: $B(a = {}, b = {})$".format(arm,a,b))
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.xlabel("reward")
plt.ylabel("Prob. density")
plt.tight_layout()


#################### QUESTION 2 ###############################################   
T = np.array(range(10,1001,10))
# RANDOM
listRandomRegrets = []
listRandomPercent = []
num_runs = 10
for t in T:
    regrets = np.mean([random_gamble(t)[0] for run in range(num_runs)])
    percent = np.mean([random_gamble(t)[1] for run in range(num_runs)])
    listRandomRegrets.append(regrets)
    listRandomPercent.append(percent)
    
plt.plot(T,listRandomRegrets, label='{} Random Gambles'.format(num_runs))
plt.plot(T, T/4, label='$R = T/4$')
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");

# NAIVE GREEDY
listNaiveGreedyRegrets = []
listNaiveGreedyPercent = []
num_runs = 10
for t in T:
    regrets = np.mean([naiveGreedy(t)[0] for run in range(num_runs)])
    percent = np.mean([naiveGreedy(t)[1] for run in range(num_runs)])
    listNaiveGreedyRegrets.append(regrets)
    listNaiveGreedyPercent.append(percent)

plt.plot(T,listRandomRegrets, label='{} Random Gambles'.format(num_runs))
plt.plot(T, listNaiveGreedyRegrets, label='{} Naive Greedy Gambles'.format(num_runs))
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");

#EPSILON-FIRST GREEDY
listEpsilonFirstGreedyRegrets = []
listEpsilonFirstGreedyPercent = []
num_runs = 10
for t in T:
    regrets = np.mean([epsilonFirstGreedy(t,4)[0] for run in range(num_runs)])
    percent = np.mean([epsilonFirstGreedy(t,4)[1] for run in range(num_runs)])
    listEpsilonFirstGreedyRegrets.append(regrets)
    listEpsilonFirstGreedyPercent.append(percent)

plt.plot(T,listRandomRegrets, label='{} Random Gambles'.format(num_runs))
plt.plot(T, listNaiveGreedyRegrets, label='{} Naive Greedy Gambles'.format(num_runs))
plt.plot(T,listEpsilonFirstGreedyRegrets, label='{} Epsilon First Greedy Gambles'.format(num_runs))
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");

#EPSILON GREEDY
listEpsilonGreedyRegrets = []
listEpsilonGreedyPercent = []
num_runs = 10
for t in T:
    regrets = np.mean([epsilonGreedy(t,15)[0] for run in range(num_runs)])
    percent = np.mean([epsilonGreedy(t,15)[1] for run in range(num_runs)])
    listEpsilonGreedyRegrets.append(regrets)
    listEpsilonGreedyPercent.append(percent)

plt.plot(T,listRandomRegrets, label='{} Random Gambles'.format(num_runs))
plt.plot(T, listNaiveGreedyRegrets, label='{} Naive Greedy Gambles'.format(num_runs))
plt.plot(T,listEpsilonFirstGreedyRegrets, label='{} Epsilon First Greedy Gambles'.format(num_runs))
plt.plot(T, listEpsilonGreedyRegrets, label='{} Epsilon Greedy Gambles'.format(num_runs))
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");


plt.plot(T,listRandomPercent, label='Random')
plt.plot(T, listNaiveGreedyPercent, label='Naive Greedy')
plt.plot(T,listEpsilonFirstGreedyPercent, label='Epsilon First Greedy')
plt.plot(T, listEpsilonGreedyPercent, label='Epsilon Greedy')
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Percent of optimal pulls");


############# RUN IT ALL ##############
T = np.array(range(10,1001,10))
listRandomRegrets = list(np.zeros([len(T)]))
listRandomPercent = list(np.zeros([len(T)]))
listNaiveGreedyRegrets = list(np.zeros([len(T)]))
listNaiveGreedyPercent = list(np.zeros([len(T)]))
listEpsilonFirstGreedyRegrets = list(np.zeros([len(T)]))
listEpsilonFirstGreedyPercent = list(np.zeros([len(T)]))
listEpsilonGreedyRegrets = list(np.zeros([len(T)]))
listEpsilonGreedyPercent = list(np.zeros([len(T)]))
num_runs = 100
for index,t in enumerate(T):
    regrets = np.mean([random_gamble(t)[0] for run in range(num_runs)])
    percent = np.mean([random_gamble(t)[1] for run in range(num_runs)])
    listRandomRegrets[index] = regrets
    listRandomPercent[index] = percent
    regrets = np.mean([naiveGreedy(t)[0] for run in range(num_runs)])
    percent = np.mean([naiveGreedy(t)[1] for run in range(num_runs)])
    listNaiveGreedyRegrets[index] = regrets
    listNaiveGreedyPercent[index] = percent
    regrets = np.mean([epsilonFirstGreedy(t,4)[0] for run in range(num_runs)])
    percent = np.mean([epsilonFirstGreedy(t,4)[1] for run in range(num_runs)])
    listEpsilonFirstGreedyRegrets[index] = regrets
    listEpsilonFirstGreedyPercent[index] = percent
    regrets = np.mean([epsilonGreedy(t,85)[0] for run in range(num_runs)])
    percent = np.mean([epsilonGreedy(t,85)[1] for run in range(num_runs)])
    listEpsilonGreedyRegrets[index] = regrets
    listEpsilonGreedyPercent[index] = percent

plt.plot(T,listRandomRegrets, label='{} Random Gambles'.format(num_runs))
plt.plot(T, listNaiveGreedyRegrets, label='{} Naive Greedy Gambles'.format(num_runs))
plt.plot(T,listEpsilonFirstGreedyRegrets, label='{} Epsilon First Greedy Gambles'.format(num_runs))
plt.plot(T, listEpsilonGreedyRegrets, label='{} Epsilon Greedy Gambles'.format(num_runs))
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");


plt.plot(T,listRandomPercent, label='Random')
plt.plot(T, listNaiveGreedyPercent, label='Naive Greedy')
plt.plot(T,listEpsilonFirstGreedyPercent, label='Epsilon First Greedy')
plt.plot(T, listEpsilonGreedyPercent, label='Epsilon Greedy')
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Percent of optimal pulls");


#################### QUESTION 3 ###############################################
# SET UP EASY/HARD BANDITS
arm2pull = {}
arm2trueExpectedReward = {}
arm = 0
hard = [1.25,1.35,1.5,1.65,1.75]
easy =[10,125,250,375,490]
# switch out easy/hard in for loop
for a in hard:
    #b = 500-a  #EASY
    b = 3-a     #HARD
    arm2pull[arm] = scipy.stats.beta(a,b)
    arm2trueExpectedReward[arm] = a/(a+b)
    arm += 1
arms = list(arm2pull.keys())


# SHOW REWARD DISTRIBUTIONS
x = np.linspace(0,1,100)
for arm in arms:
    p = arm2pull[arm]
    a,b = p.args
    plt.plot(x,p.pdf(x), label="arm {}: $B(a = {}, b = {})$".format(arm,a,b))
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.xlabel("reward")
plt.ylabel("Prob. density")
plt.tight_layout()


#################### QUESTION 4 ###############################################
listUCB1Regrets = list(np.zeros([len(T)]))
listUCB1Percent = list(np.zeros([len(T)]))
listEpsilonGreedyRegrets = list(np.zeros([len(T)]))
listEpsilonGreedyPercent = list(np.zeros([len(T)]))
listEpsilonFirstGreedyRegrets = list(np.zeros([len(T)]))
listEpsilonFirstGreedyPercent = list(np.zeros([len(T)]))
num_runs= 10
for index,t in enumerate(T):
    regrets = np.mean([UCB1(t)[0] for run in range(num_runs)])
    percent = np.mean([UCB1(t)[1] for run in range(num_runs)])
    listUCB1Regrets[index] = regrets
    listUCB1Percent[index] = percent
    regrets = np.mean([epsilonGreedy(t,85)[0] for run in range(num_runs)])
    percent = np.mean([epsilonGreedy(t,85)[1] for run in range(num_runs)])
    listEpsilonGreedyRegrets[index] = regrets
    listEpsilonGreedyPercent[index] = percent
    regrets = np.mean([epsilonFirstGreedy(t,4)[0] for run in range(num_runs)])
    percent = np.mean([epsilonFirstGreedy(t,4)[1] for run in range(num_runs)])
    listEpsilonFirstGreedyRegrets[index] = regrets
    listEpsilonFirstGreedyPercent[index] = percent


plt.plot(T,listEpsilonFirstGreedyRegrets, label='{} Epsilon First Greedy Gambles'.format(num_runs))
plt.plot(T, listEpsilonGreedyRegrets, label='{} Epsilon Greedy Gambles'.format(num_runs))
plt.plot(T,listUCB1Regrets, label='{} UCB1 Gambles'.format(num_runs))
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Regret $R$");


plt.plot(T,listEpsilonFirstGreedyPercent, label='Epsilon First Greedy')
plt.plot(T, listEpsilonGreedyPercent, label='Epsilon Greedy')
plt.plot(T,listUCB1Percent, label='UCB1')
plt.legend()
plt.xlabel("Duration of gamble $T$")
plt.ylabel("Percent of optimal pulls");











# ########################## CODEGRAVE ###################
#def naiveGreedy(timesteps):
#    armEstimatedRewards = []
#    totalRewards = 0.0
#    totalBestArm = 1
#    for i in range(5):
#        r = play_arm(i)
#        armEstimatedRewards.append(r)
#        totalRewards += r
#    pull = armEstimatedRewards.index(max(armEstimatedRewards))
#    if (pull==4):
#            totalBestArm += timesteps-5
#    for t in range(timesteps-5):
#        totalRewards += play_arm(pull)
#    return regret(timesteps,totalRewards),totalBestArm/timesteps
# #computes the regret R of a strategy as a function of time over the course of a gamble.
#def regrets(time,arm2play):
#    regret = np.zeros([time])
#    reward = np.zeros([time])
#    maxReward = max(arm2trueExpectedReward.values())
#    for t in range(time):
#        reward[t] = play_arm(arm2play)
#        regret[t]=((t+1)*maxReward) - sum(reward)
#    return regret
#
#def naiveGreedy2(timesteps):
#    arm = random.randint(0,4)
#    totalRewards = 0.0
#    for t in range(timesteps):
#        reward = play_arm(arm)
#        totalRewards += reward
#    return regret(timesteps,totalRewards)#,totalBestArm/timesteps
#
#listRandomRegrets = []
#listRandomPercent = []
#num_runs = 10
#for t in T:
#    rewardl = []
#    percentl = []
#    for run in range(num_runs):
#        reward,percent = random_gamble(t)
#        rewardl.append(reward)
#        percentl.append(percentl)
#    listRandomRegrets.append(np.mean(rewardl))
#    listRandomPercent.append(np.mean(percentl))
#    
#    
#    
#def naiveGreedy(timesteps):
#    armEstimatedRewards = [1,1,1,1,1]
#    pull = random.randint(0,4)
#    totalRewards = 0.0
#    reward = play_arm(pull)
#    totalRewards += reward
#    armEstimatedRewards[pull] = reward
#    for j in range(timesteps):
#        reward = play_arm(pull)
#        totalRewards += reward
#    return regret(timesteps,totalRewards)
#
#
#def UCB1(timesteps): 
#    armUCB1Estimates = [1,1,1,1,1]
#    armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
#    totalRewards = 0.0
#    for j in range(timesteps):
#        biggestEstimate = armUCB1Estimates.index(max(armUCB1Estimates)) 
#        reward = play_arm(biggestEstimate)
#        armRewardHistory[biggestEstimate].append(reward)
#        totalRewards += reward
#        armUCB1Estimates[biggestEstimate] = np.mean(armRewardHistory[biggestEstimate]) + np.sqrt(2*np.log(j)/len(armRewardHistory[biggestEstimate])) #sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#        #armEstimatedRewards[biggestEstimate] = reward
#    return regret(timesteps,totalRewards)
#
## ALL
#listRandomRegrets = []
#listNaiveGreedyRegrets = []
#listEpsilonGreedyRegrets = []
#listEpsilonFirstGreedyRegrets = []
#num_runs = 100
#for T in t:
#    listRandomRegrets.append(random_gamble(T))
#    #listNaiveGreedyRegrets.append(np.mean([naiveGreedy2(T) for run in range(num_runs)]))
#    listEpsilonFirstGreedyRegrets.append(np.mean([epsilonFirstGreedy(T,3) for run in range(num_runs)]))
#    listEpsilonGreedyRegrets.append(np.mean([epsilonGreedy(T) for run in range(num_runs)]))
#
#plt.plot(t,listRandomRegrets, label='1 Random Gamble')
#plt.plot(t, listNaiveGreedyRegrets, label='{} Naive Greedy Gambles'.format(num_runs))
#plt.plot(t,listEpsilonFirstGreedyRegrets, label='{} Epsilon First Greedy Gambles'.format(num_runs))
#plt.plot(t, listEpsilonGreedyRegrets, label='{} Epsilon Greedy Gambles'.format(num_runs))
#plt.legend()
#plt.xlabel("Duration of gamble $T$")
#plt.ylabel("Regret $R$");
#
#
#def naiveGreedy(timesteps):
#    armEstimatedRewards = {}
#    armRewardHistory = {}
#    listGreedyRewards = np.zeros([timesteps])
#    listGreedyRegrets = np.zeros([timesteps])
#    for i in range(5):
#        armEstimatedRewards[i] = 1
#        armRewardHistory[i] = []
#    for j in range(timesteps):
#        biggestEstimate = max(armEstimatedRewards, key=armEstimatedRewards.get)
#        reward = play_arm(biggestEstimate)
#        armRewardHistory[biggestEstimate].append(reward)
#        listGreedyRewards[j] = reward
#        armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#        listGreedyRegrets[j] = regret(j,sum(listGreedyRewards))
#    return listGreedyRegrets[-1]
#
#def epsilonGreedy(timesteps):
#    armEstimatedRewards = {}
#    armRewardHistory = {}
#    listEpsilonGreedyRewards = np.zeros([timesteps])
#    listEpsilonGreedyRegrets = np.zeros([timesteps])
#    firstArm = random.randint(0,4)
#    armEstimatedRewards[firstArm] = play_arm(firstArm)
#    armRewardHistory[firstArm] = [armEstimatedRewards[firstArm]]
#    for t in range(timesteps):
#        biggestEstimate = max(armEstimatedRewards, key=armEstimatedRewards.get)
#        # PLAY BIGEST ESTIMATE WITH PR(1-epsilon)
#        if (random.randint(0,2)>0):
#            reward = play_arm(biggestEstimate)
#            armRewardHistory[biggestEstimate].append(reward)
#            listEpsilonGreedyRewards[t] = reward
#            armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#            listEpsilonGreedyRegrets[t] = regret(t,sum(listEpsilonGreedyRewards))
#        else:
#            arm = random.randint(0,4)
#            reward = play_arm(arm)
#            armRewardHistory[biggestEstimate].append(reward)
#            listEpsilonGreedyRewards[t] = reward
#            armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#            listEpsilonGreedyRegrets[t] = regret(t,sum(listEpsilonGreedyRewards))
#    return listEpsilonGreedyRegrets
#
#
#def epsilonFirstGreedy(timesteps,m):
#    totalReward = 0.0
#    armEstimatedRewards = {}
#    armRewardHistory = {}
#    listEpsilonFirstGreedyRewards = []
#    listEpsilonFirstGreedyRegrets = []
#    counter=1
#    for l in range(m):
#        for i in range(5):
#            r = play_arm(i)
#            totalReward += r
#            try:
#                armRewardHistory[i].append(r)
#                armEstimatedRewards[i] = sum(armRewardHistory[i])/len(armRewardHistory[i])
#            except:
#                armRewardHistory[i] = [r]
#                armEstimatedRewards[i] = r
#            listEpsilonFirstGreedyRewards.append(r)
#            listEpsilonFirstGreedyRegrets.append(regret(counter,sum(listEpsilonFirstGreedyRewards)))
#            counter+= 1
#    for t in range(5*m,timesteps):
#        biggestEstimate = max(armEstimatedRewards, key=armEstimatedRewards.get)
#        reward = play_arm(biggestEstimate)
#        totalReward += reward
#        armRewardHistory[biggestEstimate].append(reward)
#        listEpsilonFirstGreedyRewards.append(reward)
#        armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#        listEpsilonFirstGreedyRegrets.append(regret(t,sum(listEpsilonFirstGreedyRewards)))
#    return listEpsilonFirstGreedyRegrets[-1]
#
#def naiveGreedy2(timesteps):
#    armEstimatedRewards = [1,1,1,1,1]
#    #armRewardHistory = {0:[],1:[],2:[],3:[],4:[]}
#    totalRewards = 0.0
#    for j in range(timesteps):
#        biggestEstimate = armEstimatedRewards.index(max(armEstimatedRewards))
#        reward = play_arm(biggestEstimate)
#        #armRewardHistory[biggestEstimate].append(reward)
#        totalRewards += reward
#        armEstimatedRewards[biggestEstimate] = reward
#        #armEstimatedRewards[biggestEstimate] = np.mean(armRewardHistory[biggestEstimate]) #sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
#        #armEstimatedRewards[biggestEstimate] = reward
#    return regret(timesteps,totalRewards)
#
## def randomAlgorithm(timesteps):
##    listRandomRegrets = []
##    maxReward = max(arm2trueExpectedReward.values())
##    for t in range(1,timesteps):
##        gambleReward = np.zeros([t])
##        gambleRegret = np.zeros([t])
##        for time in range(t):
##            gambleReward[time] = play_arm(random.randint(0,4))
##            gambleRegret[time]=((time+1)*maxReward) - sum(gambleReward)
##        listRandomRegrets.append(gambleRegret[-1])
##    return listRandomRegrets
##
##def naiveGreedy(timesteps):
##    # try all bandits first, then use the max rHAT  in dictionary and update on each call.
##    regretHistory = np.zeros([timesteps])
##    armEstimatedRewards = {}
##    armRewardHistory = {}
##    maxReward = max(arm2trueExpectedReward.values())
##    for i in range(5):
##        armRewardHistory[i] = [play_arm(i)]
##        armEstimatedRewards[i] = armRewardHistory[i][0]
##        regretHistory[i] = ((i+1)*maxReward) - armRewardHistory[i][0]
##    for t in range(5,timesteps):  
##        biggestEstimate = max(armEstimatedRewards, key=armEstimatedRewards.get)
##        #biggestEstimate = max(armRewardHistory.values(), key=operator.itemgetter(1))[0] #USE THIS ARM 
##        armRewardHistory[biggestEstimate].append(play_arm(biggestEstimate))
##        armEstimatedRewards[biggestEstimate] = sum(armRewardHistory[biggestEstimate])/len(armRewardHistory[biggestEstimate])
##        regretHistory[t] = ((t)*maxReward) - armRewardHistory[biggestEstimate][0]
##    return regretHistory
##
##def naiveGreedy2(timesteps):
##    listGreedyRegrets = []
##    maxReward = max(arm2trueExpectedReward.values())
##    for t in range(1,timesteps):
##        gambleArmEstimatedRewards = {}
##        gambleArmRewardHistory = {}
##        gambleRegret = []
##        for i in range(5):
##            gambleArmRewardHistory[i] = [play_arm(i)]
##            gambleArmEstimatedRewards[i] = gambleArmRewardHistory[i][-1]
##            gambleRegret.append( ((i+1)*maxReward) - gambleArmRewardHistory[i][-1])
##        for time in range(4,t):
##            biggestEstimate = max(gambleArmEstimatedRewards, key=gambleArmEstimatedRewards.get)
##            gambleArmRewardHistory[biggestEstimate].append(play_arm(biggestEstimate))
##            gambleArmEstimatedRewards[biggestEstimate] = sum(gambleArmRewardHistory[biggestEstimate])/len(gambleArmRewardHistory[biggestEstimate])
##            gambleRegret.append(((t)*maxReward) - gambleArmRewardHistory[biggestEstimate][-1])
##        listGreedyRegrets.append(gambleRegret[-1])
##    return listGreedyRegrets
##
### IMPLEMENTS A GAMBLE OVER T TIMESTEPS
##for pull in range(3):
##    print("Reward for pull {} =".format(pull), play_arm(4))
##
### EVALUATING REWARDS    
##r_star = max(arm2trueExpectedReward.values())
##arm2gap = {}
##for arm in arm2pull:
##    r_arm = arm2trueExpectedReward[arm]
##    gap = r_star - r_arm
##    arm2gap[arm] = gap
###arm2gap = {arm:r_star-arm2trueExpectedReward[arm] for arm in arm2pull}
##print("r* =", r_star)
