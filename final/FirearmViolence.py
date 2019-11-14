#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:18:30 2019

@author: GalenByrd
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import plotly
import plotly.graph_objs as go

# make figures better:
font = {'weight':'normal','size':20}
plt.rc('font', **font)
#plt.rc('figure', figsize=(10.0, 7.5))
plt.rc('figure', figsize=(8.0, 6.0))
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('legend',**{'fontsize':16})

######################## FUNCTIONS ###############################
def scatterFunc(data,xaxis,yaxis):
    """
    Docstring: scatter plot of variables for a given dataset
    """
    plt.scatter(data[xaxis],data[yaxis])
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    #plt.title(xaxis+' by '+yaxis)
    
def infer(some_count_data):
    """
    Docstring: Run bayesian inference on any count data. Outputs distributions/trace plots of
    lambda_1, lambda_2 and tau in addition to returning a list of expected values
    to be able to plot observed vs expected
    source: https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb
    """
    n_count_data = len(some_count_data)
    with pm.Model() as model:
        alpha = 1.0/some_count_data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
    #with model:
        idx = np.arange(n_count_data)
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
    #with model:
        observation = pm.Poisson("obs", lambda_, observed=some_count_data)
    #with model:
        step = pm.Metropolis()
        trace = pm.sample(10000, tune=5000,step=step)
    
    lambda_1_samples = trace['lambda_1']
    lambda_2_samples = trace['lambda_2']
    tau_samples = trace['tau']
    print(pm.gelman_rubin(trace))
    pm.traceplot(trace)
    N = tau_samples.shape[0]
    expected_violence = np.zeros(n_count_data)
    for day in range(0, n_count_data):
        # ix is a bool index of all tau samples corresponding to
        # the switchpoint occurring prior to value of 'day'
        ix = day < tau_samples
        # Each posterior sample corresponds to a value for tau.
        # for each day, that value of tau indicates whether we're "before"
        # (in the lambda1 "regime") or
        #  "after" (in the lambda2 "regime") the switchpoint.
        # by taking the posterior sample of lambda1/2 accordingly, we can average
        # over all samples to get an expected value for lambda on that day.
        # As explained, the "count" random variable is Poisson distributed,
        # and therefore lambda (the poisson parameter) is the expected value of
        # "count".
        expected_violence[day] = (lambda_1_samples[ix].sum()+ lambda_2_samples[~ix].sum()) / N
    return expected_violence

def plotStates(colored,title,label): 
    """
    Docstring: Create plotly visualization of US states colored by the variable colored
    Currently only using stateData, but could be generalized to accept any dataset
    """
    #scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#ff4d4d'],[0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#4d0000']] # reds
    data = [go.Choropleth(
        #colorscale = scl,
        colorscale = 'RdBu',
        autocolorscale = False,
        locations = stateData['Initials'],
        z = stateData[colored].astype(float),
        locationmode = 'USA-states',
        text = stateData['State'],
        marker = go.choropleth.Marker(
            line = go.choropleth.marker.Line(
                color = 'rgb(255,255,255)',
                width = 2)),
        colorbar = go.choropleth.ColorBar(title = label))]
    layout = go.Layout(
        title = go.layout.Title(text = title),
        geo = go.layout.Geo(
            scope = 'usa',
            projection = go.layout.geo.Projection(type = 'albers usa'),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),)
    fig = go.Figure(data = data, layout = layout)
    plotly.offline.plot(fig,auto_open=True)

##### READ IN/CLEAN DATA #########################################
happiness_data = pd.read_csv('data/2017WorldHappinessReport.csv')
world_data = pd.read_csv('data/World firearms murders.csv')
world_data.rename(columns={'Country/Territory': 'Country'}, inplace=True)
worldDataMerged = pd.merge(world_data, happiness_data, how="left", on="Country")

massData = pd.read_csv('data/Mass Shootings Dataset Ver 5.csv',encoding = "ISO-8859-1")
massData['Date'] = pd.to_datetime(massData['Date'])
massData['year'] = massData['Date'].dt.year
massData['month'] = massData['Date'].dt.month
massData['monthday'] = massData['Date'].dt.day
massData['weekday'] = massData['Date'].dt.weekday

violenceData = pd.read_csv('data/gun-violence-data_01-2013_03-2018.csv')
violenceData.loc[len(violenceData)] = ['sban_1', '2017-10-01', 'Nevada', 'Las Vegas', 'Mandalay Bay 3950 Blvd S', 59, 489, 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', '-', '-', '-', '-', '-', '36.095', 'Hotel','-115.171667', 47, 'Route 91 Harvest Festiva; concert, open fire from 32nd floor. 47 guns seized; TOTAL:59 kill, 489 inj, number shot TBD,girlfriend Marilou Danley POI', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
violenceData['date'] = pd.to_datetime(violenceData['date'])
violenceData['year'] = violenceData['date'].dt.year
violenceData['month'] = violenceData['date'].dt.month
violenceData['monthday'] = violenceData['date'].dt.day
violenceData['weekday'] = violenceData['date'].dt.weekday
violenceData['loss'] = violenceData['n_killed'] + violenceData['n_injured']

stateOwnership = pd.read_csv('data/data-bo8OY.csv')
stateDeathRates = pd.read_csv('data/FIREARMS2017.csv')
stateDeathRates.rename(columns={'STATE': 'Initials'}, inplace=True)
stateData = pd.merge(stateDeathRates, stateOwnership, how="left", on="Initials")

df = pd.DataFrame(violenceData.groupby('state')['incident_id'].nunique())
stateData = pd.merge(stateData, df, how="outer", left_on="State", right_index=True)

df = pd.DataFrame(violenceData.groupby('state')['n_killed'].sum())
stateData = pd.merge(stateData, df, how="outer", left_on="State", right_index=True)
stateData['deadPerIncident'] = stateData['n_killed']/stateData['incident_id']

df = pd.DataFrame(violenceData.groupby('state')['n_injured'].sum())
stateData = pd.merge(stateData, df, how="outer", left_on="State", right_index=True)
stateData['injuredPerIncident'] = stateData['n_injured']/stateData['incident_id']

df = pd.DataFrame(violenceData.groupby('state')['n_guns_involved'].sum())
stateData = pd.merge(stateData, df, how="outer", left_on="State", right_index=True)

df = pd.DataFrame(violenceData.set_index('state').isna().sum(level=0)) 
stateData = pd.merge(stateData, pd.DataFrame(df['n_guns_involved']), how="outer", left_on="State", right_index=True)
stateData['gunsPerIncident'] = stateData['n_guns_involved_x']/(stateData['incident_id']-stateData['n_guns_involved_y'])
# FROM ATF 2007
stateData['RegisteredFirearms'] = [161641,15824,179738,79841,344622,92435,82400,4852,343288,190050,7859,49566,146487,114019,28494,52634,81068,116831,15371,103109,37152,66742,79307,35494,72996,22133,22234,76888,64135,57507,97580,76207,152238,13272,173405,71269,61383,236377,4223,105601,21130,99159,588696,72856,5872,307822,91835,35264,64878,132806,47228]

# FROM ATF 2007
manufacturingData = pd.DataFrame({'year':list(range(1986,2016)),'manufactured':[3040934,3559663,3963877,4418393,3959968,3563106,4175836,5055637,5173217,4316342,3854439,3593504,3713590,4047747,3793541,2932655,3366895,3308404,3099025,3241494,3653324,3922613,4498944,5555818,5459240,6541886,8578610,10844792,9050626,9358661],
                                  'exported':[217448,288045,257182,261282,361625,401165,403791,431204,404473,424221,331997,275081,203344,220318,172627,172611,150756,141912,139920,194682,367521,204782,228488,194744,241977,296888,287554,393121,420932,343456],
                                  'imported':[701000,1063513,1267268,1007781,843809,720657,2846710,3043321,1880902,1103404,881578,939415,999810,891799,1096782,1366896,1629237,1466502,1910859,2106675,2432522,2743993,2606386,3607109,2839947,3252404,4844590,5539539,3625268,3930211],
                                  'ownership%house':[44.2,46,39.8,46,42.2,39.6,40.8,42,40.6,40.35,40.1,37.45,34.8,33.6,32.4,32.95,33.5,34.1,34.7,33.9,33.1,33.55,34,32.55,31.1,32.1,33.1,32.05,31.0,None],
                                  #'ownership%house':[44.2,46,39.8,46,42.2,39.6,None,42,40.6,None,40.1,None,34.8,None,32.4,None,33.5,None,34.7,None,33.1,None,34,None,31.1,None,33.1,None,31.0,None],
                                  'personal%':[30.5,28,24.9,27.1,28.5,27,28.15,29.3,28.2,27.65,27.1,24.75,22.4,22.35,22.3,24.3,26.3,25.75,25.2,23.4,21.6,22.55,23.5,22.05,20.6,21.2,21.8,22.1,22.4,None]})
                                #'personal%':[30.5,28,24.9,27.1,28.5,27,None,29.3,28.2,None,27.1,None,22.4,None,22.3,None,26.3,None,25.2,None,21.6,None,23.5,None,20.6,None,21.8,None,22.4,None]})

######################   PRELIM EXPLORATION ########################

violenceData.hist()
worldDataMerged.hist()
manufacturingData.hist()
massData.hist()
stateData.hist()

worldCorrelations = worldDataMerged.corr()
massCorrelations = massData.corr()
eventsCorrelations = violenceData.corr()
manufacturingCorrelations = manufacturingData.corr()
stateCorrelations = stateData.corr()

worldDescription = worldDataMerged.describe()
massDescription = massData.describe()
eventsDescription = violenceData.describe()
manufacturingDescription = manufacturingData.describe()
stateDescription = stateData.describe()


plt.plot(happiness_data['Happiness.Rank'],happiness_data['Happiness.Score'])
scatterFunc(worldDataMerged,'Average firearms per 100 people','Happiness.Score')
scatterFunc(worldDataMerged,'Average firearms per 100 people','Happiness.Rank')
plt.plot(worldDataMerged['Average firearms per 100 people'])
scatterFunc(worldDataMerged,'Rank by rate of ownership','Happiness.Rank')
scatterFunc(worldDataMerged,'% of homicides by firearm','Freedom')
scatterFunc(worldDataMerged,'Number of homicides by firearm','Freedom')
scatterFunc(worldDataMerged,'Number of homicides by firearm','Generosity')
scatterFunc(worldDataMerged,'Happiness.Score','Economy..GDP.per.Capita.')


scatterFunc(worldDataMerged,'Average firearms per 100 people','Happiness.Score')
scatterFunc(worldDataMerged,'Average firearms per 100 people','Happiness.Rank')
scatterFunc(worldDataMerged,'Average firearms per 100 people','Health..Life.Expectancy.')
scatterFunc(worldDataMerged,'Average firearms per 100 people','Economy..GDP.per.Capita.')
scatterFunc(worldDataMerged,'Economy..GDP.per.Capita.','Happiness.Score')
scatterFunc(worldDataMerged,'Rank by rate of ownership','Health..Life.Expectancy.')
scatterFunc(worldDataMerged,'Rank by rate of ownership','Health..Life.Expectancy.')
scatterFunc(worldDataMerged,'Rank by rate of ownership','Average firearms per 100 people')

plt.scatter(worldDataMerged['Rank by rate of ownership'],worldDataMerged['Health..Life.Expectancy.'])
plt.xlabel('Rank by Rate of Firearm Ownership')
plt.ylabel('Life Expectancy')

scatterFunc(manufacturingData['year'],manufacturingData['ownership%house'])
scatterFunc(manufacturingData['year'],manufacturingData['personal%'])
plt.plot(manufacturingData['ownership%house'].dropna())


######### BAYESIAN INFERENCE ON MASSS SHOOTINGS ###############################
#count_data = massData.groupby('year')['Date'].count()    
count_data = pd.DataFrame(massData.groupby(['year','month'])['Date'].count())
mass_count_data = count_data

#mass_count_data = pd.DataFrame(massData.groupby(['year','month'])['Date'].count())
mass_count_data=mass_count_data.iloc[0:0]
#mass_count_data = mass_count_data.append(pd.Series({'Shootings':1,'year':1966,'month':8},name =(1966,8)))
#count_data['Date'].loc[(1996,8)]
#aadding zeros to months with no observed mass shootings 
for i in range (1966,2018):
    for j in range (1,13):
        try:
            shot = count_data['Date'].loc[(i,j)]
            mass_count_data = mass_count_data.append(pd.Series({'Date':shot,'year':i,'month':j},name =(i,j)))
        except:    
            mass_count_data = mass_count_data.append(pd.Series({'Date':0,'year':i,'month':j},name =(i,j)))
            #count_data.loc[(i,j)]=0
            #count_data['year']=i
            #count_data['month']=j
n_mass_count_data = len(mass_count_data)
     

plt.bar(np.arange(n_mass_count_data), mass_count_data['Date'], color="#348ABD")
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Number of Mass shootings in America per month')

expected_violence = infer(mass_count_data['Date'])

plt.plot(range(n_mass_count_data), expected_violence, lw=4, color="#E24A33", label="expected number of mass shootings")
plt.bar(np.arange(n_mass_count_data), mass_count_data['Date'], color="#348ABD", label="observed number of mass shootings")
plt.legend(loc="upper left")
plt.xlabel('Month')
#plt.xticks(count_data.index[0])
plt.ylabel('Frequency')
plt.title('Number of Mass shootings in America per month')

mass_count_data

######### BAYESIAN INFERENCE ON ALL SHOOTINGS ###############################
#count_data = violenceData.groupby('year')['date'].count() 
count_data = violenceData.groupby(['year','month'])['date'].count() 
count_data = count_data[12:]
#count_data = violenceData.groupby(['year','month','monthday'])['date'].count()
#count_data = count_data[177:]
n_count_data = len(count_data)  
        
        
expected_violence = infer(count_data)
plt.plot(np.arange(n_count_data), expected_violence, lw=4, color="#E24A33", label="expected number of firearm violence events")
plt.bar(np.arange(n_count_data), count_data, color="#348ABD", label="observed number of firearm violence events")
plt.legend(loc="lower right")
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Firearm violence events in America per month since Jan 2014')


######### BAYESIAN INFERENCE ON MANUFACTURING ###############################
expected_violence = infer(manufacturingData['manufactured'])
plt.plot(manufacturingData['year'], expected_violence, lw=4, color="#E24A33", label="expected number of guns manufactured")
plt.scatter(manufacturingData['year'],manufacturingData['manufactured'])
plt.legend()#loc="upper right")
plt.xlabel('Year')
plt.ylabel('Number of Guns Manufactured (Millions)')
plt.title('Number of Guns Manufactured in America from 1986-2015 in Millions')
        

######### BAYESIAN INFERENCE ON IMPORTS ###############################        
expected_violence = infer(manufacturingData['imported'])
plt.plot(manufacturingData['year'], expected_violence, lw=4, color="#E24A33", label="expected number of guns imported")
plt.scatter(manufacturingData['year'],manufacturingData['imported'])
plt.legend()#loc="upper right")
plt.xlabel('Year')
plt.ylabel('Number of Guns imported')
plt.title('Number of Guns imported to America from 1986-2015')        
        

######### BAYESIAN INFERENCE ON HOME OWNERSHIP ###############################
#expected_violence = infer(manufacturingData['ownership%house'])
expected_violence = infer(manufacturingData.loc[0:28,'ownership%house'])
plt.plot(manufacturingData.loc[0:28,'year'], expected_violence, lw=4, color="#E24A33", label="expected ownership percent")
plt.scatter(manufacturingData['year'],manufacturingData['ownership%house'], label="household ownership percent")
plt.legend()#loc="upper right")
plt.xlabel('Year')
plt.ylabel('Percent')
plt.title('Percent of Americans living in a home with at least one firearm from 1986-2015') 


######### BAYESIAN INFERENCE ON PERSONAL OWNERSHIP ###############################
expected_violence = infer(manufacturingData.loc[0:28,'personal%'])
plt.plot(manufacturingData.loc[0:28,'year'], expected_violence, lw=4, color="#E24A33", label="expected individual ownership percent")
plt.scatter(manufacturingData['year'],manufacturingData['personal%'], label="individual ownership percent")
plt.legend()#loc="upper right")
plt.xlabel('Year')
plt.ylabel('Percent')
plt.title('Percent of Americans owning at least one firearm from 1986-2015')   


################################ STATE DATA VISUALIZATION ###############################

plotStates('Estimated Household Gun Ownership Percentage, 2016','2016 US Estimated Household Gun Ownership Percentage by State','Percent')
plotStates('RATE','2017 Firearm Death Rate by State','Rate')

plotStates('deadPerIncident','Deaths per Firearm Incident by State 2014-2018','Deaths')
plotStates('injuredPerIncident','Injuries per Firearm Incident by State 2014-2018','Injuries')
plotStates('gunsPerIncident','Number of guns per Firearm Incident by State 2014-2018','Guns')

        
######################## CODE GRAVE #######################################

plotly.tools.set_credentials_file(username='gbyrd', api_key='B7p2gmEq2gs2lCYjoreT')
plotly.plotly.plot(fig,auto_open=True)


