# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:57:25 2016

@author: Abhi Gupta
"""
import pandas as pd
import numpy as np
import matplotlib
import requests
import json
import statsmodels.api as sm

from matplotlib import pyplot as plt
import seaborn as sns

"""
Problem A
"""

data = pd.read_csv('mock_student_data.csv')

# data.describe doesn't give quite what we need. 
summary_stats = pd.DataFrame(index=['mean','median','mode','SD','num_missing'],
                                      columns=data.columns)
                                      
numeric_cols = ['Age','GPA','Days_missed']                                           
for col in numeric_cols:
    summary_stats.ix['mean',col] = data.ix[:,col].mean()
    summary_stats.ix['median',col] = data.ix[:,col].median()
    summary_stats.ix['SD',col] = np.sqrt(data.ix[:,col].var())
    
for col in data.columns:
    summary_stats.ix['num_missing',col] = (len(data.index)-data.count())[col]
    try:
        summary_stats.ix['mode',col] = data[col].mode()[0]
    except:
        continue

print(summary_stats)

fig =  data.State.value_counts().plot(kind = 'bar')
fig.get_figure().savefig('States.png')
fig =  data.Gender.value_counts().plot(kind = 'bar')
fig.get_figure().savefig('Gender.png')
fig =  data.Graduated.value_counts().plot(kind = 'bar')
fig.get_figure().savefig('Graduated.png')
fig = data.hist(layout=(2,2))
fig[0][1].get_figure().savefig('Hists.png')



## inferring missing genders
missing_genders = np.where(data['Gender'].isnull())[0]
data_2 = data.copy()    
for i in missing_genders:
    name = data_2.ix[i,'First_name']
    web_req = requests.get("http://api.genderize.io?name[0]="+name)
    result = json.loads(web_req.text)
    #print(result)
    if result['gender']=='male':
        data_2.ix[i,'Gender']='Male'
    else:
        data_2.ix[i,'Gender']='Female'

data_2.to_csv('gender_imputed.csv')        
        
        
    
## replacing missing values with mean
    
data_3 = data_2.copy()
   
for col in ['Age','GPA','Days_missed']:
    ind = pd.isnull(data_3[col])
    fill_val = data_3[col].mean()
    data_3.ix[ind,col] = fill_val
    
data_3.to_csv('method_A.csv')     
    
    

## replacing missing vals with conditional mean
data_4 = data_2.copy()
grouped = data_4.groupby('Graduated')
conditional_means = grouped.mean()
    
for col in ['Age','GPA','Days_missed']:
    ind = pd.isnull(data_4[col])
    fill_vals_func = lambda x: conditional_means.ix[x['Graduated'],col]
    fill_vals = fill_vals_func(data_4.ix[ind,:])
    fill_vals.index = data_4.ix[ind,:].index
    data_4.ix[ind,col] = fill_vals

data_4.to_csv('method_B.csv')



## replacing missing vals with finer conditional means     
data_5 = data_2.copy()

# first, impute missing states
grouped = data_5.groupby('Graduated')
conditional_modes = grouped.agg(lambda x:x.value_counts().index[0])
for col in ['State']:
    ind = pd.isnull(data_5[col])
    fill_vals_func = lambda x: conditional_modes.ix[x['Graduated'],col]
    fill_vals = fill_vals_func(data_5.ix[ind,:])
    fill_vals.index = data_5.ix[ind,:].index
    data_5.ix[ind,col] = fill_vals
    
# find conditional 
grouped = data_5.groupby(['Graduated','Gender','State'])
conditional_means = grouped.mean()
for col in ['Age','GPA','Days_missed']:
    ind = pd.isnull(data_5[col])
    for row in np.where(ind==True)[0]:
        fill_val = conditional_means.ix[data_5.ix[row,'Graduated'],
                  data_5.ix[row,'Gender'],data_5.ix[row,'State']]
        data_5.ix[row,col] = fill_val[col]

data_5.to_csv('method_C.csv')
