# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:34:36 2016

@author: Abhi
"""

import pipe as pipe
import handleData as handle
import pandas as pd
import  os
import difflib
import re
import numpy as np
import csv
import handleData as handle


os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')

country_codes = handle.readcsv('country_to_code.csv')
inequality = pd.read_excel('inequality_data.xlsx')
macro = pd.read_excel('wb_macro_data.xlsx')
survey = pd.read_excel('agg_survey_data.xlsx')
micro_world = handle.readcsv('micro_world.csv')

inequality = inequality.drop('Unnamed: 3',1)
survey = survey.drop('Unnamed: 4',1)

def fuzzy_match(a, b, thresh):
    left = '1' if pd.isnull(a) else a
    right = b.fillna('2')
    out = difflib.get_close_matches(left, right,cutoff=thresh)
    return out[0] if out else np.NaN

indexer = lambda data,thresh: data.economy.map(lambda x: fuzzy_match(x,country_codes.economy,thresh))

macro['economy_new'] = indexer(macro,.90)
macro = macro.dropna(subset=['economy_new'])
inequality['economy_new'] = indexer(inequality,.90)
inequality = inequality.dropna(subset=['economy_new'])
survey['economy_new'] = indexer(survey,.90)
survey = survey.dropna(subset=['economy_new'])


results = pd.merge(country_codes,macro,left_on='economy',right_on='economy_new',how='left')
results['economy'] = results['economy_x']
results = results.drop(['Unnamed: 3','economy_x','economy_y','economy_new'],1)
results = pd.merge(results,inequality,left_on='economy',right_on='economy_new',how='left')
results['economy'] = results['economy_x']
results = results.drop(['economy_x','economy_y','economy_new'],1)
results = results.replace('..',np.NaN)
results = results.convert_objects(convert_numeric=True)
#tester = lambda x: np.asarray([type(y) is not str for y in x])
#missing_economies = results[tester(results.economy)]
#for r in missing_economies.index:
    #results.ix[r,'economy'] = country_codes[country_codes.economycode==results.ix[r,'economycode']]['economy']  
results.to_excel('macro_vars.xlsx')

results2 = pd.merge(country_codes,survey,left_on='economy',right_on='economy_new',how='left')
results2['economy'] = results2['economy_x']
results2 = results2.drop(['economy_x','economy_y','economy_new'],1)
results2 = results2.replace('..',np.NaN)
results2 = results2.convert_objects(convert_numeric=True)
results2.to_excel('agg_survey_vars.xlsx')

# a list of macro var names for future use
macro_var_names = list(results.columns)+list(results2.columns)
while 'economy' in macro_var_names:
    macro_var_names.remove('economy')
while 'economycode' in macro_var_names:
    macro_var_names.remove('economycode')    
macro_var_names = pd.DataFrame(macro_var_names)
macro_var_names.to_csv('macro_var_names.csv',index=False,header=False)


final_data = pd.merge(micro_world,results,on=['economycode','economy'],how='left')
final_data = pd.merge(final_data,results2,on=['economycode','economy'],how='left')
final_data = final_data.fillna(0)
#missing_economies = final_data[final_data.economy==np.NaN]
#for r in missing_economies.index:
#    final_data.ix[r,'economy'] = country_codes[country_codes.economycode==final_data.ix[r,'economycode']]['economy']
duplicated = [x for x in final_data.columns if '_x' in x]
normal = lambda x: x[0:x.index('_')]
duplicated = [normal(x) for x in duplicated]
for col in duplicated:
    final_data[col] = final_data[col+'_x']
    final_data = final_data.drop(col+'_x',1)
    final_data = final_data.drop(col+'_y',1)

# imputation by region
grouped = final_data.groupby('regionwb')
n_groups = len(grouped.groups)
n_macro = len(macro_var_names)
final_data = handle.replace_value(final_data,['regionwb'],np.NaN,'Non_OECD_Rich')
i = 0
for name,group in grouped:
    for col in macro_var_names.ix[:,0]:
        try:
            final_data[col] = grouped.transform(lambda x: x.fillna(x.mean()))
        except:
            try:
                final_data[col] = grouped.transform(lambda x: x.fillna(x.mode()[0]))
            except:
                final_data[col] = grouped.transform(lambda x: x.fillna(0))
        print(str(i)+' out of ' + str(n_groups*n_macro))
        i += 1
    

clusters = handle.readcsv('sample_country_clusters.csv',0)
clusters.columns = ['economy','cluster']
final_data = pd.merge(final_data,clusters,'left','economy')

final_data.to_csv('macro_data.csv')
