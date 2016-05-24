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


os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')

country_codes = handle.readcsv('country_to_code.csv')
inequality = pd.read_excel('inequality_data.xlsx')
macro = pd.read_excel('macro_data.xlsx')
survey = pd.read_excel('agg_survey_data.xlsx')

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
#tester = lambda x: np.asarray([type(y) is not str for y in x])
#missing_economies = results[tester(results.economy)]
#for r in missing_economies.index:
    #results.ix[r,'economy'] = country_codes[country_codes.economycode==results.ix[r,'economycode']]['economy']  
results.to_excel('macro_vars.xlsx')

results2 = pd.merge(country_codes,survey,left_on='economy',right_on='economy_new',how='left')
results2['economy'] = results2['economy_x']
results2 = results2.drop(['economy_x','economy_y','economy_new'],1)
results2 = results2.replace('..',np.NaN)
results2.to_excel('agg_survey_vars.xlsx')


micro_world = handle.readcsv('micro_world.csv')

final_data = pd.merge(micro_world,results,on=['economycode','economy'],how='left')
final_data = pd.merge(final_data,results2,on=['economycode','economy'],how='left')

#missing_economies = final_data[final_data.economy==np.NaN]
#for r in missing_economies.index:
#    final_data.ix[r,'economy'] = country_codes[country_codes.economycode==final_data.ix[r,'economycode']]['economy']
final_data.to_csv('final_data.csv')
