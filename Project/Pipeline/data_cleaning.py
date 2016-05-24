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



data = handle.readcsv('micro_world.csv')
"""
subq_regex = re.compile("q[0-9]{1,2}[a-z]+")
subquestion_cols = [m.group(0) for l in list(data.columns) for m in [subq_regex.search(l)] if m]
q_regex = re.compile("q[0-9]{1,2}[a-z]+")
question_cols = [m.group(0) for l in list(data.columns) for m in [q_regex.search(l)] if m]



data = handle.fill_missing(data,list(data.columns),replacement=0)
"""

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
results['economy'] = results['economy_new']
results = results.drop(['Unnamed: 3','economy_x','economy_y','economy_new'],1)
results = pd.merge(results,inequality,left_on='economy',right_on='economy_new',how='left')
results['economy'] = results['economy_new']
results = results.drop(['economy_x','economy_y','economy_new'],1)
results = results.replace('..',np.NaN)
results.to_excel('macro_vars.xlsx')


results = pd.merge(country_codes,survey,left_on='economy',right_on='economy_new',how='left')
results['economy'] = results['economy_new']
results = results.drop(['economy_x','economy_y','economy_new'],1)
results = results.replace('..',np.NaN)
results.to_excel('agg_survey_vars.xlsx')
