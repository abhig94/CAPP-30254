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

country_codes = pd.read_csv('country_to_code.csv',engine='python')
inequality = pd.read_excel('inequality_data.xlsx')
macro = pd.read_excel('macro_data.xlsx')
survey = pd.read_excel('agg_survey_data.xlsx')

inequality = inequality.drop('Unnamed: 3',1)
survey = survey.drop('Unnamed: 4',1)

country_codes = country_codes.set_index('economy')
inequality = inequality.set_index('economy')
macro = macro.set_index('economy')
survey = survey.set_index('economy')

def fuzzy_match(a, b):
    left = '1' if pd.isnull(a) else a
    right = b.fillna('2')
    out = difflib.get_close_matches(left, right)
    return out[0] if out else np.NaN

indexer = lambda data: data.index.map(lambda x: fuzzy_match(x,country_codes.index))


macro.index = indexer(macro)
inequality.index = indexer(inequality)
survey.index = indexer(survey)


result = country_codes.join(macro)
result = result.join(inequality)
result = result.drop('Unnamed: 3',1)
result.to_excel('macro_vars.xlsx')

new_survey = country_codes.join(survey)
new_survey.to_excel('agg_survey_vars.xlsx')
