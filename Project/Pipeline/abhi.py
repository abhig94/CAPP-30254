# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:34:36 2016

@author: Abhi
"""

import pipe as pipe
import handleData as handle
import pandas as pd
import  os
import re


os.chdir('..')
os.chdir('Data')



data = handle.readcsv('micro_world.csv')

subq_regex = re.compile("q[0-9]{1,2}[a-z]+")
subquestion_cols = [m.group(0) for l in list(data.columns) for m in [subq_regex.search(l)] if m]
q_regex = re.compile("q[0-9]{1,2}[a-z]+")
question_cols = [m.group(0) for l in list(data.columns) for m in [q_regex.search(l)] if m]



data = handle.fill_missing(data,list(data.columns),replacement=0)