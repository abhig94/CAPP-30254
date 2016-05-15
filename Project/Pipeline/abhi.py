# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:34:36 2016

@author: Abhi
"""

import pipe as pipe
import handleData as handle
import pandas as pd
import  os



os.chdir('..')
os.chdir('Data')



data = pd.read_csv('micro_world.csv',engine='python')