# Pure awesomeness right here

'''
@author: Mike
'''

'''
def get_q_24(data):
	df = data[data.q24 != 5]
	df = data[data.q24 != 6]

	df = replace_value(df, ['q24'], 1, 0)
	df = replace_value(df, ['q24'], 2, 0)
	df = replace_value(df, ['q24'], 3, 1)
	return replace_value(df, ['q24'], 4, 1)
'''

'''
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
'''

import os
from os import listdir
from os.path import isfile, join
import subprocess

def root_path():
    return os.path.abspath(os.sep)


def getAllFiles(path):
	files = [f for f in listdir(path) if isfile(join(path, f))]
	return files


def save_to_dropbox(df):
	path = os.getcwd()

	fOne = getAllFiles(path)
	if 'Users' in fOne:
		path += 'Users'


