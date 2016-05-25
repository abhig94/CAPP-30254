from handleData import *
from pipe import *

"""
#Feature Importance#
#Feature Importance#
filename = 'micro_world.csv'
data = readcsv(filename)
data = replace_value(data,['regionwb'],np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'
bad_inds = [0,4,5,48]
#index 0 = economy
#index 4 = random id
#index 5 = weighting 
#index 48 = follow-up on target
x_names = [i for j, i in enumerate(list(data)) if j not in bad_inds]
x_names.remove(y_name)
y = data[y_name]
x = data[x_names]
weights = x['wgt']
new_names = list(x)
new_names.remove('pop_adult')
new_names.remove('age')
x = create_dummies(x, new_names)
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
os.chdir('Output')

identify_important_features(x,y, 10, True,'first_pass')

weights.to_csv("weights.csv",header=True)
x.to_csv("x_nodiscrete.csv")
y.to_csv("y.csv",header = True)
"""

os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
filename = 'final_data.csv'
data = readcsv(filename)
data = readcsv(filename)
data = replace_value(data,['regionwb'],np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'

bad_inds = [0,4,5,48]
#index 0 = economy
#index 4 = random id
#index 5 = weighting 
#index 48 = follow-up on target
x_names = [i for j, i in enumerate(list(data)) if j not in bad_inds]
x_names.remove(y_name)
weights = x['wgt']
y = data[y_name]
x = data[x_names]
new_names = list(x)
new_names.remove('pop_adult')
new_names.remove('age')

macro_var_names = readcsv('macro_var_names.csv')
x = create_dummies(x, [n  for n in new_names if n not in macro_var_names])

identify_important_features(x,y, 10, True,'first_pass')

weights.to_csv("weights.csv",header=True)
x.to_csv("x_final_data.csv")
y.to_csv("y.csv",header = True)

