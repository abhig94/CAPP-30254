from handleData import *
from pipe import *
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
filename = 'macro_data.csv'
data = readcsv(filename, index_col = 0)
data = replace_value(data,['regionwb'],np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
weights = data['wgt']
y_name = 'q24'
bad_inds = [0,4,5,48]
#index 0 = economy
#index 4 = random id
#index 5 = weighting 
#index 48 = fiollow-up on target
x_names = [i for j, i in enumerate(list(data)) if j not in bad_inds]
y = data[y_name]
x = data[x_names]
new_names = list(x)
new_names.remove('pop_adult')
new_names.remove('age')
new_names.remove(y_name)

macro_var_names = readcsv('macro_var_names.csv')
macro_var_names_list = macro_var_names.values.tolist()
macro_names = [val for sublist in macro_var_names_list for val in sublist]

x = create_dummies(x, [n for n in new_names if n not in macro_names])
grouped = x.groupby('q24')
neg_data = grouped.get_group(0)
pos_data = grouped.get_group(1)

os.chdir('Output')
os.chdir('Graphics')

x = x.drop(y_name,axis=1)
important_var_names = identify_important_features(x,y, 20, True,'macro_pass')
explore_data(neg_data, important_var_names, save_toggle=True,  file_prefix = 'negative_data')
explore_data(pos_data, important_var_names, save_toggle=True, file_prefix = 'positive_data')
os.chdir('..')

weights.to_csv("weights.csv",header=True)
x.to_csv("x_macro_data.csv")
y.to_csv("y.csv",header = True)