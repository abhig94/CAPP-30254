from handleData import *
from pipe import *
filename = 'micro_world.csv'
data = pd.read_csv(filename)
data = replace_value(data,['regionwb'],np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'
bad_inds = [0,4,5]
x_names = [i for j, i in enumerate(list(data)) if j not in bad_inds]
x_names.remove(y_name)
y = data[y_name]
x = data[x_names]
x, bins = discretize(x, ['pop_adult','age'])
new_names = list(x)
#new_names.remove('pop_adult')
#new_names.remove('age')
x = x[new_names]
x = create_dummies(x, new_names)
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
os.chdir('Output')
x.to_csv("x.csv")
y.to_csv("y.csv")