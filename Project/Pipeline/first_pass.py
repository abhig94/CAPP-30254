from handleData import *
from pipe import *
filename = 'micro_world.csv'
data = readcsv(filename)
data = replace_value(data,'regionwb',np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'
x_names = list(data).remove('q24','wgt','wpi_id','economy')
new_x =create_dummies(data,x_names)
y = data[y_name]
x = data[x_names]

data, bins = discretize(data, ['pop_adult','age'])
data = create_dummies(data, x_names)