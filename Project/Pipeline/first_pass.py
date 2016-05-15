from handleData import *
from pipe import *
filename = 'micro_world.csv'
data = readcsv(filename)
data = replace_value(data,'regionwb',np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'
x_names = list(data).remove('q24','wgt')
y = data[y_name]
x = data[x_names]
