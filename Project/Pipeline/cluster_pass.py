from handleData import *
from pipe import *

#Feature Importance#
#Feature Importance#
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
filename = 'micro_world.csv'
data = readcsv(filename)
data = replace_value(data,['regionwb'],np.NaN,'Non_OECD_Rich')
data = replace_value(data, list(data), np.NaN, 0)
data = get_q_24(data)
y_name = 'q24'
<<<<<<< HEAD
clusters = readcsv('sample_country_clusters.csv',0)
=======
clusters = handle.readcsv('sample_country_clusters.csv',0)
>>>>>>> origin/master
clusters.columns = ['economy','cluster']
data = pd.merge(data,clusters,'left','economy')
bad_inds = [0,2,4,5,48]


#index 0 = economy
#index 4 = random id
#index 5 = weighting 
#index 48 = follow-up on target
x_names = [i for j, i in enumerate(list(data)) if j not in bad_inds]
x_names.remove(y_name)
y = data[y_name]
x = data[x_names]
<<<<<<< HEAD
=======
weights = x['wgt']
>>>>>>> origin/master
new_names = list(x)
new_names.remove('pop_adult')
new_names.remove('age')
x = create_dummies(x, new_names)

#identify_important_features(x,y, 10, True,'first_pass')

#weights.to_csv("weights.csv",header=True)
<<<<<<< HEAD
os.chdir('Output')
=======
>>>>>>> origin/master
x.to_csv("x_clusterpass.csv")
#y.to_csv("y.csv",header = True)
