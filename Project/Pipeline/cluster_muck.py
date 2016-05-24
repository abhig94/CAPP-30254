from handleData import *
from pipe import *
from sklearn.cluster import KMeans
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')

aggr_surv = pd.read_excel('agg_survey_vars.xlsx')
aggr_surv_filled = replace_value(aggr_surv,list(aggr_surv),np.NaN,0)
var_names = list(aggr_surv_filled)
var_names.remove('economycode')
var_names.remove('economy')
countries = aggr_surv_filled['economy']
aggr_nums_only = aggr_surv_filled[var_names]
test = KMeans(n_init = 200)
clusters = pd.DataFrame(test.fit_predict(aggr_nums_only))
countries_clusters = pd.concat([countries, clusters],axis=1)
cluster_sort = countries_clusters.sort_values(0)
os.chdir('Output')
cluster_sort.to_csv("sample_country_clusters.csv")