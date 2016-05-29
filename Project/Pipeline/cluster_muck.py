from handleData import *
from pipe import *
from sklearn.cluster import KMeansm
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

macro_surv = pd.read_excel('macro_vars.xlsx')
macro_surv_nums = macro_surv.convert_objects(convert_numeric=True)
macro_surv_filled = replace_value(macro_surv_nums,list(macro_surv_nums),np.NaN,0)
var_names = list(macro_surv_filled)
var_names.remove('economycode')
var_names.remove('economy')
macro_countries = macro_surv_filled['economy']
var_nums_only = macro_surv_filled[var_names]
macro_clusters = pd.DataFrame(test.fit_predict(var_nums_only))
macro_countries_clusters = pd.concat([macro_countries, macro_clusters],axis=1)
macro_cluster_sort = macro_countries_clusters.sort_values(0)

os.chdir('Output')
cluster_sort.to_csv("sample_country_clusters.csv")
macro_cluster_sort.to_csv("sample_macro_country_clusters.csv")