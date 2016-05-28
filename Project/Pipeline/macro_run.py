from handleData import *
from pipe import *


############################################################
''' Model Defintions '''
cpus = mp.cpu_count()
cores = max(cpus/2-1,1)
depth = [10, 20, 50]
simple_modelSVC = {'model': svm.LinearSVC}
simple_modelLR = {'model': LogisticRegression}
simple_modelRF  = {'model': RandomForestClassifier}
simple_modelNB  = {'model': GaussianNB}
simple_modelDT  = {'model': DecisionTreeClassifier, 'max_depth': [50], 'max_features': ['sqrt'],
			'min_samples_split': [50]}
simple_modelDTR = {'model': DecisionTreeRegressor}

modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [50, 100], #1, 5, 10,20,
            'max_features': ['sqrt','log2'],'min_samples_split': [2, 5, 10, 20, 50]}
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [20, 50], #min sample split also had 2, 5, 10
            'bootstrap': [True], 'n_jobs':[3]} #bootstrap also had False
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [25, 50]}#, 200]}
modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth,
            'bootstrap': [True, False], 'n_jobs':[cores]}
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1],#, 5, 10, 25],
          'class_weight': ['balanced', None], 'n_jobs' : [cores],
          'tol' : [1e-5, 1e-3, 1], 'penalty': ['l1', 'l2']}

modelDT = [simple_modelDT]
modelList = [modelDT, modelRF, modelAB, modelET, simple_modelDTR, simple_modelNB, modelLR, simple_modelSVC]
modelList2 = [simple_modelDT, simple_modelLR, simple_modelDTR]
###########################################################

os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
os.chdir('Output')

x = readcsv('x_macro_data.csv',index_col = 0)
y = readcsv('y.csv',index_col = 0)
weights = readcsv('weights.csv',index_col = 0)
weights = weights['wgt']
to_discretize =  ['pop_adult','age']
results = clf_loop_reloaded(x,y,5,modelDT,to_discretize,10,weights)#pipeLine(y,x, modelList, 5)
write_results_to_file('modelList_results.csv', results)
#weight_results = clf_loop_reloaded(x,y,5,modelDT,to_discretize,10,weights,True)
#write_results_to_file('modelList_weight_results.csv', weight_results)
'''
# doesn't use the model weight results
all_results = pd.read_csv('modelList_results.csv')
criteria = criteriaHeader
if 'Function called' in criteria:
    criteria.remove('Function called')
if 'classifier' in criteria:
    criteria.remove('classifier')
all_results = clean_results(all_results,criteria)
best_clfs = best_by_each_metric(all_results)
best_clfs.to_csv('best_clfs.csv')

comparison = compare_clf_across_metric(all_results,'AUC')
comparison.to_csv('comparison_of_clfs.csv')


# using the model weights
all_results = pd.read_csv('modelList_weight_results.csv')
all_results = clean_results(all_results,criteria)
# strip LR runs
not_LR_rows =  [not 'LogisticRegression' in x for x in all_results.classifier]
all_results = all_results.ix[not_LR_rows,:]

best_clfs = best_by_each_metric(all_results)
best_clfs.to_csv('best_clfs_weights.csv')

comparison = compare_clf_across_metric(all_results,'AUC')
comparison.to_csv('comparison_of_clfs_weights.csv')
