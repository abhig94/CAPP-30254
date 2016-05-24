from handleData import *
from pipe import *


############################################################
''' Model Defintions '''
simple_modelSVC = {'model': svm.LinearSVC}
simple_modelLR = {'model': LogisticRegression}
simple_modelRF  = {'model': RandomForestClassifier}
simple_modelNB  = {'model': GaussianNB}
simple_modelDT  = {'model': DecisionTreeClassifier, 'max_depth': [50], 'max_features': ['sqrt'],
			'min_samples_split': [50]}
simple_modelDTR = {'model': DecisionTreeRegressor}

modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [50, 100], #1, 5, 10,20,
            'max_features': ['sqrt','log2'],'min_samples_split': [2, 5, 10, 20, 50]}


modelList = [modelDT, simple_modelRF, simple_modelDTR, simple_modelNB, simple_modelLR, simple_modelSVC]
modelList2 = [simple_modelDT, simple_modelLR, simple_modelDTR]
###########################################################

os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
os.chdir('Output')
x = readcsv('x_nodiscrete.csv',index_col = 0)
y = readcsv('y.csv',index_col = 0)
weights = readcsv('weights.csv',index_col = 0)
weights = weights['wgt']
to_discretize =  ['pop_adult','age']
results = clf_loop_reloaded(x,y,5,modelList,to_discretize,10,weights)#pipeLine(y,x, modelList, 5)
write_results_to_file('modelList_results.csv', results)
weight_results = clf_loop_reloaded(x,y,5,modelList,to_discretize,10,weights,True)
write_results_to_file('modelList_weight_results.csv', weight_results)