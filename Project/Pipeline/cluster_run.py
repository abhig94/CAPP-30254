from handleData import *
from pipe import *
from rfHandler import *

import sys
import pandas as pd
import numpy as np


############################################################
''' Model Defintions '''
cpus = mp.cpu_count()
cores = max(cpus-1,1)
depth = [10, 50, 100]
###########################################################
simple_modelSVC = {'model': svm.LinearSVC}
simple_modelLR = {'model': LogisticRegression}
simple_modelRF  = {'model': RandomForestClassifier}
simple_modelDT  = {'model': DecisionTreeClassifier, 'max_depth': [50], 'max_features': ['sqrt'],
      'min_samples_split': [50]}
simple_modelDTR = {'model': DecisionTreeRegressor}

########################################################

modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': depth, #1, 5, 10,20,
            'max_features': ['sqrt','log2'],'min_samples_split': [2, 10, 50]}
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [2,10, 50], #min sample split also had 2, 5, 10
            'bootstrap': [True], 'n_jobs':[cores], 'warm_start':[False]} #bootstrap also had False
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [25, 50]}#, 200]}
modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth,
            'bootstrap': [True, False], 'n_jobs':[cores]}
modelLR = {'model': LogisticRegression, 'solver': ['sag'], 'C' : [.01, .1, .5, 1],#, 5, 10, 25],
          'class_weight': ['balanced', None], 'n_jobs' : [cores],
          'tol' : [1e-5, 1e-3, 1], 'penalty': ['l2']}
modelNB  = {'model': GaussianNB}     
#modelDTR = {'model': DecisionTreeRegressor, 'max_features': ['sqrt', 'log2'], 'max_depth': depth,
#            'min_samples_split': [2, 10, 50]}     
modelgoodRF = {'model': RandomForestClassifier, 'n_estimators': [100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt'], 'max_depth': [100], 'min_samples_split': [50], #min sample split also had 2, 5, 10
            'bootstrap': [True], 'n_jobs':[cores], 'warm_start':[False]}
modelgoodclusterLR = {{'model': LogisticRegression, 'solver': ['sag'], 'C' : [.01],#, 5, 10, 25],
          'class_weight': [None], 'n_jobs' : [cores],
          'tol' : [1e-5], 'penalty': ['l2']}}
modelList = [modelDT, modelRF, modelAB, modelLR, modelNB]
cluster_test = [modelgoodRF, modelgoodclusterLR]
#modelList = [modelDT, modelRF, modelAB, modelET, simple_modelDTR, simple_modelNB, modelLR, simple_modelSVC]
#modelList2 = [simple_modelDT, simple_modelLR, simple_modelDTR]
###########################################################
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('Data')
os.chdir('Output')

# macro?, ensemble, or blind?

if __name__ == '__main__':
  n_arg = len(sys.argv)

  if n_arg != 4:
    raise Exception('Entered wrong number of arguments. Must enter 3 arguments.')

  try:
    macro = sys.argv[1]
    ensemble = sys.argv[2]
    blind = sys.argv[3]
  except:
    raise Exception('Entered wrong number of arguments. I require 3')

  rf = raw_input("Enter 1 now to do the random forest thang: ")
  RF = False

  if rf == 1 or rf == '1':
    RF = True

  name = ''
  mac = macro == '1'
  ens = ensemble == '1'
  b = blind == '1'
  name = ''

  if mac and ens and b:
    name = 'macro_ensemble_blind'
  elif mac and ens and not b:
    name = 'macro_ensemble'
  elif mac and not ens and b:
    name = 'macro_blind'
  elif mac and not ens and not b:
    name = 'macro_only'
  elif not mac and ens and b:
    raise Exception('Must enter macro as true if running ensemble') 
  elif not mac and not ens and not b:
    name = 'original_baseline'
  elif not mac and ens and not b:
    name = 'original_ensemble'

  if RF:
    modelList = [modelRF]



  name += '.csv'

  if macro == '1' or b:
    x = readcsv('x_macro_data.csv', index_col = 0)
  else:
    x = readcsv('x_clusterpass.csv', index_col = 0)

  y = readcsv('y.csv', index_col = 0)
  weights = readcsv('weights.csv',index_col = 0)
  weights = weights['wgt']
  to_discretize =  ['pop_adult','age']
  create_dummies(x, 'cluster')

  check = list(x)
  if 'cluster' in check:
    x.drop('cluster',axis=1)
  
  questions = ['q2','q3','q4','q5','q6','q8a','q8b','q8c','q8d','q8e',
           'q8f','q8g','q8h','q8i','q9', 'q10','q11','q12','q13',
           'q14','q16','q17a','q17b','q17c','q18a','q18b','q20','q21a',
           'q21b','q21c','q21d','q22a','q22b','q22c','q24','q25','q26',
           'q27a','q27b','q27c','q27d','q28','q29a','q29b','q29c','q29d',
           'q30','q31a','q31b','q31c','q32','q33a','q33b','q33c','q34',
           'q35','q36a','q36bc','q36d','q37','q38','q39','q40a','q40bc',
           'q40d','q41','q42','q43','q44a','q44b','q44c']
  tester = lambda n: np.any([q in n for q in questions])
  drops = [c for c in x.columns if tester(c)]

  if not ens:
    if not mac:
      if not b:
        results = clf_loop_reloaded(x,y,5,modelList,to_discretize, 10, weights, True, True, False)
      else:
        x = x.drop(drops, 1)
        results = clf_loop_reloaded(x,y,5,modelList,to_discretize, 10, weights, True, True, False)
    else:
      results = clf_loop_reloaded(x,y,5,modelList,to_discretize, 10, weights, True, True, True)

  else:
    if not b and not mac:
      results = clf_loop_revolutions(x, y, 5, modelList, to_discretize, 10, weights, True, False, 'cluster')
    elif not b and mac:
      results = clf_loop_revolutions(x, y, 5, modelList, to_discretize, 10, weights, True, True 'cluster')
    else: #blind and macro
      x = x.drop(drops, 1)
      results = clf_loop_revolutions(x, y, 5, modelList, to_discretize, 10, weights, True, True, 'cluster')

  os.chdir('Results')
  if RF:
    makeTmpFile(name)

  write_results_to_file(name, results)

  if RF:
    recombineData(name)

  all_results = pd.read_csv(name)
  criteria = criteriaHeader
  if 'Function called' in criteria:
      criteria.remove('Function called')
  if 'classifier' in criteria:
      criteria.remove('classifier')
  all_results = clean_results(all_results,criteria)
  best_clfs = best_by_each_metric(all_results)

  best_name = 'best_clfs_' + name
  best_clfs.to_csv(best_name)
  comparison_name = 'comparison_' + name

  comparison = compare_clf_across_metric(all_results,'AUC')
  comparison.to_csv(comparison_name)

