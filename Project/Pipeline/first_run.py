############################################################
''' Model Defintions '''
simple_modelSVC = {'model': svm.LinearSVC}
simple_modelLR = {'model': LogisticRegression}
simple_modelRF  = {'model': RandomForestClassifier}
simple_modelNB  = {'model': GaussianNB}
simple_modelDT  = {'model': DecisionTreeClassifier, 'max_depth': [50], 'max_features': ['sqrt']
			'min_samples_split': [50]}
simple_modelDTR = {'model': DecisionTreeRegressor}

modelList = [simple_modelDT]

###########################################################    
x = pd.read_csv('x.csv',index_col = 0)
y = pd.read_csv('y.csv',index_col = 0)
results = pipeLine(y,x modelList, 5)
write_results_to_file(first_results, results)