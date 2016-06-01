import matplotlib
from matplotlib import pyplot as plt
from rfHandler import *
import os

def getNum(s):
	if '(' in s:
		num = s[0:s.index('(')]
		return float(num)
	else:
		return float(s)
	return None

def getName(dRow, mIndx):
	namey = dRow[mIndx]
	namey = namey[0:namey.index('(')]
	return namey

def getSpecificThresh(s):
	tIndx = s.index('.')
	return float(s[tIndx:])

def getThreshes(header, indxs):
	iLen = len(indxs)
	thresholds = [0]*iLen

	for k in range(0, iLen):
		thresholds[k] = getSpecificThresh(header[indxs[k]])
	return thresholds


def arrangeData(fName, desiredStat):
	data = readFile(fName)
	dLen = len(data)
	modelIndx = data[0].index('classifier')
	indxs = []

	i = 0
	for x in data[0]:
		if desiredStat in x:
			indxs.append(i)
		i +=1

	threshs = getThreshes(data[0], indxs)

	names= []

	for j in range(1, dLen):
		namey = data[j][modelIndx]
		namey = namey[0:namey.index('(')]
		if namey not in names:
			names.append(namey)

	d = {}
	iLen = len(indxs)
	for n in names:
		d[n] = [0]*iLen

	for j in range(1, dLen):
		name = getName(data[j], modelIndx)
		row = data[j]
		for i in range(0, iLen):
			number = getNum(row[indxs[i]])
			if number > d[name][i]:
				d[name][i] = number

	return (d, threshs)

def plotDat(d, threshes, mainTitle, saveName):
	colores = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
	i = 0
	for key in d.keys():
		if i > 6:
			break
		plt.plot(threshes, d[key], colores[i], label = key)
		i +=1
	plt.legend(loc='lower left')

	plt.title(mainTitle)
	plt.xlabel('Threshold')
	plt.ylabel('Level')
	plt.savefig(saveName)
	plt.close()

def getSaveName(s, extra):
	indx = s.index('.')
	s = s[0:indx] + extra + '.png'
	return s 

def arrangeDataII(fName):
	data = readFile(fName)
	dLen = len(data)
	modelIndx = data[0].index('classifier')
	aucIndx = data[0].index('AUC')
	names = []
	for j in range(1, dLen):
		namey = data[j][modelIndx]
		namey = namey[0:namey.index('(')]
		if namey not in names:
			names.append(namey)

	d = {}
	for n in names:
		d[n] = [0]

	for j in range(1, dLen):
		name = getName(data[j], modelIndx)
		row = data[j]
		number = getNum(row[aucIndx])
		if number > d[name][0]:
			d[name][0] = number

	return d

def getRunType(s):
	fIndx = s.index('_')
	eIndx = s.rindex('.')
	return s[fIndx+1:eIndx]

def makeBarChart(aucs, runs, modelName):
	width = .35
	ind = np.arange(len(aucs)) 

	fig, ax = plt.subplots()
	ax.set_ylim([min(aucs)-.01,max(aucs) + .01])
	rects1 = ax.bar(ind, aucs, width, color='b')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(runs)
	ax.set_ylabel('AUC')
	ax.set_xlabel('Run')
	ax.set_title(modelName)

	for item in (ax.get_xticklabels()):
		item.set_fontsize(8)
	plt.savefig(modelName+'.png')

def doThingsII(fList):
	l = []
	cnt = 0
	for f in fList:
		if 'comparison' in f:
			l.append({getRunType(f):arrangeDataII(f)})
			cnt += 1

	colNames = [x.keys()[0] for x in l]

	tKey = l[0].keys()[0]

	classes = l[0][tKey].keys()

	points = [[]]*len(classes)

	for item in l:
		dTemp = item.values()[0]
		for key in dTemp.keys():
			indx = classes.index(key)
			points[indx] = points[indx] + dTemp[key]

	k = 0
	for p in points:
		if len(p) == cnt:
			makeBarChart(p, colNames, classes[k])
		k +=1


def doThings(fList):
	for f in fList:
		if 'comparison' in f:
			sNameP = getSaveName(f, '_precision')
			sNameR = getSaveName(f, '_recall')
			d, threshes = arrangeData(f, 'Precision')
			plotDat(d, threshes, 'Precision', sNameP)
			d, threshes = arrangeData(f, 'Recall')
			plotDat(d, threshes, 'Recall', sNameR)

if __name__ == '__main__'
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')
	os.chdir('Data')
	os.chdir('Output')
	os.chdir('Results')
	fList = os.listdir(os.getcwd())
	doThings(fList)
	doThingsII(fList)	














