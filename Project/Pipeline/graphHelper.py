import matplotlib
from matplotlib import pyplot as plt
from rfHandler import *

def getNum(s):
	if '(' in s:
		num = s[0:s.index('(')]
		return float(num)
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










