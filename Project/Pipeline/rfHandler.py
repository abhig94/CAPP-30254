import csv

def readFile(fName):
	with open(fName, "r") as fout:
		try:
			reader = csv.reader(fout)
		except:
			raise Exception('Can\'t read in file.')
		return list(reader)

def makeTmpFile(fName):
	tmp = readFile(fName)
	header = tmp[0]
	modelIndx = header.index('classifier')
	results = []
	fLen = len(tmp)
	for j in range(0, fLen):
		if 'RandomForestClassifier' not in tmp[j][modelIndx]:
			results.append(tmp[j])

	with open('tmp.csv', "w") as fout:
		writer = csv.writer(fout)
		for f in results:
			writer.writerow(f)
		fout.close()

def recombineData(fName):
	t0 = readFile(fName)
	t1 = readFile('tmp.csv')

	del t1[0]
	results = t0 + t1

	with open(fName, "w") as fout:
		writer = csv.writer(fout)
		for f in results:
			writer.writerow(f)
		fout.close()

