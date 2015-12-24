import time
startt = time.time()
import numpy as np
import ujson
import load
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
print "test"

DIM = 100

mel = load.loadMel()[0]
res_class = load.loadClass()[0]
print "loaded"
# print res[0]
# def myfunc(a):
# 	print a
# 	return a.tolist().index(1)

def save_to_file(X, filename='afterLDA'):
	with open(filename + str(DIM) + ".db", 'w') as f:
		ujson.dump(X.tolist(), f)
		
# vfunc = np.vectorize(myfunc)
# res_class = vfunc(res)
clf = LinearDiscriminantAnalysis(n_components=DIM)
print "train"
clf.fit(mel, res_class)
print "trained"
print clf.predict(mel[:10])
pred = clf.predict(mel)
print res_class[:10]
print np.mean(pred == res_class)
save_to_file(clf.transform(mel))
endt = time.time()
print endt - startt
