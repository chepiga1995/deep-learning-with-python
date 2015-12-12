import numpy as np
import os
import json

ALL = 882512
TRAN_SET_N = 100000
TEST_SET_N = 10000

SOUND_MAP = [u"sil", u"a0", u"a1", u"a2", u"a4", u"b", u"b'", u"c", u"ch", u"d", u"d'", u"e0", u"f",\
			u"f'", u"g", u"g'", u"h", u"h'", u"i0", u"i1", u"i4", u"j", u"k", u"k'", u"l", u"l'", u"m",\
			u"m'", u"n", u"n'", u"o0", u"o1", u"p", u"p'", u"r", u"r'", u"s", u"s'", u"sc", u"sh", u"t",\
			u"t'", u"u0", u"u1", u"u4", u"v", u"v'", u"y0", u"y1", u"y4", u"z", u"z'", u"zh"]
def loadMel(file="train_mel.db"):
	mels = []
	with open(file) as data_file:    
		temp = json.load(data_file)
		mels = np.array(temp, dtype=np.float16)
	return mels[:TRAN_SET_N], mels[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]	
def loadRes(file="train_res.db"):
	res = []
	with open(file) as data_file:    
		temp = json.load(data_file)
		MAP = []	
		for i in range(TEST_SET_N+TRAN_SET_N):
			MAP.append(transform(temp[i], temp[(i - 1) if i > 0 else 0]))
		res = np.array(MAP, dtype=np.uint8)
	return res[:TRAN_SET_N], res[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]		
def transform(el, prev):
	temp = [0]*len(SOUND_MAP)
	i0 = 0
	j0 = 0
	for i in range(len(SOUND_MAP)):
		if SOUND_MAP[i] == el:
			i0 = i
		if SOUND_MAP[i] == prev:
			j0 = i	
	temp[i0]		
	try:
		temp.index(1)
	except Exception, e:
		temp[0] = 1
	return temp
TRAIN_MEL, TEST_MEL = loadMel()
TRAIN_RES, TEST_RES = loadRes()
print len(TRAIN_MEL), len(TEST_MEL)