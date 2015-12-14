import numpy as np
import os
import ujson

ALL = 882512
TRAN_SET_N = 300000
TEST_SET_N = 100000
SOUND_MAP_LEN = 53
SOUND_MAP = [u"sil", u"a0", u"a1", u"a2", u"a4", u"b", u"b'", u"c", u"ch", u"d", u"d'", u"e0", u"f",\
			u"f'", u"g", u"g'", u"h", u"h'", u"i0", u"i1", u"i4", u"j", u"k", u"k'", u"l", u"l'", u"m",\
			u"m'", u"n", u"n'", u"o0", u"o1", u"p", u"p'", u"r", u"r'", u"s", u"s'", u"sc", u"sh", u"t",\
			u"t'", u"u0", u"u1", u"u4", u"v", u"v'", u"y0", u"y1", u"y4", u"z", u"z'", u"zh"]
def loadMel(file="train_mel.db"):
	mels = []
	with open(file) as data_file:  
		temp = ujson.load(data_file)
		mels = np.array(temp[:TEST_SET_N+TRAN_SET_N + 1])
	return mels[:TRAN_SET_N], mels[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]	
def loadRes(file="train_res.db"):
	res = []
	with open(file) as data_file:    
		temp = ujson.load(data_file)
		MAP = []	
		for i in range(TEST_SET_N + TRAN_SET_N + 1):
			MAP.append(transform(temp[i]))
		res = np.array(MAP, dtype=np.int8)
	return res[:TRAN_SET_N], res[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]		
def transform(el):
	temp = np.array([0] * SOUND_MAP_LEN, dtype=np.int8)
	i0 = 0
	for i in range(SOUND_MAP_LEN):
		if SOUND_MAP[i] == el:
			i0 = i
	temp[i0] = 1
	return temp
