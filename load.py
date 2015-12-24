import numpy as np
import os
import ujson

ALL = 882512
TRAN_SET_N = 600000
TEST_SET_N = 250000
SOUND_MAP_LEN = 55
LETER_MAP_LEN = 28
PRUG = [u"sil", u"midsp", u"bad_sp", u"b", u"b'", u"c", u"ch", u"d", u"d'", u"f",\
			u"f'", u"g", u"g'", u"h", u"h'", u"j", u"k", u"k'", u"l", u"l'", u"m",\
			u"m'", u"n", u"n'", u"p", u"p'", u"r", u"r'", u"s", u"s'", u"sc", u"sh", u"t",\
			u"t'", u"u0", u"u1", u"u4", u"v", u"v'", u"z", u"z'", u"zh"]
SOUND_MAP = [u"sil", u"midsp", u"bad_sp", u"a0", u"a1", u"a2", u"a4", u"b", u"b'", u"c", u"ch", u"d", u"d'", u"e0", u"f",\
			u"f'", u"g", u"g'", u"h", u"h'", u"i0", u"i1", u"i4", u"j", u"k", u"k'", u"l", u"l'", u"m",\
			u"m'", u"n", u"n'", u"o0", u"o1", u"p", u"p'", u"r", u"r'", u"s", u"s'", u"sc", u"sh", u"t",\
			u"t'", u"u0", u"u1", u"u4", u"v", u"v'", u"y0", u"y1", u"y4", u"z", u"z'", u"zh"]
LETER_MAP = [u"sil", u"a", u"b", u"c", u"ch", u"d", u"e", u"f",\
			u"g", u"h", u"i", u"j", u"k", u"l", u"m",\
			u"n", u"o", u"p", u"r", u"s", u"sc", u"sh", u"t",\
			u"u", u"v", u"y", u"z", u"zh"]
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
		for i in range(TEST_SET_N + TRAN_SET_N):
			MAP.append(transform(temp[i]))
		res = np.array(MAP, dtype=np.int16)
	return res[:TRAN_SET_N], res[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]	
def retNum(string):
	return SOUND_MAP.index(string)
def loadClass(file="train_res.db"):
	res = []
	with open(file) as data_file:    
		temp = ujson.load(data_file)
		MAP = [0]	
		for i in range(1, TEST_SET_N + TRAN_SET_N):
			MAP.append(isPrug(temp[i - 1]) * 55 + retNum(temp[i]))
		res = np.array(MAP, dtype=np.int8)
	return res[:TRAN_SET_N], res[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]		
def loadLeter(file="train_res.db"):
	res = []
	with open(file) as data_file:    
		temp = ujson.load(data_file)
		MAP = []	
		for i in range(TEST_SET_N + TRAN_SET_N):
			MAP.append(leter(temp[i]))
		res = np.array(MAP, dtype=np.int8)
	return res[:TRAN_SET_N], res[TRAN_SET_N:TEST_SET_N+TRAN_SET_N]	

def leter(el):
	temp = np.array([0] * LETER_MAP_LEN, dtype=np.int8)
	el = el.replace("'", '').replace("0", '').replace("1", '').replace("2", '').replace("3", '').replace("4", '')
	try:
		temp[LETER_MAP.index(el)] = 1
	except Exception, e:
		temp[0] = 1
	return temp
def transform(el):
	temp = np.array([0] * SOUND_MAP_LEN, dtype=np.int8)
	i0 = 0
	for i in range(SOUND_MAP_LEN):
		if SOUND_MAP[i] == el:
			i0 = i
	temp[i0] = 1
	return temp	
def isPrug(string):
	try:
		PRUG.index(string)
	except Exception, e:
		return 0
	return 1	
	