import os
import scipy.io.wavfile as wav
import numpy as np
from features import mfcc
import re
import json 


DIRECTORY_PATH = '/home/ura/Documents/Deep-learning/Base_Example/Train_Set'
DIRECTORY_PATH_SER = '/dev/Deep-learning/Base_Example/Train_Set'
SAVE_PATH = '/home/ura/Documents/Deep-learning/'
SAVE_PATH_SER = '/dev/Deep-learning/'
TARAIN = 'train_mel.db'
TARAIN_RES = 'train_res.db'
FILE = 'train_phones.mlf'
TIME_COF = 10000000.0
WINLEN = 0.025
WINSTEP = 0.01
np.set_printoptions(threshold='nan',precision=4, suppress=True)
def mel_from_file(filename, arr_time):
	data_dir = os.path.join(DIRECTORY_PATH,'wav/')
	rate, arr_frec = wav.read(os.path.join(data_dir, filename))
	COF = rate / TIME_COF
	res = np.array([], ndmin=2)
	for start, end in arr_time:
		time = (end - start) / TIME_COF
		print time
		start = int(COF*start)
		end = int(COF*end) + 1
		if (end > len(arr_frec)):
			end = len(arr_frec)
		mfcc_feat = mfcc(arr_frec[start:end], rate, time)
		if len(res[0]) == 0:
			res = [mfcc_feat[0]]
		else:
			res = np.append(res, [mfcc_feat[0]], axis=0)	
		
	return res
def mel_seq_from_file(filename, arr_time, slices=4):
	data_dir = os.path.join(DIRECTORY_PATH_SER,'wav/')
	rate, arr_frec = wav.read(os.path.join(data_dir, filename))
	arr_mfcc = mfcc(arr_frec, samplerate=rate, winlen=WINLEN, winstep=WINSTEP)
	COF = rate / TIME_COF
	res = np.array([], ndmin=2)
	for start, end in arr_time:
		central_mel = (start + end) / (2 * TIME_COF)
		number_mel = int((central_mel - WINLEN) / WINSTEP)
		result = np.concatenate(get_neighboring_mel(arr_mfcc, number_mel, slices))
		if len(res[0]) == 0:
			res = [result]
		else:
			res = np.append(res, [result], axis=0)
	return res		

def get_neighboring_mel(arr_mfcc, number_mel, slices):
	res = []
	for pos in range(number_mel - slices, number_mel + slices + 1):
		if pos < 0:
			res.append(arr_mfcc[0])
		elif pos >= len(arr_mfcc):
			res.append(arr_mfcc[len(arr_mfcc) - 1])
		else:
			res.append(arr_mfcc[pos])
	return res				




def mel_from_mlf(filename=FILE):
	data_file = os.path.join(DIRECTORY_PATH_SER, filename)
	data = np.array([], ndmin=2)
	res = []
	with open(data_file) as openfileobject:
		wav_name = ''
		arr_time = []
		for line in openfileobject:
			mfile = re.search('^"\*/(.+)\.lab"', line)
			if(mfile): 
				wav_name = mfile.group(1) + '.wav'
			mtime = re.search('^(\d+)\s+(\d+)\s+(.+)\r$', line)
			if(mtime):
				fromt = int(mtime.group(1))
				tot = int(mtime.group(2))
				char = mtime.group(3)
				arr_time.append([fromt, tot])
				res.append(char)
			mdot = re.search('^.\r$', line)
			if(mdot):
				if len(data[0]) == 0:
					data = mel_seq_from_file(wav_name, arr_time)
				else:
					data = np.append(data, mel_seq_from_file(wav_name, arr_time), axis=0)				
				print len(data), len(res)
				arr_time = []
				
	res = np.array(res)		
	print len(data), len(res)	
	write_to_file(data, os.path.join(SAVE_PATH_SER, TARAIN))
	write_to_file(res, os.path.join(SAVE_PATH_SER, TARAIN_RES))
	
def write_to_file(arr, filename):
	with open(filename, 'w') as f:
		json.dump(arr.tolist(), f)


# print 130500000 / TIME_COF
mel_from_mlf()       
