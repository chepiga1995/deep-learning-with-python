# -*- encoding: koi8-r -*-
import scipy.io.wavfile as wav
import numpy as np
from mel_from_mlf import mfcc
import os

mas = ['а','б','в','г','д','е','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']

DIRECTORY_PATH = '/home/ura/Documents/Deep-learning/'
WINLEN = 0.025
WINSTEP = 0.01
def main_decode(num):
	if ord(num) >= 224:
		return mas[ord(num) - 224]
	elif ord(num) >= 128:
		return '-'
	else:	 
		return num
def translate(filename):		
	with open(filename, 'r') as f:
		string = f.read()
		arr_res = map(main_decode, string)
		res = ''.join(arr_res)
		return res
def procces_file(filename):
	rate, arr_frec = wav.read(os.path.join(DIRECTORY_PATH, filename + '.wav'))
	arr_mfcc = mfcc(arr_frec, samplerate=rate, winlen=WINLEN, winstep=WINSTEP)
	return arr_mfcc, translate(filename + ".txt")


