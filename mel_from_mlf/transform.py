import os

DIRECTORY_PATH = '/home/ura/Documents/Deep-learning/Base_Example/Train_Set'
SAVE_PATH = '/home/ura/Documents/Deep-learning/'
TARAIN = 'train_mel.db'
TARAIN_RES = 'train_res.db'
FILE = 'train_phone.mlf'
TIME_COF = 10000000.0

def mel_from_file(filename, arr_time):
	data_dir = os.path.join(DIRECTORY_PATH,'wav/')
	rate, arr = wav.read(os.path.join(data_dir, filename))

	COF = rate / TIME_COF