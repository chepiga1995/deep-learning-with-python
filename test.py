import scipy.io.wavfile as wav
import numpy as np
from mel_from_mlf.features import mfcc

TIME_COF = 10000000.0

file_wav = '/home/ura/Documents/Deep-learning/Base_Example/Train_Set/A00/c1_m15_00000.wav'
np.set_printoptions(threshold='nan')
rate, arr = wav.read(file_wav)
COF = rate / TIME_COF
start, end = 116200000, 116500000
time = (end - start) / TIME_COF
start = int(COF*start)
end = int(COF*end)
mfcc_feat = mfcc(arr[start:end], rate, time)
print arr[start:end]
print mfcc_feat[]
