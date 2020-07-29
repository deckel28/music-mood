import librosa
import numpy as np
import os
from os.path import isfile, join
from os import listdir
import pickle
import matplotlib.pyplot as plt
import librosa.display
import sys

sys.path.append("./../feature_extraction")
from features import GetSpectrumFeatures

sys.path.append("./../")
from configs import config

wav_files = [config.wav_files_dir + f for f in listdir(config.wav_files_dir) if isfile(join(config.wav_files_dir, f))]
wav_files.sort()

feats = np.zeros((len(wav_files), 20, 211))


def divide_chunks(l, n):
    arr = np.zeros((int(len(l)/int(n)), int(n)))
    i=0
    for j in range(0, (int(len(l)/int(n)))*n, n):
        arr[i] = l[j:j + n]
        i+=1
    print(arr.shape)
    return arr

i = 0
for wav_file in wav_files:
    name = os.path.basename(wav_file)
    print(name)

    y, sr = librosa.load(wav_file)
    shape = y.shape

    yt, index = librosa.effects.trim(y)
    print(yt.shape)
    arr = divide_chunks(yt, 110000)
    mfccarr = np.zeros((20, 1))
    for j in range(arr.shape[0]):
        get = GetSpectrumFeatures(arr[i], sr)
        mfcc1 = get.get_mfcc()
        mfccarr = np.append(mfccarr, mfcc1, axis=1)

    mfccarr = np.delete(mfccarr, 0, axis=1)
    print(mfccarr.shape)


    break
