import h5py
import librosa
import numpy as np
import os
from os.path import isfile, join
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

### Appending the configs and Feature Extraction classes
import sys

sys.path.append("./../")
from configs import config

from getfeatures import GetBroaderFeatures
from getfeatures import GetSpectrumFeatures


########## Extracting Features ###########

def extract_features(wav_files_dir):
    wav_files = [wav_files_dir + f for f in listdir(wav_files_dir) if
                 isfile(join(wav_files_dir, f))]
    wav_files.sort()

    ### Broader Features
    print('---------Extracting Higher Level Features------')
    broad_feats = np.zeros((len(wav_files), 12))
    i = 0
    for wav_file in wav_files:
        name = os.path.basename(wav_file)
        print(name)

        y, sr = librosa.load(wav_file)
        yt, index = librosa.effects.trim(y)
        shape = yt.shape
        print(shape)

        get = GetBroaderFeatures(y, sr)
        rms = get.get_rmsenergy()
        tempo = get.get_tempo()
        pitchmed, pitchmean = get.get_pitch()
        freqs = get.estimate_freq()
        print('RMS-', rms)
        print('Tempo-', tempo)
        print('Pitch-', 'Median', pitchmed, 'Mean', pitchmean)
        print('Spectral Contrast-', freqs)

        broad_feats[i, 0] = rms
        broad_feats[i, 1] = pitchmed
        broad_feats[i, 2] = pitchmean
        broad_feats[i, 3] = tempo
        broad_feats[i, 4:12] = freqs
        break

    print('---------BROADER FEATURES DUMPING DONE-----------')

    ### Low Level Features
    print('---------Extracting Lower Level Features------')
    low_feats = np.zeros((len(wav_files), 20, 9762))
    i = 0
    for wav_file in wav_files:
        name = os.path.basename(wav_file)
        print(name)

        y, sr = librosa.load(wav_file)
        yt, index = librosa.effects.trim(y)
        shape = yt.shape
        print(shape)

        ### Padding/Cropping
        if shape[0] < 5000000:
            add_y = int((5000000 - shape[0]) / 2)
            temp_img = np.zeros(5000000, dtype='float64')
            temp_img[add_y:(add_y + shape[0])] = yt
            y2 = temp_img
            print('pad')

        if shape[0] >= 5000000:
            extra = int((shape[0] - 5000000) / 2)
            y2 = yt[extra:(extra + 5000000)]
            print('crop')

        get = GetSpectrumFeatures(y2, sr)
        # mel1 = get.get_melspectrogram()
        mfcc1 = get.get_mfcc()
        # rms1 = get.get_rms()

        # print(mel1.shape)
        print(mfcc1.shape)
        # print(rms1.shape)

        # feats[i, 0:12, 0:len(mel1[1])] = mel1
        low_feats[i, :, :] = mfcc1
        # feats[i, 32, 0:len(rms1[0])] = rms1
        break

    print('---------LOWER FEATURES DUMPING DONE-----------')

    return low_feats


### Predictions
def predict_mood(model, dataX, thresholds):
    predictions = model.predict(dataX)
    y_pred = np.zeros((dataX.shape[0], 8))
    for i in range(dataX.shape[0]):
        for j in range(8):
            y_pred[i][j] = (predictions[i][j] > thresholds[j][1])

    return y_pred, predictions


################ Set Testing Directory ##################
test_directory = '../data/test/'
wav_files = [test_directory + f for f in listdir(test_directory) if
             isfile(join(test_directory, f))]
wav_files.sort()
names=[]
for wav_file in wav_files:
    name = os.path.basename(wav_file)
    names.append(name)
### Load the Saved Model
model = keras.models.load_model('model.h5')
model.summary()

### Extract Features
feats = extract_features(test_directory)
data_X = np.expand_dims(feats, axis=-1)
print(data_X.shape)

### Fix Thresholds
thresholds = [['Happy', 0.8],
              ['Excited', 0.5],
              ['Frantic', 0.5],
              ['Anxious/Sad', 0.3],
              ['Anger', 0.5],
              ['Calm', 0.22],
              ['Tired', 0.5],
              ['Sensual', 0.1]]

### Predictions
predictions, preds = predict_mood(model, data_X, thresholds)
print(preds)
print(predictions)
tags = ['Happy', 'Excited', 'Frantic', 'Anxious/Sad', 'Anger', 'Calm', 'Tired', 'Sensual']

print('------------Predicted Mood Tags----------')
for i in range(predictions.shape[0]):
    print(names[i])
    print('---------------')
    for j in range(8):
        if predictions[i][j] > 0:
            print(tags[j])
    print('---------------')