import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import os
from os.path import isfile, join
from os import listdir
import librosa.display
import h5py
from numpy import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

### Appending the configs and Feature Extraction classes
import sys
sys.path.append("./../")
from configs import config

from getfeatures import GetBroaderFeatures, GetSpectrumFeatures

########## Encoding Labels ###########

def encode_labels(tags_file, tags_map):
    moods_list = np.genfromtxt(tags_file, delimiter=',')
    print('Total no. of Mood tags in Metadata-', moods_list.shape)
    newmoods = np.genfromtxt(tags_map, delimiter=',')
    newmoods_list = np.zeros((3460, 8))

    for j in range(moods_list.shape[0]):
        for x in range(len(newmoods)):
            for y in range(8):
                if newmoods[x][1] == y:
                    newmoods_list[j][y] += moods_list[j][x]

    for i in range(moods_list.shape[0]):
        for j in range(8):
            if newmoods_list[i][j] > 0:
                newmoods_list[i][j] = 1

    np.savetxt('../Dataset/encoded_newtags_main.csv', newmoods_list, delimiter=',')
    print('---------LABELS DUMPING DONE-----------')

    return newmoods_list


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
    h5f = h5py.File('../Dataset/feats_broad_main.h5', 'w')
    h5f.create_dataset('dataset', data=broad_feats)
    h5f.close()
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

    # feats = np.vstack(broad_feats, low_feats)

    h5f = h5py.File('../Dataset/feats_low_main.h5', 'w')
    h5f.create_dataset('dataset', data=low_feats)
    h5f.close()
    print('---------LOWER FEATURES DUMPING DONE-----------')

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc1, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    return low_feats


############ Model Training #############

### Data Generator
def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, IN_SHAPE[0], IN_SHAPE[1], 1))
    batch_labels = np.zeros((batch_size, 8))
    while True:
        for i in range(batch_size):
            index = random.choice(len(features), 1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


### F-Beta metric    (available in sklearn bur deprecated in Keras, so used a self-defined function)
def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score


### Model Architecture
def define_model(in_shape, out_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(60, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(out_shape, activation='sigmoid'))
    return model


############ MAIN ##############

train_dir = config.wav_files_dir
wav_files = [train_dir + f for f in listdir(train_dir) if
             isfile(join(train_dir, f))]
wav_files.sort()
print('No. of songs in directory-', len(wav_files))


### Importing the Song Features and Labels

# newmoods_list = np.genfromtxt('../Dataset/encoded_newtags_main.csv', delimiter=',')
# h5f = h5py.File('../Dataset/feats_low_main.h5','r')
# feats = h5f['dataset'][:]
# h5f.close()
# h5f = h5py.File('../Dataset/feats_broad_main.h5','r')
# broad_feats = h5f['dataset'][:]
# h5f.close()


################ OR
 

### Prepare Data

newmoods_list = encode_labels('../Dataset/encoded_alltags_main.csv', '../tags_data/newtags.csv')
feats = extract_features(config.wav_files_dir)

# feats = np.zeros((3460,20,9762))
data_X = np.expand_dims(feats, axis=-1)
print('Input Shape-', data_X.shape)
data_Y = newmoods_list
print('Output Shape-', data_Y.shape)

IN_SHAPE = data_X.shape[1:]

print(' ')
print('INPUT SHAPE TO THE CNN-', IN_SHAPE)
data_X = data_X[:50]
data_Y = data_Y[:50]

### Shuffle
randomize = np.arange(len(data_Y))
np.random.shuffle(randomize)
train_X_shuffle = data_X[randomize]
train_Y_shuffle = data_Y[randomize]

### Test Set
trainX, testX, trainY, testY = train_test_split(train_X_shuffle,
                                                train_Y_shuffle,
                                                test_size=0.1,
                                                random_state=42)
### Validation Set
trainX, valX, trainY, valY = train_test_split(trainX, trainY,
                                              test_size=0.2,
                                              random_state=42)

print('trainX-', trainX.shape, '   trainY- ', trainY.shape)
print('valX-', valX.shape, '      valY- ', valY.shape)
print('testX-', testX.shape, '     testY- ', testY.shape)

### Running the Model
model = define_model(in_shape=IN_SHAPE, out_shape=8)
model.summary()
opt = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy', fbeta, keras.metrics.categorical_accuracy])

model.fit_generator(generator(trainX, trainY, 2),
                    verbose=1,
                    steps_per_epoch=1,
                    epochs=1,
                    validation_data=(valX, valY))

### Save Model
model.save('../Models/modelxxx.h5')
print('--------Model Saved---------')
