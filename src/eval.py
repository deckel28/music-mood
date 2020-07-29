import h5py
import numpy as np
from numpy import savetxt
from numpy import genfromtxt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from keras import backend
from sklearn.metrics import accuracy_score, hamming_loss
import matplotlib.pyplot as plt
import csv

### Appending the configs and Feature Extraction classes
import sys

sys.path.append("./../")
from configs import config

### Importing the Song Features and Labels
newmoods_list = genfromtxt('encoded_newtags_main.csv', delimiter=',')
# h5f = h5py.File('../data/feats_main.h5', 'r')
# feats = h5f['dataset'][:]
# h5f.close()

feats = np.zeros((3460,20,9762))
data_X = np.expand_dims(feats, axis=-1)
print('Input Shape-', data_X.shape)
data_Y = newmoods_list
print('Output Shape-', data_Y.shape)

IN_SHAPE = data_X.shape[1:]
print(IN_SHAPE)

tags = ['Happy', 'Excited', 'Frantic', 'Anxious/Sad', 'Anger', 'Calm', 'Tired', 'Sensual']

### Load the Saved Model
model = keras.models.load_model('model.h5')
model.summary()

### F-Beta metric
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

### Predictions
def predict_mood(model, dataX, thresholds):
    predictions = model.predict(dataX)
    y_pred = np.zeros((dataX.shape[0], 8))
    for i in range(dataX.shape[0]):
        for j in range(8):
            y_pred[i][j] = (predictions[i][j] > thresholds[j][1])

    return y_pred, predictions


### Confusion Matrix
def multilabel_cm(trues, preds):
    shape = trues.shape[0]
    cm = np.zeros((8, 8), dtype=int)
    for i in range(shape):
        for j in range(8):
            for k in range(8):
                if trues[i][j] == 1 and preds[i][k] == 1:
                    cm[j][k] = cm[j][k] + 1
    return cm


### Classification Report
def classification_report(trues, preds, thresholds, cm):
    fn = np.zeros((8,))
    fp = np.zeros((8,))
    tp = np.zeros((8,))
    print('CLASSIFICATION REPORT')
    print('---------------------')
    print(' ')

    cm = multilabel_cm(trues, preds)
    fps = np.sum(cm, axis=0)
    fns = np.sum(cm, axis=1)
    occ = np.sum(trues, axis=0)
    report = np.zeros((8, 4))

    for i in range(8):
        tp[i] = cm[i][i]
        fp[i] = fps[i] - cm[i][i]
        fn[i] = fns[i] - cm[i][i]

        precision = tp[i] / (tp[i] + fp[i])
        report[i][0] = precision
        recall = tp[i] / (tp[i] + fn[i])
        report[i][1] = recall
        f1score = 2 * precision * recall / (precision + recall)
        report[i][2] = f1score
        support = occ[i]
        report[i][3] = support

        print(thresholds[i][0])
        print('.....')
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1score)
        print('Support:', support)
        print('')
        print('')

    return report


### Fix Thresholds
thresholds = [['Happy', 0.8],
              ['Excited', 0.5],
              ['Frantic', 0.5],
              ['Anxious/Sad', 0.3],
              ['Anger', 0.5],
              ['Calm', 0.22],
              ['Tired', 0.5],
              ['Sensual', 0.1]]

### Real Evaluation
y_pred, y_pred_val = predict_mood(model, data_X[:50], thresholds)
cm = multilabel_cm(data_Y[:50], y_pred.astype(np.float))
report = classification_report(data_Y[:50], y_pred.astype(np.float), thresholds, cm)




### Print and Save
tags1 = np.array(tags).reshape((1, 8))
y_pred1 = np.vstack((tags1, y_pred))
print(y_pred1)

cm1 = np.vstack((tags, cm))
print('-------- Confusion Matrix-----------')
print(cm1)
savetxt('cm.csv', cm1, delimiter=',', fmt='%s')

report = np.vstack((['Precision', 'Recall', 'F1-Score', 'Support'], report))
x = np.array(thresholds)
names = np.append(['Label'], x[:, 0])
names = names.reshape(9, 1)
report = np.hstack((names, report))
print(report)
savetxt('report.csv', report, delimiter=',', fmt='%s')

### Box Plotting
for x in range(2):
    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        axs[i].boxplot(y_pred_val[1:, i + 4 * x].astype(np.float), whis=100)
        axs[i].set_title(thresholds[i + 4 * x][0])
    plt.savefig('train_box_' + str(x) + '.png')

### Overall Scores
print('')
print('')
print('')
print('---------------------------------')
accuracy = accuracy_score(data_Y[:50], y_pred)
print('Accuracy: %f' % accuracy)

f1 = fbeta(data_Y[:50], y_pred)
print('F1 score: %f' % f1)

hl = hamming_loss(data_Y[:50], y_pred)
print('Hamming Loss: %f' % hl)

print('')
print('')
print('')