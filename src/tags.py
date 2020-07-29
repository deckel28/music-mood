import json, operator
import numpy as np
from numpy import savetxt
import sys
import os
import csv

sys.path.append("./../")
from configs import config
tags_file = '../tags_data/total_tags.csv'
json_path = '../tags_data/sample_jio_savan_data.json'

with open(tags_file, newline='') as f:
    reader = csv.reader(f)
    totaltags = list(reader)
tags = []
for i in range(1, 153):
    tags.append(totaltags[i][0])
print(tags)
print('Total Number of unique tags-', len(tags))

songs_list = os.listdir(config.wav_files_dir)
songs_list.sort()

list = []
for song in songs_list:
    list.append(os.path.basename(song))
print('Number of Songs in directory-', len(list))

songs_meta_list = []
for line in open(json_path, 'r'):
    songs_meta_list.append(json.loads(line))

songs_meta_list.sort(key=operator.itemgetter('pid'))

array = np.zeros((len(list), 152), dtype=int)
names = []
i = 0
for song in songs_meta_list:
    if song['language'] == 'hindi':
        for j in list:
            if j[:8] == song['pid']:
                names.append(j)
                # print(j)
                for x in song['mood']:
                    if x in tags:
                        array[i][tags.index(x)] = 1
                i = i + 1

savetxt('../Dataset/encoded_alltags_main.csv', array, delimiter=',')
print('SAVED')
