import json
import numpy as np
import pandas as pd
import subprocess

import sys

sys.path.append("./../")
from configs import config

json_path = config.json_file

songs_meta_list = []
for line in open(json_path, 'r'):
    songs_meta_list.append(json.loads(line))

print("number of total songs are ", len(songs_meta_list))

genres = []
moods = []
for song in songs_meta_list:
    # genres.append(song['genre'])
    moods.append(song['mood_tags'])


def unique(total):
    unique_list = []
    count = []
    for i in total:
        for x in i:
            if x not in unique_list:
                unique_list.append(x)
                count.append(1)
            if x in unique_list:
                count[unique_list.index(x)] = count[unique_list.index(x)] + 1

    return unique_list, count


umoods, mcount = unique(moods)
# ugenres, gcount = unique(genres)

print(len(umoods))
print(len(mcount))
array = list(zip(umoods, mcount))

sarray = sorted(array, key=lambda x: x[1], reverse=True)
top50 = []
for i in range(len(sarray)):
    top50.append(sarray[i][0])

df = pd.DataFrame(sarray)
print(df)
df.to_csv('../data/total_tags.csv', index=False)