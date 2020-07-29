import os
import sys
from os import listdir
from os.path import isfile, join
from mutagen.mp3 import MP3
import pandas as pd

sys.path.append("./../")
from configs import config

mp3_files = [config.mp3_file_dir + f for f in listdir(config.mp3_file_dir) if isfile(join(config.mp3_file_dir, f))]
mp3_files.sort()

lengths=[]
names=[]

for mp3_file in mp3_files:
    try:
        audio = MP3(mp3_file)
        audio_info = audio.info
        length_in_secs = int(audio_info.length)
        names.append(os.path.basename(mp3_file))
        lengths.append(length_in_secs)
        print(os.path.basename(mp3_file))
        print(length_in_secs)

    except Exception as err:
        print(err)

df1 = pd.DataFrame(names)
df2 = pd.DataFrame(lengths)
df = pd.concat([df1, df2], axis=1)

df.to_csv('durations.csv', index=False)
