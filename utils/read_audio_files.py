import math
import sys

import librosa

sys.path.append("./../")
from configs import config


for wav_file in wav_files:
    print(wav_file)
    sr = librosa.get_samplerate(wav_file)
    frame_length = int(math.pow(2, math.ceil(math.log2((sr * config.frame_size_in_ms * 0.001)))))
    hop_length = int(config.percentage_overlap * frame_length / 100)
    print(sr, frame_length, hop_length)

    stream = librosa.stream(wav_file, block_length=1, frame_length=frame_length, hop_length=hop_length)
    for frame in stream:
        print(list(frame))
        break
