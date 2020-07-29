import subprocess
import sys
from os import listdir
import librosa
from os.path import isfile, join

sys.path.append("./../")
from configs import config

wav_files = [config.wav_files_dir + f for f in listdir(config.wav_files_dir) 
            if isfile(join(config.wav_files_dir, f))]
wav_files.sort()

for wav_file in wav_files:
    try:
        print(wav_file)

        output_file_name = wav_file.split("/")[-1].split(".")[0]
        # sr = librosa.get_samplerate(wav_file)

        command = "ffmpeg -i " + wav_file + " -f segment -segment_time 5 -c:a copy " + config.split_files_dir + output_file_name + "%03d.wav"
        subprocess.call(command, shell=True)

    except Exception as err:
        print(err)

    # break