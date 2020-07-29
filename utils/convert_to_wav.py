import subprocess
import sys
from os import listdir
import librosa
from os.path import isfile, join

sys.path.append("./../")
from configs import config
print(config.mp3_file_dir)
mp3_files = [config.mp3_file_dir + f for f in listdir(config.mp3_file_dir) if isfile(join(config.mp3_file_dir, f))]
mp3_files.sort()

for mp3_file in mp3_files:
    try:
        print(mp3_file)

        output_file_name = mp3_file.split("/")[-1].split(".")[0]
        sr = librosa.get_samplerate(mp3_file)

        command = "ffmpeg -i " + mp3_file + "  -acodec pcm_u8 -ar 22500 " + config.wav_files_dir + output_file_name + ".wav"
        # command = "ffmpeg -i " + mp3_file + "  -acodec pcm_s16le -ac 2 -ar " + str(sr) + " -vn " + config.wav_files_dir + output_file_name + ".wav"
        subprocess.call(command, shell=True)

    except Exception as err:
        print(err)
