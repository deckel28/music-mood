import os

home_dir = os.path.abspath(os.path.join(__file__, "../../"))
data_dir = home_dir + "/data/"

mp3_file_dir = data_dir + "mp3_files/"
wav_files_dir = data_dir + "wav_files_hindi/"
split_files_dir = data_dir + "split_wav_files/"

json_file = home_dir + "sample_jio_savan_data.json"

frame_size_in_ms = 50
percentage_overlap = 50
n_fft_size = 2048
number_of_mfcc = 20
window = 'hann'
number_of_mels = 20

hop_length = 256
frame_length = 512
