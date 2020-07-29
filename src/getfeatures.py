import numpy as np
import librosa
from librosa import stft
from librosa.feature import chroma_stft, chroma_cqt, chroma_cens, melspectrogram, mfcc, rms, spectral_centroid, \
    spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, \
    zero_crossing_rate

import librosa.display

import sys

sys.path.append("../")
from configs import config


class GetSpectrumFeatures:

    def __init__(self, audio_array, sampling_rate, fft_size=None, win_length=None, window=None):

        self._audio_array = audio_array
        self._sampling_rate = sampling_rate

        if fft_size is None:
            self._fft_size = config.n_fft_size
        else:
            self._fft_size = fft_size
        self._hop_length = int(self._fft_size / 4)
        if win_length is None:
            self._win_length = self._fft_size
        else:
            self._win_length = win_length
        if window is None:
            self._window = config.window
        else:
            self._window = window

        self._center = False
        self._n_mfcc = config.number_of_mfcc
        self._n_mels = config.number_of_mels

        audio_stft = stft(
            self._audio_array,
            n_fft=self._fft_size,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._window,
            center=self._center
        )
        self._magnitude_spectrum = np.abs(audio_stft)
        self._power_spectrum = self._magnitude_spectrum ** 2

    def get_melspectrogram(self):
        melspectrum = melspectrogram(S=self._power_spectrum,
                                     sr=self._sampling_rate,
                                     n_mels=self._n_mels)
        return melspectrum

    def get_mfcc(self):
        mfcc_features = mfcc(S=librosa.power_to_db(self._power_spectrum),
                             n_mfcc=self._n_mfcc)
        return mfcc_features

    def get_rms(self):
        rms_features = rms(S=self._magnitude_spectrum)
        return rms_features


class GetBroaderFeatures:

    def __init__(self, audio_array, sampling_rate, fft_size=None, win_length=None, window=None, hop_length=None,
                 frame_length=None):

        self._audio_array = audio_array
        self._sampling_rate = sampling_rate

        if fft_size is None:
            self._fft_size = config.n_fft_size
        else:
            self._fft_size = fft_size
        self._hop_length = int(self._fft_size / 4)
        if win_length is None:
            self._win_length = self._fft_size
        else:
            self._win_length = win_length
        if window is None:
            self._window = config.window
        else:
            self._window = window
        if hop_length is None:
            self._hop_length = config.hop_length
        else:
            self._hop_length = hop_length
        if frame_length is None:
            self._frame_length = config.frame_length
        else:
            self._frame_length = frame_length

        self._center = False
        self._n_mfcc = config.number_of_mfcc
        self._n_mels = config.number_of_mels

    def get_rmsenergy(self):
        energy = np.array([
            sum(abs(self._audio_array[i:i + self._frame_length] ** 2))
            for i in range(0, len(self._audio_array), self._hop_length)
        ])
        energy = np.mean(energy)
        return energy

    def get_tempo(self):
        tempo = librosa.beat.tempo(y=self._audio_array,
                                   sr=self._sampling_rate)
        return tempo[0]

    def estimate_freq(self, n_bands=3):
        spec_con = librosa.feature.spectral_contrast(y=self._audio_array,
                                                     sr=self._sampling_rate,
                                                     n_bands=n_bands)

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        return spec_con_feature

    def get_pitch(self, fmin=50.0, fmax=2000.0):
        freqs = []
        sr = self._sampling_rate
        x = self._audio_array
        onset_samples = librosa.onset.onset_detect(x, sr=sr, units='samples',
                                                   hop_length=self._hop_length,
                                                   backtrack=False,
                                                   pre_max=20,
                                                   post_max=20,
                                                   pre_avg=100,
                                                   post_avg=100,
                                                   delta=0.2,
                                                   wait=0)
        onset_boundaries = np.concatenate([[0], onset_samples, [len(x)]])

        for i in range(len(onset_boundaries) - 3):
            n0 = onset_samples[i]
            n1 = onset_samples[i + 1]
            r = librosa.autocorrelate(x[n0:n1])
            i_min = sr / fmax
            i_max = sr / fmin
            r[:int(i_min)] = 0
            r[int(i_max):] = 0

            # Find the location of the maximum autocorrelation.
            i = r.argmax()
            f0 = float(sr) / i
            freqs.append(f0)
            freqs1 = np.array(freqs)
        return np.median(freqs1), freqs1.mean()
