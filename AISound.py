# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:31:39 2022

@author: sp7012
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import librosa
audio_data = 'M.wav'
#Reading the amplitude and sampling rate
x , sr = librosa.load(audio_data)

#Define and implementation of the filters
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

b,a=butter_lowpass(2000, sr, order=16)
x=lfilter(b, a, x)



import IPython.display as ipd
ipd.Audio(audio_data)
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
X = librosa.stft(x, n_fft=2048, hop_length=512, win_length=2048, window="hann", center=True, dtype=None, pad_mode="constant")
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr,x_axis='time', y_axis='hz')
plt.ylim(top=3000)
plt.colorbar()
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
spectral_centroids.shape
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(x, sr=sr, alpha=0.5)
plt.plot(t, normalize(spectral_centroids), color='b')
mfccs = librosa.feature.mfcc(x, sr)
print(mfccs.shape)
#Displaying  the MFCCs:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs[0:5], sr=sr, x_axis='time')
