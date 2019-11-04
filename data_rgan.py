import os
import librosa
import numpy as np
import time
from hparam import hparam as hp
from utils import for_stft_2

print('start to prepare data for rgan')
noisy_path = '/workspace/data/rgan/train/noisy_wav'
clean_path = '/workspace/data/rgan/train/clean_wav'
np_file_list = os.listdir(noisy_path)
i = 0
window_length = int(hp.data.window_gan * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
for item in np_file_list:
    i = i + 1
    item_name = item.split(".")[0]
    clean_tf_path = os.path.join('/workspace/data/rgan/train/clean_tf', item)
    noisy_tf_path = os.path.join('/workspace/data/rgan/train/noisy_tf', item)
    clean_wav = np.load(os.path.join(clean_path, item))
    noisy_wav = np.load(os.path.join(noisy_path, item))
    clean_tf = for_stft_2(clean_wav)
    noisy_tf = for_stft_2(noisy_wav)
    noisy_ma = librosa.stft(noisy_wav, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    noisy_phase = np.angle(noisy_ma, deg=False)
    np.save(os.path.join('/workspace/data/rgan/train/noisy_phase', item), noisy_phase)
    np.save(clean_tf_path, clean_tf)
    np.save(noisy_tf_path, noisy_tf)