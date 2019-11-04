import librosa
import numpy as np
from pypesq import pesq
from pystoi.stoi import stoi
clean_audio, _ = librosa.load('./clean.wav', sr=16000)
impro_gri, _ = librosa.load('./improved_gri.wav', sr=16000)
gri, _ = librosa.load('./gri.wav', sr=16000)
old, _ = librosa.load('./old.wav', sr=16000)
score_1 = pesq(clean_audio, impro_gri, 16000)
score_2 = stoi(clean_audio, impro_gri, 16000)
score_3 = pesq(clean_audio, gri, 16000)
score_4 = stoi(clean_audio, gri, 16000)
score_5 = pesq(clean_audio, old, 16000)
score_6 = stoi(clean_audio, old, 16000)
print(score_1, score_2, score_3, score_4, score_5, score_6)