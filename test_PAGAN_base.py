import librosa
from pypesq import pesq
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparam import hparam as hp
from pystoi.stoi import stoi
from model_gan_multi_SN_ori import *
from utils import gri_lim_1
# rms = lambda y: np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))
n = 0
n_20 = 0
n_10 = 0
n_15 = 0
n_5 = 0
n_0 = 0
n__5 = 0
score_20_pesq = 0
score_10_pesq = 0
score_15_pesq = 0
score_0_pesq = 0
score_5_pesq = 0
score__5_pesq = 0
score_20_stoi = 0
score_10_stoi = 0
score_15_stoi = 0
score_0_stoi = 0
score_5_stoi = 0
score__5_stoi = 0
score_20_pesq_gl = 0
score_10_pesq_gl = 0
score_15_pesq_gl = 0
score_0_pesq_gl = 0
score_5_pesq_gl = 0
score__5_pesq_gl = 0
score_20_stoi_gl = 0
score_10_stoi_gl = 0
score_15_stoi_gl = 0
score_0_stoi_gl = 0
score_5_stoi_gl = 0
score__5_stoi_gl = 0
window_length = int(hp.data.window * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
# clean_audio_path = '/data2/lps/data_gan_test/clean_wav/'
# mix_audio_path = '/data2/lps/data_gan_test/noisy_wav/'
# mix_path = '/data2/lps/data_gan_test/noisy_tf/'
clean_audio_path = '/workspace/data/rgan/test/clean_wav'
mix_audio_path = '/workspace/data/rgan/test/noisy_wav'

if __name__ == '__main__':
    np_file_list = os.listdir(mix_audio_path)
    # device = torch.device(hp.device)
    voicefilter_net = VoiceFilter_SN()
    voicefilter_net.eval()
    save_model = torch.load("/workspace/model/rgan_base/multi_epoch_15.pth")
    model_dict_vf = voicefilter_net.state_dict()

    new_state_dict = {k[7:]: v for k, v in save_model.items()}
    state_dict_vf = {k: v for k, v in new_state_dict.items() if k in model_dict_vf}
    model_dict_vf.update(state_dict_vf)
    voicefilter_net.load_state_dict(model_dict_vf)
    # vf_loss = nn.MSELoss()

    for item in np_file_list:
        item_name = item.split(".")[0]
        print('item:', item)
        som_th = item_name[-2: ]
        # item_name = item_name[:-3]
        # print('som_th:', som_th)
        if (som_th == '20'):

            item_clean_name = item_name
            clean_name = "%s.npy"%item_clean_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_name = "%s.npy" % item_name
            mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
            # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            # mix_audio,_ = librosa.load(mix_audio_path_real, sr=16000)

            clean_audio_0 = np.load(clean_audio_path_real)
            mix_audio_0 = np.load(mix_audio_path_real)
            #
            # clean_audio_0 = clean_audio[0: 16000]
            # clean_audio_1 = clean_audio[16000: 32000]
            # clean_audio_2 = clean_audio[32000: 48000]
            # mix_audio_0 = mix_audio[0: 16000]
            # mix_audio_1 = mix_audio[16000: 32000]
            # mix_audio_2 = mix_audio[32000: 48000]

            n_20 = n_20 + 1
            mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # mix_angle = np.angle(mix_tf).T
            utters_mix_0 = np.abs(mix_tf_0)
            # mix_tf_name = "%s.npy"%item_name
            # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            #
            # utters_mix = np.load(mix_tf_path)

            mix = torch.Tensor(utters_mix_0)
            mix = mix.transpose(1, 0)
            mix = mix.unsqueeze(0)
            mix = mix.unsqueeze(0)
            # mix = mix.contiguous().view(1, 1, -1, 257)
            out = voicefilter_net(mix)

            aaaa = mix * out

            aaaa = torch.squeeze(aaaa)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            # out_something = aaaa.reshape(301, 257).detach().numpy().T

            aaa = librosa.istft(aaaa*(mix_tf_0/np.abs(mix_tf_0)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio_0, out, 16000)
            score_stoi_gl = stoi(clean_audio_0, out, 16000)
            score_pesq = pesq(clean_audio_0, aaa, 16000)
            score_stoi = stoi(clean_audio_0, aaa, 16000)
            score_20_pesq = score_20_pesq + score_pesq
            score_20_stoi = score_20_stoi + score_stoi
            score_20_pesq_gl = score_20_pesq_gl + score_pesq_gl
            score_20_stoi_gl = score_20_stoi_gl + score_stoi_gl

            print('n_20', n_20, 'score_20_pesq:', score_20_pesq, 'val_pesq:', score_20_pesq/n_20)
            print('n_20', n_20, 'score_20_stoi:', score_20_stoi, 'val_stoi:', score_20_stoi/n_20)
            print('n_20', n_20, 'score_20_pesq_gl:', score_20_pesq_gl, 'val_pesq_gl:', score_20_pesq_gl / n_20)
            print('n_20', n_20, 'score_20_stoi_gl:', score_20_stoi_gl, 'val_stoi_gl:', score_20_stoi_gl / n_20)
            #
            # n_20 = n_20 + 1
            # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_1 = np.abs(mix_tf_1)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_1)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_1, out, 16000)
            # score_stoi_gl = stoi(clean_audio_1, out, 16000)
            # score_pesq = pesq(clean_audio_1, aaa, 16000)
            # score_stoi = stoi(clean_audio_1, aaa, 16000)
            # score_20_pesq = score_20_pesq + score_pesq
            # score_20_stoi = score_20_stoi + score_stoi
            # score_20_pesq_gl = score_20_pesq_gl + score_pesq_gl
            # score_20_stoi_gl = score_20_stoi_gl + score_stoi_gl
            #
            # print('n_20', n_20, 'score_20_pesq:', score_20_pesq, 'val_pesq:', score_20_pesq / n_20)
            # print('n_20', n_20, 'score_20_stoi:', score_20_stoi, 'val_stoi:', score_20_stoi / n_20)
            # print('n_20', n_20, 'score_20_pesq_gl:', score_20_pesq_gl, 'val_pesq_gl:', score_20_pesq_gl / n_20)
            # print('n_20', n_20, 'score_20_stoi_gl:', score_20_stoi_gl, 'val_stoi_gl:', score_20_stoi_gl / n_20)
            #
            # n_20 = n_20 + 1
            # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_2 = np.abs(mix_tf_2)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_2)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_2, out, 16000)
            # score_stoi_gl = stoi(clean_audio_2, out, 16000)
            # score_pesq = pesq(clean_audio_2, aaa, 16000)
            # score_stoi = stoi(clean_audio_2, aaa, 16000)
            # score_20_pesq = score_20_pesq + score_pesq
            # score_20_stoi = score_20_stoi + score_stoi
            # score_20_pesq_gl = score_20_pesq_gl + score_pesq_gl
            # score_20_stoi_gl = score_20_stoi_gl + score_stoi_gl
        #
        #     print('n_20', n_20, 'score_20_pesq:', score_20_pesq, 'val_pesq:', score_20_pesq / n_20)
        #     print('n_20', n_20, 'score_20_stoi:', score_20_stoi, 'val_stoi:', score_20_stoi / n_20)
        #     print('n_20', n_20, 'score_20_pesq_gl:', score_20_pesq_gl, 'val_pesq_gl:', score_20_pesq_gl / n_20)
        #     print('n_20', n_20, 'score_20_stoi_gl:', score_20_stoi_gl, 'val_stoi_gl:', score_20_stoi_gl / n_20)
        elif (som_th == '10'):
            item_clean_name = item_name
            clean_name = "%s.npy" % item_clean_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_name = "%s.npy" % item_name
            mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
            # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            # mix_audio,_ = librosa.load(mix_audio_path_real, sr=16000)
            # clean_audio_0 = clean_audio[0: 16000]
            # clean_audio_1 = clean_audio[16000: 32000]
            # clean_audio_2 = clean_audio[32000: 48000]
            # mix_audio_0 = mix_audio[0: 16000]
            # mix_audio_1 = mix_audio[16000: 32000]
            # mix_audio_2 = mix_audio[32000: 48000]

            clean_audio_0 = np.load(clean_audio_path_real)
            mix_audio_0 = np.load(mix_audio_path_real)

            n_10 = n_10 + 1
            mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # mix_angle = np.angle(mix_tf).T
            utters_mix_0 = np.abs(mix_tf_0)
            # mix_tf_name = "%s.npy"%item_name
            # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            #
            # utters_mix = np.load(mix_tf_path)

            mix = torch.Tensor(utters_mix_0)
            mix = mix.transpose(1, 0)
            mix = mix.unsqueeze(0)
            mix = mix.unsqueeze(0)
            # mix = mix.contiguous().view(1, 1, -1, 257)
            out = voicefilter_net(mix)

            aaaa = mix * out

            aaaa = torch.squeeze(aaaa)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            # out_something = aaaa.reshape(301, 257).detach().numpy().T

            aaa = librosa.istft(aaaa * (mix_tf_0 / np.abs(mix_tf_0)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio_0, out, 16000)
            score_stoi_gl = stoi(clean_audio_0, out, 16000)
            score_pesq = pesq(clean_audio_0, aaa, 16000)
            score_stoi = stoi(clean_audio_0, aaa, 16000)
            score_10_pesq = score_10_pesq + score_pesq
            score_10_stoi = score_10_stoi + score_stoi
            score_10_pesq_gl = score_10_pesq_gl + score_pesq_gl
            score_10_stoi_gl = score_10_stoi_gl + score_stoi_gl

            print('n_10', n_10, 'score_10_pesq:', score_10_pesq, 'val_pesq:', score_10_pesq / n_10)
            print('n_10', n_10, 'score_10_stoi:', score_10_stoi, 'val_stoi:', score_10_stoi / n_10)
            print('n_10', n_10, 'score_10_pesq_gl:', score_10_pesq_gl, 'val_pesq_gl:', score_10_pesq_gl / n_10)
            print('n_10', n_10, 'score_10_stoi_gl:', score_10_stoi_gl, 'val_stoi_gl:', score_10_stoi_gl / n_10)
            #
            # n_10 = n_10 + 1
            # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_1 = np.abs(mix_tf_1)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_1)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_1, out, 16000)
            # score_stoi_gl = stoi(clean_audio_1, out, 16000)
            # score_pesq = pesq(clean_audio_1, aaa, 16000)
            # score_stoi = stoi(clean_audio_1, aaa, 16000)
            # score_10_pesq = score_10_pesq + score_pesq
            # score_10_stoi = score_10_stoi + score_stoi
            # score_10_pesq_gl = score_10_pesq_gl + score_pesq_gl
            # score_10_stoi_gl = score_10_stoi_gl + score_stoi_gl
            #
            # print('n_10', n_10, 'score_10_pesq:', score_10_pesq, 'val_pesq:', score_10_pesq / n_10)
            # print('n_10', n_10, 'score_10_stoi:', score_10_stoi, 'val_stoi:', score_10_stoi / n_10)
            # print('n_10', n_10, 'score_10_pesq_gl:', score_10_pesq_gl, 'val_pesq_gl:', score_10_pesq_gl / n_10)
            # print('n_10', n_10, 'score_10_stoi_gl:', score_10_stoi_gl, 'val_stoi_gl:', score_10_stoi_gl / n_10)
            #
            # n_10 = n_10 + 1
            # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_2 = np.abs(mix_tf_2)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_2)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_2, out, 16000)
            # score_stoi_gl = stoi(clean_audio_2, out, 16000)
            # score_pesq = pesq(clean_audio_2, aaa, 16000)
            # score_stoi = stoi(clean_audio_2, aaa, 16000)
            # score_10_pesq = score_10_pesq + score_pesq
            # score_10_stoi = score_10_stoi + score_stoi
            # score_10_pesq_gl = score_10_pesq_gl + score_pesq_gl
            # score_10_stoi_gl = score_10_stoi_gl + score_stoi_gl
            #
            # print('n_10', n_10, 'score_10_pesq:', score_10_pesq, 'val_pesq:', score_10_pesq / n_10)
            # print('n_10', n_10, 'score_10_stoi:', score_10_stoi, 'val_stoi:', score_10_stoi / n_10)
            # print('n_10', n_10, 'score_10_pesq_gl:', score_10_pesq_gl, 'val_pesq_gl:', score_10_pesq_gl / n_10)
            # print('n_10', n_10, 'score_10_stoi_gl:', score_10_stoi_gl, 'val_stoi_gl:', score_10_stoi_gl / n_10)
        elif(som_th == '_0'):
            item_clean_name = item_name
            clean_name = "%s.npy" % item_clean_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_name = "%s.npy" % item_name
            mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
            # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            # mix_audio,_ = librosa.load(mix_audio_path_real, sr=16000)
            # clean_audio_0 = clean_audio[0: 16000]
            # clean_audio_1 = clean_audio[16000: 32000]
            # clean_audio_2 = clean_audio[32000: 48000]
            # mix_audio_0 = mix_audio[0: 16000]
            # mix_audio_1 = mix_audio[16000: 32000]
            # mix_audio_2 = mix_audio[32000: 48000]

            clean_audio_0 = np.load(clean_audio_path_real)
            mix_audio_0 = np.load(mix_audio_path_real)

            n_0 = n_0 + 1
            mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # mix_angle = np.angle(mix_tf).T
            utters_mix_0 = np.abs(mix_tf_0)
            # mix_tf_name = "%s.npy"%item_name
            # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            #
            # utters_mix = np.load(mix_tf_path)

            mix = torch.Tensor(utters_mix_0)
            mix = mix.transpose(1, 0)
            mix = mix.unsqueeze(0)
            mix = mix.unsqueeze(0)
            # mix = mix.contiguous().view(1, 1, -1, 257)
            out = voicefilter_net(mix)

            aaaa = mix * out

            aaaa = torch.squeeze(aaaa)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            # out_something = aaaa.reshape(301, 257).detach().numpy().T

            aaa = librosa.istft(aaaa * (mix_tf_0 / np.abs(mix_tf_0)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio_0, out, 16000)
            score_stoi_gl = stoi(clean_audio_0, out, 16000)
            score_pesq = pesq(clean_audio_0, aaa, 16000)
            score_stoi = stoi(clean_audio_0, aaa, 16000)
            score_0_pesq = score_0_pesq + score_pesq
            score_0_stoi = score_0_stoi + score_stoi
            score_0_pesq_gl = score_0_pesq_gl + score_pesq_gl
            score_0_stoi_gl = score_0_stoi_gl + score_stoi_gl

            print('n_0', n_0, 'score_0_pesq:', score_0_pesq, 'val_pesq:', score_0_pesq / n_0)
            print('n_0', n_0, 'score_0_stoi:', score_0_stoi, 'val_stoi:', score_0_stoi / n_0)
            print('n_0', n_0, 'score_0_pesq_gl:', score_0_pesq_gl, 'val_pesq_gl:', score_0_pesq_gl / n_0)
            print('n_0', n_0, 'score_0_stoi_gl:', score_0_stoi_gl, 'val_stoi_gl:', score_0_stoi_gl / n_0)
            #
            # n_0 = n_0 + 1
            # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_1 = np.abs(mix_tf_1)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_1)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_1, out, 16000)
            # score_stoi_gl = stoi(clean_audio_1, out, 16000)
            # score_pesq = pesq(clean_audio_1, aaa, 16000)
            # score_stoi = stoi(clean_audio_1, aaa, 16000)
            # score_0_pesq = score_0_pesq + score_pesq
            # score_0_stoi = score_0_stoi + score_stoi
            # score_0_pesq_gl = score_0_pesq_gl + score_pesq_gl
            # score_0_stoi_gl = score_0_stoi_gl + score_stoi_gl
            #
            # print('n_0', n_0, 'score_0_pesq:', score_0_pesq, 'val_pesq:', score_0_pesq / n_0)
            # print('n_0', n_0, 'score_0_stoi:', score_0_stoi, 'val_stoi:', score_0_stoi / n_0)
            # print('n_0', n_0, 'score_0_pesq_gl:', score_0_pesq_gl, 'val_pesq_gl:', score_0_pesq_gl / n_0)
            # print('n_0', n_0, 'score_0_stoi_gl:', score_0_stoi_gl, 'val_stoi_gl:', score_0_stoi_gl / n_0)
            #
            # n_0 = n_0 + 1
            # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_2 = np.abs(mix_tf_2)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_2)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_2, out, 16000)
            # score_stoi_gl = stoi(clean_audio_2, out, 16000)
            # score_pesq = pesq(clean_audio_2, aaa, 16000)
            # score_stoi = stoi(clean_audio_2, aaa, 16000)
            # score_0_pesq = score_0_pesq + score_pesq
            # score_0_stoi = score_0_stoi + score_stoi
            # score_0_pesq_gl = score_0_pesq_gl + score_pesq_gl
            # score_0_stoi_gl = score_0_stoi_gl + score_stoi_gl
            #
            # print('n_0', n_0, 'score_0_pesq:', score_0_pesq, 'val_pesq:', score_0_pesq / n_0)
            # print('n_0', n_0, 'score_0_stoi:', score_0_stoi, 'val_stoi:', score_0_stoi / n_0)
            # print('n_0', n_0, 'score_0_pesq_gl:', score_0_pesq_gl, 'val_pesq_gl:', score_0_pesq_gl / n_0)
            # print('n_0', n_0, 'score_0_stoi_gl:', score_0_stoi_gl, 'val_stoi_gl:', score_0_stoi_gl / n_0)
        elif(som_th == '15'):
            item_clean_name = item_name
            clean_name = "%s.npy" % item_clean_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_name = "%s.npy" % item_name
            mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
            # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            # mix_audio,_ = librosa.load(mix_audio_path_real, sr=16000)
            # clean_audio_0 = clean_audio[0: 16000]
            # clean_audio_1 = clean_audio[16000: 32000]
            # clean_audio_2 = clean_audio[32000: 48000]
            # mix_audio_0 = mix_audio[0: 16000]
            # mix_audio_1 = mix_audio[16000: 32000]
            # mix_audio_2 = mix_audio[32000: 48000]

            clean_audio_0 = np.load(clean_audio_path_real)
            mix_audio_0 = np.load(mix_audio_path_real)

            n_15 = n_15 + 1
            mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # mix_angle = np.angle(mix_tf).T
            utters_mix_0 = np.abs(mix_tf_0)
            # mix_tf_name = "%s.npy"%item_name
            # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            #
            # utters_mix = np.load(mix_tf_path)

            mix = torch.Tensor(utters_mix_0)
            mix = mix.transpose(1, 0)
            mix = mix.unsqueeze(0)
            mix = mix.unsqueeze(0)
            # mix = mix.contiguous().view(1, 1, -1, 257)
            out = voicefilter_net(mix)

            aaaa = mix * out

            aaaa = torch.squeeze(aaaa)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            # out_something = aaaa.reshape(301, 257).detach().numpy().T

            aaa = librosa.istft(aaaa * (mix_tf_0 / np.abs(mix_tf_0)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio_0, out, 16000)
            score_stoi_gl = stoi(clean_audio_0, out, 16000)
            score_pesq = pesq(clean_audio_0, aaa, 16000)
            score_stoi = stoi(clean_audio_0, aaa, 16000)
            score_15_pesq = score_15_pesq + score_pesq
            score_15_stoi = score_15_stoi + score_stoi
            score_15_pesq_gl = score_15_pesq_gl + score_pesq_gl
            score_15_stoi_gl = score_15_stoi_gl + score_stoi_gl

            print('n_15', n_15, 'score_15_pesq:', score_15_pesq, 'val_pesq:', score_15_pesq / n_15)
            print('n_15', n_15, 'score_15_stoi:', score_15_stoi, 'val_stoi:', score_15_stoi / n_15)
            print('n_15', n_15, 'score_15_pesq_gl:', score_15_pesq_gl, 'val_pesq_gl:', score_15_pesq_gl / n_15)
            print('n_15', n_15, 'score_15_stoi_gl:', score_15_stoi_gl, 'val_stoi_gl:', score_15_stoi_gl / n_15)
            #
            # n_15 = n_15 + 1
            # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_1 = np.abs(mix_tf_1)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_1)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_1, out, 16000)
            # score_stoi_gl = stoi(clean_audio_1, out, 16000)
            # score_pesq = pesq(clean_audio_1, aaa, 16000)
            # score_stoi = stoi(clean_audio_1, aaa, 16000)
            # score_15_pesq = score_15_pesq + score_pesq
            # score_15_stoi = score_15_stoi + score_stoi
            # score_15_pesq_gl = score_15_pesq_gl + score_pesq_gl
            # score_15_stoi_gl = score_15_stoi_gl + score_stoi_gl
            #
            # print('n_15', n_15, 'score_15_pesq:', score_15_pesq, 'val_pesq:', score_15_pesq / n_15)
            # print('n_15', n_15, 'score_15_stoi:', score_15_stoi, 'val_stoi:', score_15_stoi / n_15)
            # print('n_15', n_15, 'score_15_pesq_gl:', score_15_pesq_gl, 'val_pesq_gl:', score_15_pesq_gl / n_15)
            # print('n_15', n_15, 'score_15_stoi_gl:', score_15_stoi_gl, 'val_stoi_gl:', score_15_stoi_gl / n_15)
            #
            # n_15 = n_15 + 1
            # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            # # mix_angle = np.angle(mix_tf).T
            # utters_mix_2 = np.abs(mix_tf_2)
            # # mix_tf_name = "%s.npy"%item_name
            # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # #
            # # utters_mix = np.load(mix_tf_path)
            #
            # mix = torch.Tensor(utters_mix_2)
            # mix = mix.transpose(1, 0)
            # mix = mix.unsqueeze(0)
            # mix = mix.unsqueeze(0)
            # # mix = mix.contiguous().view(1, 1, -1, 257)
            # out = voicefilter_net(mix)
            #
            # aaaa = mix * out
            #
            # aaaa = torch.squeeze(aaaa)
            # aaaa = aaaa.transpose(0, 1).detach().numpy()
            # # out_something = aaaa.reshape(301, 257).detach().numpy().T
            #
            # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length, win_length=window_length)
            # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
            # score_pesq_gl = pesq(clean_audio_2, out, 16000)
            # score_stoi_gl = stoi(clean_audio_2, out, 16000)
            # score_pesq = pesq(clean_audio_2, aaa, 16000)
            # score_stoi = stoi(clean_audio_2, aaa, 16000)
            # score_15_pesq = score_15_pesq + score_pesq
            # score_15_stoi = score_15_stoi + score_stoi
            # score_15_pesq_gl = score_15_pesq_gl + score_pesq_gl
            # score_15_stoi_gl = score_15_stoi_gl + score_stoi_gl
            #
            # print('n_15', n_15, 'score_15_pesq:', score_15_pesq, 'val_pesq:', score_15_pesq / n_15)
            # print('n_15', n_15, 'score_15_stoi:', score_15_stoi, 'val_stoi:', score_15_stoi / n_15)
            # print('n_15', n_15, 'score_15_pesq_gl:', score_15_pesq_gl, 'val_pesq_gl:', score_15_pesq_gl / n_15)
            # print('n_15', n_15, 'score_15_stoi_gl:', score_15_stoi_gl, 'val_stoi_gl:', score_15_stoi_gl / n_15)
        elif(som_th == '_5'):
            som_th_1 = item_name[-3: ]
            print('name:', som_th_1)
            if(som_th_1 == '__5'):
                item_clean_name = item_name
                clean_name = "%s.npy" % item_clean_name
                clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
                mix_name = "%s.npy" % item_name
                mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
                # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
                # mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
                # clean_audio_0 = clean_audio[0: 16000]
                # clean_audio_1 = clean_audio[16000: 32000]
                # clean_audio_2 = clean_audio[32000: 48000]
                # mix_audio_0 = mix_audio[0: 16000]
                # mix_audio_1 = mix_audio[16000: 32000]
                # mix_audio_2 = mix_audio[32000: 48000]

                clean_audio_0 = np.load(clean_audio_path_real)
                mix_audio_0 = np.load(mix_audio_path_real)

                n__5 = n__5 + 1
                mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length,
                                        win_length=window_length)
                # mix_angle = np.angle(mix_tf).T
                utters_mix_0 = np.abs(mix_tf_0)
                # mix_tf_name = "%s.npy"%item_name
                # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                #
                # utters_mix = np.load(mix_tf_path)

                mix = torch.Tensor(utters_mix_0)
                mix = mix.transpose(1, 0)
                mix = mix.unsqueeze(0)
                mix = mix.unsqueeze(0)
                # mix = mix.contiguous().view(1, 1, -1, 257)
                out = voicefilter_net(mix)

                aaaa = mix * out

                aaaa = torch.squeeze(aaaa)
                aaaa = aaaa.transpose(0, 1).detach().numpy()
                # out_something = aaaa.reshape(301, 257).detach().numpy().T

                aaa = librosa.istft(aaaa * (mix_tf_0 / np.abs(mix_tf_0)), hop_length=hop_length,
                                    win_length=window_length)
                out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
                score_pesq_gl = pesq(clean_audio_0, out, 16000)
                score_stoi_gl = stoi(clean_audio_0, out, 16000)
                score_pesq = pesq(clean_audio_0, aaa, 16000)
                score_stoi = stoi(clean_audio_0, aaa, 16000)
                score__5_pesq = score__5_pesq + score_pesq
                score__5_stoi = score__5_stoi + score_stoi
                score__5_pesq_gl = score__5_pesq_gl + score_pesq_gl
                score__5_stoi_gl = score__5_stoi_gl + score_stoi_gl

                print('n__5', n__5, 'score__5_pesq:', score__5_pesq, 'val_pesq:', score__5_pesq / n__5)
                print('n__5', n__5, 'score__5_stoi:', score__5_stoi, 'val_stoi:', score__5_stoi / n__5)
                print('n__5', n__5, 'score__5_pesq_gl:', score__5_pesq_gl, 'val_pesq_gl:', score__5_pesq_gl / n__5)
                print('n__5', n__5, 'score__5_stoi_gl:', score__5_stoi_gl, 'val_stoi_gl:', score__5_stoi_gl / n__5)
                #
                # n__5 = n__5 + 1
                # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length,
                #                         win_length=window_length)
                # # mix_angle = np.angle(mix_tf).T
                # utters_mix_1 = np.abs(mix_tf_1)
                # # mix_tf_name = "%s.npy"%item_name
                # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # #
                # # utters_mix = np.load(mix_tf_path)
                #
                # mix = torch.Tensor(utters_mix_1)
                # mix = mix.transpose(1, 0)
                # mix = mix.unsqueeze(0)
                # mix = mix.unsqueeze(0)
                # # mix = mix.contiguous().view(1, 1, -1, 257)
                # out = voicefilter_net(mix)
                #
                # aaaa = mix * out
                #
                # aaaa = torch.squeeze(aaaa)
                # aaaa = aaaa.transpose(0, 1).detach().numpy()
                # # out_something = aaaa.reshape(301, 257).detach().numpy().T
                #
                # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length,
                #                     win_length=window_length)
                # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
                # score_pesq_gl = pesq(clean_audio_1, out, 16000)
                # score_stoi_gl = stoi(clean_audio_1, out, 16000)
                # score_pesq = pesq(clean_audio_1, aaa, 16000)
                # score_stoi = stoi(clean_audio_1, aaa, 16000)
                # score__5_pesq = score__5_pesq + score_pesq
                # score__5_stoi = score__5_stoi + score_stoi
                # score__5_pesq_gl = score__5_pesq_gl + score_pesq_gl
                # score__5_stoi_gl = score__5_stoi_gl + score_stoi_gl
                #
                # print('n__5', n__5, 'score__5_pesq:', score__5_pesq, 'val_pesq:', score__5_pesq / n__5)
                # print('n__5', n__5, 'score__5_stoi:', score__5_stoi, 'val_stoi:', score__5_stoi / n__5)
                # print('n__5', n__5, 'score__5_pesq_gl:', score__5_pesq_gl, 'val_pesq_gl:', score__5_pesq_gl / n__5)
                # print('n__5', n__5, 'score__5_stoi_gl:', score__5_stoi_gl, 'val_stoi_gl:', score__5_stoi_gl / n__5)
                #
                # n__5 = n__5 + 1
                # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length,
                #                         win_length=window_length)
                # # mix_angle = np.angle(mix_tf).T
                # utters_mix_2 = np.abs(mix_tf_2)
                # # mix_tf_name = "%s.npy"%item_name
                # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # #
                # # utters_mix = np.load(mix_tf_path)
                #
                # mix = torch.Tensor(utters_mix_2)
                # mix = mix.transpose(1, 0)
                # mix = mix.unsqueeze(0)
                # mix = mix.unsqueeze(0)
                # # mix = mix.contiguous().view(1, 1, -1, 257)
                # out = voicefilter_net(mix)
                #
                # aaaa = mix * out
                #
                # aaaa = torch.squeeze(aaaa)
                # aaaa = aaaa.transpose(0, 1).detach().numpy()
                # # out_something = aaaa.reshape(301, 257).detach().numpy().T
                #
                # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length,
                #                     win_length=window_length)
                # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
                # score_pesq_gl = pesq(clean_audio_2, out, 16000)
                # score_stoi_gl = stoi(clean_audio_2, out, 16000)
                # score_pesq = pesq(clean_audio_2, aaa, 16000)
                # score_stoi = stoi(clean_audio_2, aaa, 16000)
                #
                # score__5_pesq = score__5_pesq + score_pesq
                # score__5_stoi = score__5_stoi + score_stoi
                # score__5_pesq_gl = score__5_pesq_gl + score_pesq_gl
                # score__5_stoi_gl = score__5_stoi_gl + score_stoi_gl
                #
                # print('n__5', n__5, 'score__5_pesq:', score__5_pesq, 'val_pesq:', score__5_pesq / n__5)
                # print('n__5', n__5, 'score__5_stoi:', score__5_stoi, 'val_stoi:', score__5_stoi / n__5)
                # print('n__5', n__5, 'score__5_pesq_gl:', score__5_pesq_gl, 'val_pesq_gl:', score__5_pesq_gl / n__5)
                # print('n__5', n__5, 'score__5_stoi_gl:', score__5_stoi_gl, 'val_stoi_gl:', score__5_stoi_gl / n__5)
            else:
                item_clean_name = item_name
                clean_name = "%s.npy" % item_clean_name
                clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
                mix_name = "%s.npy" % item_name
                mix_audio_path_real = os.path.join(mix_audio_path, mix_name)
                # clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
                # mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
                # clean_audio_0 = clean_audio[0: 16000]
                # clean_audio_1 = clean_audio[16000: 32000]
                # clean_audio_2 = clean_audio[32000: 48000]
                # mix_audio_0 = mix_audio[0: 16000]
                # mix_audio_1 = mix_audio[16000: 32000]
                # mix_audio_2 = mix_audio[32000: 48000]

                clean_audio_0 = np.load(clean_audio_path_real)
                mix_audio_0 = np.load(mix_audio_path_real)

                n_5 = n_5 + 1
                mix_tf_0 = librosa.stft(mix_audio_0, n_fft=hp.data.nfft, hop_length=hop_length,
                                        win_length=window_length)
                # mix_angle = np.angle(mix_tf).T
                utters_mix_0 = np.abs(mix_tf_0)
                # mix_tf_name = "%s.npy"%item_name
                # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                #
                # utters_mix = np.load(mix_tf_path)

                mix = torch.Tensor(utters_mix_0)
                mix = mix.transpose(1, 0)
                mix = mix.unsqueeze(0)
                mix = mix.unsqueeze(0)
                # mix = mix.contiguous().view(1, 1, -1, 257)
                out = voicefilter_net(mix)

                aaaa = mix * out

                aaaa = torch.squeeze(aaaa)
                aaaa = aaaa.transpose(0, 1).detach().numpy()
                # out_something = aaaa.reshape(301, 257).detach().numpy().T

                aaa = librosa.istft(aaaa * (mix_tf_0 / np.abs(mix_tf_0)), hop_length=hop_length,
                                    win_length=window_length)
                out = gri_lim_1(aaaa, mix_audio_0, hp.data.nfft, hop_length, window_length)
                score_pesq_gl = pesq(clean_audio_0, out, 16000)
                score_stoi_gl = stoi(clean_audio_0, out, 16000)
                score_pesq = pesq(clean_audio_0, aaa, 16000)
                score_stoi = stoi(clean_audio_0, aaa, 16000)
                score_5_pesq = score_5_pesq + score_pesq
                score_5_stoi = score_5_stoi + score_stoi
                score_5_pesq_gl = score_5_pesq_gl + score_pesq_gl
                score_5_stoi_gl = score_5_stoi_gl + score_stoi_gl

                print('n_5', n_5, 'score_5_pesq:', score_5_pesq, 'val_pesq:', score_5_pesq / n_5)
                print('n_5', n_5, 'score_5_stoi:', score_5_stoi, 'val_stoi:', score_5_stoi / n_5)
                print('n_5', n_5, 'score_5_pesq_gl:', score_5_pesq_gl, 'val_pesq_gl:', score_5_pesq_gl / n_5)
                print('n_5', n_5, 'score_5_stoi_gl:', score_5_stoi_gl, 'val_stoi_gl:', score_5_stoi_gl / n_5)
                #
                # n_5 = n_5 + 1
                # mix_tf_1 = librosa.stft(mix_audio_1, n_fft=hp.data.nfft, hop_length=hop_length,
                #                         win_length=window_length)
                # # mix_angle = np.angle(mix_tf).T
                # utters_mix_1 = np.abs(mix_tf_1)
                # # mix_tf_name = "%s.npy"%item_name
                # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # #
                # # utters_mix = np.load(mix_tf_path)
                #
                # mix = torch.Tensor(utters_mix_1)
                # mix = mix.transpose(1, 0)
                # mix = mix.unsqueeze(0)
                # mix = mix.unsqueeze(0)
                # # mix = mix.contiguous().view(1, 1, -1, 257)
                # out = voicefilter_net(mix)
                #
                # aaaa = mix * out
                #
                # aaaa = torch.squeeze(aaaa)
                # aaaa = aaaa.transpose(0, 1).detach().numpy()
                # # out_something = aaaa.reshape(301, 257).detach().numpy().T
                #
                # aaa = librosa.istft(aaaa * (mix_tf_1 / np.abs(mix_tf_1)), hop_length=hop_length,
                #                     win_length=window_length)
                # out = gri_lim_1(aaaa, mix_audio_1, hp.data.nfft, hop_length, window_length)
                # score_pesq_gl = pesq(clean_audio_1, out, 16000)
                # score_stoi_gl = stoi(clean_audio_1, out, 16000)
                # score_pesq = pesq(clean_audio_1, aaa, 16000)
                # score_stoi = stoi(clean_audio_1, aaa, 16000)
                # score_5_pesq = score_5_pesq + score_pesq
                # score_5_stoi = score_5_stoi + score_stoi
                # score_5_pesq_gl = score_5_pesq_gl + score_pesq_gl
                # score_5_stoi_gl = score_5_stoi_gl + score_stoi_gl
                #
                # print('n_5', n_5, 'score_5_pesq:', score_5_pesq, 'val_pesq:', score_5_pesq / n_5)
                # print('n_5', n_5, 'score_5_stoi:', score_5_stoi, 'val_stoi:', score_5_stoi / n_5)
                # print('n_5', n_5, 'score_5_pesq_gl:', score_5_pesq_gl, 'val_pesq_gl:', score_5_pesq_gl / n_5)
                # print('n_5', n_5, 'score_5_stoi_gl:', score_5_stoi_gl, 'val_stoi_gl:', score_5_stoi_gl / n_5)
                #
                # n_5 = n_5 + 1
                # mix_tf_2 = librosa.stft(mix_audio_2, n_fft=hp.data.nfft, hop_length=hop_length,
                #                         win_length=window_length)
                # # mix_angle = np.angle(mix_tf).T
                # utters_mix_2 = np.abs(mix_tf_2)
                # # mix_tf_name = "%s.npy"%item_name
                # # mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # #
                # # utters_mix = np.load(mix_tf_path)
                #
                # mix = torch.Tensor(utters_mix_2)
                # mix = mix.transpose(1, 0)
                # mix = mix.unsqueeze(0)
                # mix = mix.unsqueeze(0)
                # # mix = mix.contiguous().view(1, 1, -1, 257)
                # out = voicefilter_net(mix)
                #
                # aaaa = mix * out
                #
                # aaaa = torch.squeeze(aaaa)
                # aaaa = aaaa.transpose(0, 1).detach().numpy()
                # # out_something = aaaa.reshape(301, 257).detach().numpy().T
                #
                # aaa = librosa.istft(aaaa * (mix_tf_2 / np.abs(mix_tf_2)), hop_length=hop_length,
                #                     win_length=window_length)
                # out = gri_lim_1(aaaa, mix_audio_2, hp.data.nfft, hop_length, window_length)
                # score_pesq_gl = pesq(clean_audio_2, out, 16000)
                # score_stoi_gl = stoi(clean_audio_2, out, 16000)
                # score_pesq = pesq(clean_audio_2, aaa, 16000)
                # score_stoi = stoi(clean_audio_2, aaa, 16000)
                #
                # score_5_pesq = score_5_pesq + score_pesq
                # score_5_stoi = score_5_stoi + score_stoi
                # score_5_pesq_gl = score_5_pesq_gl + score_pesq_gl
                # score_5_stoi_gl = score_5_stoi_gl + score_stoi_gl
                #
                # print('n_5', n_5, 'score_5_pesq:', score_5_pesq, 'val_pesq:', score_5_pesq / n_5)
                # print('n_5', n_5, 'score_5_stoi:', score_5_stoi, 'val_stoi:', score_5_stoi / n_5)
                # print('n_5', n_5, 'score_5_pesq_gl:', score_5_pesq_gl, 'val_pesq_gl:', score_5_pesq_gl / n_5)
                # print('n_5', n_5, 'score_5_stoi_gl:', score_5_stoi_gl, 'val_stoi_gl:', score_5_stoi_gl / n_5)


