import librosa
from pypesq import pesq
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparam import hparam as hp
from pystoi.stoi import stoi
# rms = lambda y: np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))
n = 0
n_20 = 0
n_10 = 0
n_15 = 0
n_5 = 0
n_0 = 0
n__5 = 0
score_20 = 0
score_10 = 0
score_15 = 0
score_0 = 0
score_5 = 0
score__5 = 0
window_length = int(hp.data.window * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
# clean_path = '/home/lps/voicefilter/voicefilter1/test_data_all/clean/'
clean_audio_path = '/home/lps/voicefilter/voicefilter1/test_all/clean/'
mix_audio_path = '/home/lps/voicefilter/voicefilter1/test_all/mixed/'
mix_path = '/home/lps/voicefilter/voicefilter1/test_data_all/mixed/'
reference_path = '/home/lps/voicefilter/voicefilter1/test_data_all/reference/'
noise_path = '/home/lps/voicefilter/voicefilter1/test_data_all/noise/'

def gri_lim_1(stft_amg, mix, n_fft, hop_length, win_length):
    # print("stft_amg:", stft_amg.shape)
    shape = (stft_amg.shape[1]-1) * hop_length
    x = mix
    for i in range(100):
        x_stft = librosa.stft(x, n_fft, hop_length, win_length)
        x = librosa.istft(stft_amg * (x_stft/np.abs(x_stft)), hop_length, win_length)
    return x


class VoiceFilter(nn.Module):
    def __init__(self):
        super(VoiceFilter, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), padding=(0, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(6, 2), dilation=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(10, 2), dilation=(5, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(26, 2), dilation=(13, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        # for name, param in self.CNN.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)
        # self.CNN1 = nn.Conv2d(1, 64, 1, 1)
        
        self.LSTM1 = nn.LSTM(257 * 8 + 256, 257, num_layers=1, batch_first=True)
        for name, param in self.LSTM1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.LSTM2 = nn.LSTM(257 * 8 + 256 + 257, 400, num_layers=1, batch_first=True)
        for name, param in self.LSTM1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        # self.FC1 = nn.Linear(256, 257)
        # for name, param in self.FC1.named_parameters():
        #   if 'bias' in name:
        #      nn.init.constant_(param, 0.0)
        #   elif 'weight' in name:
        #      nn.init.xavier_normal_(param)
        # self.FC2 = nn.Linear(600, 257)
        self.FC1 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(257, 1), nn.ReLU())
        for name, param in self.FC1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.FC3 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(400, 600), nn.ReLU())
        for name, param in self.FC3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        self.FC4 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(600, 257))
        for name, param in self.FC4.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x, y):
        x = self.CNN(x)
        
        # x = torch.mean(x, 1)
        # x, _ = self.LSTM1(x, )
        # x = self.FC1(x)
        # x = self.FC2(x)
        # y = self.FC1(y)
        # y = self.FC2(y)
        # y = y.view(-1, 1, 1, 257)
        # x = x.view(-1, 301, 257)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        # y = y.unsqueeze(1)
        y = y.repeat(1, x.size(1), 1)
        x_y = torch.cat((x, y), 2)
        # score = F.softmax(torch.bmm(x_y, x_y.transpose(1, 2)), dim=2)
        # out_1 = torch.bmm(score, x_y)
        # score_1 = F.softmax(torch.bmm(x_y.transpose(1, 2), x_y), dim=1)
        # out_2 = torch.bmm(x_y, score_1)
        # out_real = out_1 + out_2
        out_3, _ = self.LSTM1(x_y)
        out_3 = F.relu(out_3)
        x_vad = self.FC1(out_3)
        x_y_1 = torch.cat((x_y, out_3), 2)
        out_4, _ = self.LSTM2(x_y_1)
        out_4 = F.relu(out_4)
        out = self.FC3(out_4)
        out = self.FC4(out)
        out = torch.sigmoid(out)
        # out = torch.sigmoid(out)
        # print("out_x.shape:", x.shape)
        # print("y_in_shape:", y.shape)
        # print("x_y.shape:", x_y.shape)
        # print("score.shape:", score.shape)
        # print("out.shape:", out.shape)
        return (out, x_vad)


np_file_list = os.listdir(noise_path)
# device = torch.device(hp.device)
voicefilter_net = VoiceFilter()
voicefilter_net.load_state_dict(torch.load("/home/lps/voicefilter/voicefilter1/model_vf_all_vad_sigmoid/epoch_20.pth"))
voicefilter_net.eval()
vf_loss = nn.MSELoss()

with torch.no_grad():
    for item in np_file_list:
        item_name = item.split(".")[0]
        som_th = item_name[-2:]
        # print('som_th:', som_th)
        if (som_th == '20'):
            n_20 = n_20 + 1
            clean_name = "%s.wav"%item_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
            clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            mix_audio,_ = librosa.load(mix_audio_path_real, sr=16000)
            mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_angle = np.angle(mix_tf).T
            
            mix_tf_name = "%s.npy"%item_name
            mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # noise_tf_name = "%s.npy"%item_name
            # noise_tf_path = os.path.join(noise_path, noise_tf_name)
            reference_name = "%s.npy"%item_name
            reference_tf_path = os.path.join(reference_path, reference_name)
            
            # utters_noise = np.load(noise_tf_path)
            utters_mix = np.load(mix_tf_path)
            utters_reference = np.load(reference_tf_path)
            
            # noise = torch.Tensor(utters_noise)
            mix = torch.Tensor(utters_mix)
            d_vector = torch.Tensor(utters_reference)
            # clean = mix - noise
            mix = mix.view(1, 1, -1, 257)
            # mix = mix.to(device)
            # d_vector = d_vector.to(device)
            # clean = clean.to(device)
            # vad = vad.to(device)
            out, out_vad = voicefilter_net(mix, d_vector)
            
            # clean = clean.view(1, -1, 257)
            
            aaaa = mix * out
            if(torch.max(aaaa)>0.9):
                n = n + 1
                aaaa = aaaa*(0.9/torch.max(aaaa))
                # print('n:', n)
            
            # loss_20 = loss_20 + vf_loss(aaaa, clean)
            aaa = np.zeros((1, 257))
            som_angle = np.concatenate((mix_angle, aaa), axis=0).T
            out_something = aaaa.reshape(301, 257).detach().numpy().T
            out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
            # score = pesq(clean_audio, out, 16000)
            score = stoi(clean_audio, out, 16000)
            score_20 = score_20 + score
            print('n_20:', n_20, 'score_20:', score_20, 'val:', score_20 / n_20)
            
        elif (som_th == '10'):
            n_10 = n_10 + 1
            clean_name = "%s.wav" % item_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
            clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
            mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_angle = np.angle(mix_tf).T
        
            mix_tf_name = "%s.npy" % item_name
            mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # noise_tf_name = "%s.npy"%item_name
            # noise_tf_path = os.path.join(noise_path, noise_tf_name)
            reference_name = "%s.npy" % item_name
            reference_tf_path = os.path.join(reference_path, reference_name)
        
            # utters_noise = np.load(noise_tf_path)
            utters_mix = np.load(mix_tf_path)
            utters_reference = np.load(reference_tf_path)
        
            # noise = torch.Tensor(utters_noise)
            mix = torch.Tensor(utters_mix)
            d_vector = torch.Tensor(utters_reference)
            # clean = mix - noise
            mix = mix.view(1, 1, -1, 257)
            # mix = mix.to(device)
            # d_vector = d_vector.to(device)
            # clean = clean.to(device)
            # vad = vad.to(device)
            out, out_vad = voicefilter_net(mix, d_vector)
        
            # clean = clean.view(1, -1, 257)
        
            aaaa = mix * out
            if (torch.max(aaaa) > 0.9):
                n = n + 1
                aaaa = aaaa * (0.9 / torch.max(aaaa))
                # print('n:', n)
        
            # loss_20 = loss_20 + vf_loss(aaaa, clean)
            aaa = np.zeros((1, 257))
            som_angle = np.concatenate((mix_angle, aaa), axis=0).T
            out_something = aaaa.reshape(301, 257).detach().numpy().T
            out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
            # score = pesq(clean_audio, out, 16000)
            score = stoi(clean_audio, out, 16000)
            score_10 = score_10 + score
            print('n_10:', n_10, 'score_10:', score_10, 'val:', score_10 / n_10)
        elif(som_th == '_0'):
            n_0 = n_0 + 1
            clean_name = "%s.wav" % item_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
            clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
            mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_angle = np.angle(mix_tf).T
        
            mix_tf_name = "%s.npy" % item_name
            mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # noise_tf_name = "%s.npy"%item_name
            # noise_tf_path = os.path.join(noise_path, noise_tf_name)
            reference_name = "%s.npy" % item_name
            reference_tf_path = os.path.join(reference_path, reference_name)
        
            # utters_noise = np.load(noise_tf_path)
            utters_mix = np.load(mix_tf_path)
            utters_reference = np.load(reference_tf_path)
        
            # noise = torch.Tensor(utters_noise)
            mix = torch.Tensor(utters_mix)
            d_vector = torch.Tensor(utters_reference)
            # clean = mix - noise
            mix = mix.view(1, 1, -1, 257)
            # mix = mix.to(device)
            # d_vector = d_vector.to(device)
            # clean = clean.to(device)
            # vad = vad.to(device)
            out, out_vad = voicefilter_net(mix, d_vector)
        
            # clean = clean.view(1, -1, 257)
        
            aaaa = mix * out
            if (torch.max(aaaa) > 0.9):
                n = n + 1
                aaaa = aaaa * (0.9 / torch.max(aaaa))
                # print('n:', n)
        
            # loss_20 = loss_20 + vf_loss(aaaa, clean)
            aaa = np.zeros((1, 257))
            som_angle = np.concatenate((mix_angle, aaa), axis=0).T
            out_something = aaaa.reshape(301, 257).detach().numpy().T
            out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
            # score = pesq(clean_audio, out, 16000)
            score = stoi(clean_audio, out, 16000)
            score_0 = score_0 + score
            print('n_0:', n_0, 'score_0:', score_0, 'val:', score_0 / n_0)
        elif(som_th == '15'):
            n_15 = n_15 + 1
            clean_name = "%s.wav" % item_name
            clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
            mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
            clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
            mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
            mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_angle = np.angle(mix_tf).T
        
            mix_tf_name = "%s.npy" % item_name
            mix_tf_path = os.path.join(mix_path, mix_tf_name)
            # noise_tf_name = "%s.npy"%item_name
            # noise_tf_path = os.path.join(noise_path, noise_tf_name)
            reference_name = "%s.npy" % item_name
            reference_tf_path = os.path.join(reference_path, reference_name)
        
            # utters_noise = np.load(noise_tf_path)
            utters_mix = np.load(mix_tf_path)
            utters_reference = np.load(reference_tf_path)
        
            # noise = torch.Tensor(utters_noise)
            mix = torch.Tensor(utters_mix)
            d_vector = torch.Tensor(utters_reference)
            # clean = mix - noise
            mix = mix.view(1, 1, -1, 257)
            # mix = mix.to(device)
            # d_vector = d_vector.to(device)
            # clean = clean.to(device)
            # vad = vad.to(device)
            out, out_vad = voicefilter_net(mix, d_vector)
        
            # clean = clean.view(1, -1, 257)
        
            aaaa = mix * out
            if (torch.max(aaaa) > 0.9):
                n = n + 1
                aaaa = aaaa * (0.9 / torch.max(aaaa))
                # print('n:', n)
        
            # loss_20 = loss_20 + vf_loss(aaaa, clean)
            aaa = np.zeros((1, 257))
            som_angle = np.concatenate((mix_angle, aaa), axis=0).T
            out_something = aaaa.reshape(301, 257).detach().numpy().T
            out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
            # score = pesq(clean_audio, out, 16000)
            score = stoi(clean_audio, out, 16000)
            score_15 = score_15 + score
            print('n_15:', n_15, 'score_15:', score_15, 'val:', score_15 / n_15)
        elif(som_th == '_5'):
            som_th_2 = item_name[-3:]
            if(som_th_2 == '__5'):
                n__5 = n__5 + 1
                clean_name = "%s.wav" % item_name
                clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
                mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
                clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
                mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
                mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
                mix_angle = np.angle(mix_tf).T
        
                mix_tf_name = "%s.npy" % item_name
                mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # noise_tf_name = "%s.npy"%item_name
                # noise_tf_path = os.path.join(noise_path, noise_tf_name)
                reference_name = "%s.npy" % item_name
                reference_tf_path = os.path.join(reference_path, reference_name)
        
                # utters_noise = np.load(noise_tf_path)
                utters_mix = np.load(mix_tf_path)
                utters_reference = np.load(reference_tf_path)
        
                # noise = torch.Tensor(utters_noise)
                mix = torch.Tensor(utters_mix)
                d_vector = torch.Tensor(utters_reference)
                # clean = mix - noise
                mix = mix.view(1, 1, -1, 257)
                # mix = mix.to(device)
                # d_vector = d_vector.to(device)
                # clean = clean.to(device)
                # vad = vad.to(device)
                out, out_vad = voicefilter_net(mix, d_vector)
        
                # clean = clean.view(1, -1, 257)
        
                aaaa = mix * out
                if (torch.max(aaaa) > 0.9):
                    n = n + 1
                    aaaa = aaaa * (0.9 / torch.max(aaaa))
                    # print('n:', n)
        
                # loss_20 = loss_20 + vf_loss(aaaa, clean)
                aaa = np.zeros((1, 257))
                som_angle = np.concatenate((mix_angle, aaa), axis=0).T
                out_something = aaaa.reshape(301, 257).detach().numpy().T
                out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
                # score = pesq(clean_audio, out, 16000)
                score = stoi(clean_audio, out, 16000)
                score__5 = score__5 + score
                print('n__5:', n__5, 'score__5:', score__5, 'val:', score__5 / n__5)
            else:
                n_5 = n_5 + 1
                clean_name = "%s.wav" % item_name
                clean_audio_path_real = os.path.join(clean_audio_path, clean_name)
                mix_audio_path_real = os.path.join(mix_audio_path, clean_name)
                clean_audio, _ = librosa.load(clean_audio_path_real, sr=16000)
                mix_audio, _ = librosa.load(mix_audio_path_real, sr=16000)
                mix_tf = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
                mix_angle = np.angle(mix_tf).T
        
                mix_tf_name = "%s.npy" % item_name
                mix_tf_path = os.path.join(mix_path, mix_tf_name)
                # noise_tf_name = "%s.npy"%item_name
                # noise_tf_path = os.path.join(noise_path, noise_tf_name)
                reference_name = "%s.npy" % item_name
                reference_tf_path = os.path.join(reference_path, reference_name)
        
                # utters_noise = np.load(noise_tf_path)
                utters_mix = np.load(mix_tf_path)
                utters_reference = np.load(reference_tf_path)
        
                # noise = torch.Tensor(utters_noise)
                mix = torch.Tensor(utters_mix)
                d_vector = torch.Tensor(utters_reference)
                # clean = mix - noise
                mix = mix.view(1, 1, -1, 257)
                # mix = mix.to(device)
                # d_vector = d_vector.to(device)
                # clean = clean.to(device)
                # vad = vad.to(device)
                out, out_vad = voicefilter_net(mix, d_vector)
        
                # clean = clean.view(1, -1, 257)
        
                aaaa = mix * out
                if (torch.max(aaaa) > 0.9):
                    n = n + 1
                    aaaa = aaaa * (0.9 / torch.max(aaaa))
                    # print('n:', n)
        
                # loss_20 = loss_20 + vf_loss(aaaa, clean)
                aaa = np.zeros((1, 257))
                som_angle = np.concatenate((mix_angle, aaa), axis=0).T
                out_something = aaaa.reshape(301, 257).detach().numpy().T
                out = gri_lim_1(out_something, mix_audio, hp.data.nfft, hop_length, window_length)
                # score = pesq(clean_audio, out, 16000)
                score = stoi(clean_audio, out, 16000)
                score_5 = score_5 + score
                print('n_5:', n_5, 'score_5:', score_5, 'val:', score_5 / n_5)
        else:
            print('something is wrong')
print('n_20:', n_20, 'score_20:', score_20, 'val:', score_20/n_20)
print('n_15:', n_15, 'score_15:', score_15, 'val:', score_15/n_15)
print('n_10:', n_10, 'score_10:', score_10, 'val:', score_10/n_10)
print('n_20:', n_5, 'score_5:', score_5, 'val:', score_5/n_5)
print('n_0:', n_0, 'score_0:', score_0, 'val:', score_0/n_0)
print('n__5:', n__5, 'score__5:', score__5, 'val:', score__5/n__5)





# ref, _ = librosa.load('/home/lps/voicefilter/voicefilter1/691/reference.wav', sr=16000)
# # mix, _ = librosa.load('/home/lps/voicefilter/voicefilter1/source_3/lps_694_-5db_50.wav', sr=16000)
# mix, _ = librosa.load('/home/lps/voicefilter/voicefilter/source_1/lps_691_-10db_120.wav', sr=16000)
# # mix, _ = librosa.load('/home/lps/voicefilter/voicefilter1/694/4_4.wav', sr=16000)
# # snr = 20*(np.log10(rms(ref) / rms(mix)))
# score = pesq(ref, mix, 16000)
# print(score)
