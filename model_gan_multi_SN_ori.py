import os
import time
import numpy as np
import random
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import DataLoader, Dataset
from hparam import hparam as hp
# from dvector_create import get_dvector_vf
from utils import for_stft
from torchsummary import summary
from torch_stft import STFT
from src.snlayers.snconv1d import SNConv1d
from src.snlayers.snlinear import SNLinear

window_length = int(hp.data.window_gan * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)

class VoiceFilter_SN(nn.Module):
    def __init__(self):
        super(VoiceFilter_SN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), padding=(0, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(6, 2), dilation=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(10, 2), dilation=(5, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(26, 2), dilation=(13, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False))

        self.LSTM1 = nn.LSTM(257 * 8, 400, num_layers=1, batch_first=True)
        for name, param in self.LSTM1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.FC3 = nn.Sequential(nn.Dropout(0.5, False), nn.Linear(400, 600), nn.ReLU())
        for name, param in self.FC3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.FC4 = nn.Sequential(nn.Dropout(0.5, False), nn.Linear(600, 257))
        for name, param in self.FC4.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        # self.loss1 = nn.L1Loss()
        self.stft = STFT(filter_length=hp.data.nfft, hop_length=hop_length, win_length=window_length)
        # self.loss2 = nn.BCELoss()

    def forward(self, noisy_tf_gpu):
        x = self.CNN(noisy_tf_gpu)
        x = x.transpose(1, 2).contiguous()
        # print('x1.shape:', x.shape)
        # x = x.view(x.size(0), x.size(1), -1)
        # print('x2.shape:', x.shape)
        x = x.view(-1, 101, 2056)
        out_real = x
        out_3, _ = self.LSTM1(out_real)
        out_3 = F.relu(out_3)
        out = self.FC3(out_3)
        out = self.FC4(out)
        G_mask = torch.sigmoid(out)

        return G_mask

class discriminator_SN(nn.Module):
    def __init__(self):
        super(discriminator_SN, self).__init__()
        self.CNN = nn.Sequential(
            SNConv1d(in_channels=2, out_channels=16, kernel_size=31, stride=2, padding=207),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=16, out_channels=32, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=32, out_channels=32, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=32, out_channels=64, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=64, out_channels=64, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=64, out_channels=128, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=128, out_channels=128, kernel_size=31, stride=2, padding=15),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=128, out_channels=256, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=256, out_channels=256, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=256, out_channels=512, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=512, out_channels=1024, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(0.3),
            SNConv1d(in_channels=1024, out_channels=1, kernel_size=1),
            nn.LeakyReLU(0.3)
        )
        self.FC1 = SNLinear(8, 1)
        # self.loss = nn.BCELoss()
        # self.LSTM1 = nn.LSTM(257 * 8, 400, num_layers=1, batch_first=True)
        # for name, param in self.LSTM1.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)
        #
        # self.FC3 = nn.Sequential(nn.Dropout(0.5, False), nn.Linear(400, 600), nn.ReLU())
        # for name, param in self.FC3.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)
        #
        # self.FC4 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(600, 257))
        # for name, param in self.FC4.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, a):
        # a = torch.cat((x, y), 0)
        # print('a.shape:', a.shape)
        # print('b.shape:', b.shape)
        x = self.CNN(a)
        x = x.squeeze(1)
        x = self.FC1(x)
        # y = self.CNN(b)
        # y = y.squeeze(1)
        # y = self.FC1(y)
        #
        # d_outD = F.sigmoid(x - y)
        # loss = self.loss(d_outD, real_vector)

        # x = x.transpose(1, 2).contiguous()
        # x = x.view(x.size(0), x.size(1), -1)
        # out_real = x
        # out_3, _ = self.LSTM1(out_real)
        # out_3 = F.relu(out_3)
        # out = self.FC3(out_3)
        # out = self.FC4(out)
        # out = torch.sigmoid(out)
        # print('x.shape:', x.shape)
        return (x)

if __name__ == '__main__':
    model = discriminator_SN()
    summary(model, (2, 16000), device='cpu')
    # model = VoiceFilter()
    # summary(model.cuda(), (1, 301, 257))
    # a = np.random.random((2, 16000))
    # a = torch.Tensor(a)
    # b = model(a).numpy()
    # print(b.shape())