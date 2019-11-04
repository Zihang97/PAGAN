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
from utils import for_stft, for_stft_2
# from torchsummary import summary
from model_gan_multi_SN_ori import *
from torch_stft import STFT
# from tensorboardX import SummaryWriter

window_length = int(hp.data.window_gan * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
stft = STFT(filter_length=hp.data.nfft, hop_length=hop_length, win_length=window_length)


class dataset_preprocess_FTGAN(Dataset):
    def __init__(self, shuffle=False):
        self.clean_wav_npy = hp.data.gan_clean_wav_npy
        self.noisy_wav_npy = hp.data.gan_noisy_wav_npy
        self.clean_tf = hp.data.gan_clean_tf
        self.noisy_tf = hp.data.gan_noisy_tf
        self.noisy_phase = hp.data.gan_noisy_phase

        self.file_list = os.listdir(self.clean_wav_npy)
        self.shuffle = shuffle
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):

        selected_tf = self.file_list[item]

        clean_tf = np.load(os.path.join(self.clean_tf, selected_tf))
        noisy_tf = np.load(os.path.join(self.noisy_tf, selected_tf))

        clean_tf = torch.Tensor(clean_tf)
        noisy_tf = torch.Tensor(noisy_tf)


        return clean_tf, noisy_tf

def train():
    device_ids = [8, 9, 10, 11 , 12, 13, 14, 15]
    lr_factor = 2
    iteration = 0
    # writer = SummaryWriter('Gan_Loss_Log')
    train_dataset = dataset_preprocess_FTGAN()
    # num_loader < 2*12
    train_loader = DataLoader(train_dataset, batch_size=hp.train.batch_gan * len(device_ids), shuffle=True, num_workers=hp.train.num_workers, drop_last=True)
    G = VoiceFilter_SN()

    G = G.cuda(device_ids[0])

    G = torch.nn.DataParallel(G, device_ids=device_ids)

    G.eval()
    G.load_state_dict(torch.load("/workspace/model/rgan_base/multi_epoch_15.pth"))
    G.train()


    L1loss = nn.L1Loss()
    # optimizer = torch.optim.SGD(voicefilter_net.parameters(), lr=hp.train.lr * lr_factor, momentum=hp.train.momentum)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001*lr_factor)

    G.train()
    # D.train()

    for e in range(1000):

        total_loss_G_all = 0
        # (10, 257, 101), (10, 257, 101), (10, 16000), (10, 16000), (10, 257, 101)
        for num, (clean_tf, noisy_tf) in enumerate(train_loader):

            clean_tf_gpu = clean_tf.transpose(1, 2).cuda(device_ids[0])   # (10, 101, 257)
            noisy_tf_gpu = noisy_tf.transpose(1, 2).unsqueeze(1).cuda(device_ids[0])   # (10, 1, 101, 257)

        # -----------------------------------------------------
            optimizer_G.zero_grad()
            G_mask = G(noisy_tf_gpu)
            noisy_tf_gpu = torch.squeeze(noisy_tf_gpu)
            G_tf = G_mask * noisy_tf_gpu
            loss_G = 100 * L1loss(G_tf, clean_tf_gpu)
            loss_G.backward()
            optimizer_G.step()
            iteration = iteration + 1
            total_loss_G_all = total_loss_G_all + loss_G
            if (iteration % 50 == 0):
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss_G:{5:.5f}\tLoss_G_all_ave:{6:.5f}\t\n".format(
                    time.ctime(), e + 1, num, len(train_dataset) // (hp.train.batch_gan * len(device_ids)), iteration, loss_G, total_loss_G_all / num)
                print(mesg)

        mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss_G:{5:.5f}\tLoss_G_all_ave:{6:.5f}\t\n".format(
            time.ctime(), e + 1, num, len(train_dataset) // (hp.train.batch_gan * len(device_ids)), iteration, loss_G,
                          total_loss_G_all / num)
        print(mesg)
        if (e + 1) % 1 == 0:
            model_mid_name = 'multi_epoch_' + str(e + 1) + '.pth'
            G.eval()
            G.cpu()
            model_mid_path_G = os.path.join('/workspace/model/rgan_base', model_mid_name)
            torch.save(G.state_dict(), model_mid_path_G)
            G.cuda(device_ids[0]).train()

if __name__ == '__main__':
    train()