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
from model_gan_multi_SN_ori_ff import *
from torch_stft import STFT
# from tensorboardX import SummaryWriter

window_length = int(hp.data.window_gan * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
stft = STFT(filter_length=hp.data.nfft, hop_length=hop_length, win_length=window_length)


class dataset_preprocess_FTGAN(Dataset):
    def __init__(self, shuffle=False):
        self.clean_tf = hp.data.gan_clean_tf
        self.noisy_tf = hp.data.gan_noisy_tf
        # self.clean_wav = hp.data.gan_clean_wav
        # self.noisy_wav = hp.data.gan_noisy_wav

        # self.file_list = os.listdir(self.clean_wav)
        self.file_list = os.listdir(self.clean_tf)
        self.shuffle = shuffle
        # self.utter_start = utter_start
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        # np_file_list = os.listdir(self.clean_wav_npy)
        # if self.shuffle:
        #     selected_file = random.sample(np_file_list, 1)[0]
        # else:
        selected_tf = self.file_list[item]
        # selected_name = selected_wav.split(".")[0]
        # selected_tf = "%s.npy" % selected_name

        # clean_wav, _ = librosa.load(os.path.join(self.clean_wav, selected_wav), sr=16000)
        # noisy_wav, _ = librosa.load(os.path.join(self.noisy_wav, selected_wav), sr=16000)

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
    D = discriminator_SN_ff()
    # G.eval()
    # D.eval()
    # G.load_state_dict(torch.load("/data2/lps/model_gan/G/epoch_4.pth"))
    # D.load_state_dict(torch.load("/data2/lps/model_gan/D/epoch_4.pth"))
    # save_model_G = torch.load("/data2/lps/model_gan/G/multi_epoch_6.pth")
    # save_model_D = torch.load("/data2/lps/model_gan/D/multi_epoch_6.pth")
    # model_dict_G = G.state_dict()
    # model_dict_D = D.state_dict()
    # state_dict_G = {k:v for k,v in save_model_G.items() if k in model_dict_G.keys()}
    # state_dict_D = {k:v for k,v in save_model_D.items() if k in model_dict_D.keys()}
    # model_dict_G.update(state_dict_G)
    # model_dict_D.update(state_dict_D)
    # G.load_state_dict(model_dict_G)
    # D.load_state_dict(model_dict_D)
    # stft = STFT(filter_length=hp.data.nfft, hop_length=hop_length, win_length=window_length).cuda(device_ids[0])
    G = G.cuda(device_ids[0])
    D = D.cuda(device_ids[0])
    G = torch.nn.DataParallel(G, device_ids=device_ids)
    D = torch.nn.DataParallel(D, device_ids=device_ids)

    BCE_loss_fn = nn.BCELoss()
    L1loss = nn.L1Loss()
    # optimizer = torch.optim.SGD(voicefilter_net.parameters(), lr=hp.train.lr * lr_factor, momentum=hp.train.momentum)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001*lr_factor)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0004*lr_factor)
    G.train()
    D.train()

    real_vector = torch.ones(hp.train.batch_gan * len(device_ids), 1).cuda(device_ids[0])
    fake_vector = torch.zeros(hp.train.batch_gan * len(device_ids), 1).cuda(device_ids[0])
    # a = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # b = a.transpose(1, 2)
    # c = a.permute(0, 2, 1)
    # d = a.view(2, 6)
    # print('a:', a)
    # print('b:', b)
    # print('c:', c)
    # print('d:' d) # 执行的是按照数字的顺序将数组变成所想要的形状的数组
    for e in range(1000):
        # print('e:', e)
        total_loss_L1 = 0
        total_loss_G = 0
        total_loss_D = 0
        total_loss_G_all = 0
        # (10, 257, 101), (10, 257, 101), (10, 16000), (10, 16000), (10, 257, 101)
        for num, (clean_tf, noisy_tf) in enumerate(train_loader):
            # print('num:', num)
            # print('noisy_phase.size:', noisy_phase.shape)
            # clean_wav_gpu = clean_wav.unsqueeze(1).cuda(device_ids[0])    # (10, 1, 16000)
            # noisy_wav_gpu = noisy_wav.unsqueeze(1).cuda(device_ids[0])    # (10, 1, 16000)
            clean_tf_gpu = clean_tf.transpose(1, 2).cuda(device_ids[0])   # (10, 101, 257)
            noisy_tf_gpu = noisy_tf.transpose(1, 2).unsqueeze(1).cuda(device_ids[0])   # (10, 1, 101, 257)
            # noisy_phase_gpu = noisy_phase.cuda(device_ids[0])    # (10, 257, 101)
            # print('nosiy_phase_gpu.size:', noisy_phase_gpu.shape)
            # print('clean_wav_gpu.size:', clean_wav_gpu.shape)
            # print('noisy_wav_gpu.size:', noisy_wav_gpu.shape)
        # -----------------------------------------------------
            optimizer_G.zero_grad()
            G_mask = G(noisy_tf_gpu)
            noisy_tf_gpu = torch.squeeze(noisy_tf_gpu)
            G_tf = G_mask * noisy_tf_gpu
            loss_G_1 = L1loss(G_tf, clean_tf_gpu)
            # G_tf_1 = G_tf.transpose(1, 2)
            # print('G_tf_1.shape:', G_tf_1.shape, 'noisy_phase_gpu.shape:', noisy_phase_gpu.shape)
            # G_wav_gpu = stft.inverse(G_tf_1, noisy_phase_gpu)
            G_tf = G_tf.unsqueeze(1)
            noisy_tf_gpu = noisy_tf_gpu.unsqueeze(1)
            clean_tf_gpu = clean_tf_gpu.unsqueeze(1)
            G_wav_D = torch.cat((G_tf, noisy_tf_gpu), dim=1)
            D_clean = torch.cat((clean_tf_gpu, noisy_tf_gpu), dim=1)
            in_1 = D(G_wav_D)
            in_2 = D(D_clean)
            in_3 = F.sigmoid(in_1 - in_2)
            # print('in_3.shape:', in_3.shape)
            # print('real_cevtor:', real_vector.shape)
            loss_G_2 = BCE_loss_fn(in_3, real_vector)
            loss_G = 100*loss_G_1 + loss_G_2
            loss_G.backward()
            optimizer_G.step()
            #    Train Discriminator


            optimizer_D.zero_grad()
            G_wav_gpu_1 = G_tf.detach()
            G_wav_D_1 = torch.cat((G_wav_gpu_1, noisy_tf_gpu), dim=1)
            in_4 = D(G_wav_D_1)
            in_5 = D(D_clean)
            in_6 = F.sigmoid(in_5 - in_4)
            in_7 = F.sigmoid(in_4 - in_5)
            # loss_D = D(D_clean, G_wav_D_1, real_vector)
            loss_D_real = BCE_loss_fn(in_6, real_vector)
            loss_D_fake = BCE_loss_fn(in_7, fake_vector)
            loss_D = (loss_D_real + loss_D_fake)/2
            loss_D.backward()
            optimizer_D.step()
            # print('loss_G:', loss_G, 'loss_D:', loss_D)
            # print('finish the first')
        # -------------------------------------------------
            iteration = iteration + 1
            total_loss_G = total_loss_G + loss_G_2
            total_loss_L1 = total_loss_L1 + loss_G_1
            total_loss_D = total_loss_D + loss_D
            total_loss_G_all = total_loss_G_all + loss_G
            # if (iteration % 10 == 0):
            #     niter = hp.train.batch_gan*len(device_ids)*len(train_loader) + iteration
            #     writer.add_scalar('Train/G_1', loss_G_1, niter)
            #     writer.add_scalar('Train/G_2', loss_G_2, niter)
            #     writer.add_scalar('Train/G', loss_G, niter)
            #     writer.add_scalar('Train/D', loss_D, niter)
            if (iteration % 50 == 0):
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss_L1:{5:.5f}\tLoss_G:{6:.5f}\tLoss_D:{7:.5f}\tLoss_G_all:{8:.5f}\tL1_loss_ave:{9:.6f}\tG_loss_ave:{10:.6f}\tD_loss_ave:{11:.6f}\tG_all_loss_ave:{12:.6f}\t\n".format(
                    time.ctime(), e + 1, num, len(train_dataset) // (hp.train.batch_gan * len(device_ids)), iteration, loss_G_1, loss_G_2, loss_D, loss_G, total_loss_L1 / num, total_loss_G / num, total_loss_D / num, total_loss_G_all / num)
                print(mesg)

        mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss_L1:{5:.5f}\tLoss_G:{6:.5f}\tLoss_D:{7:.5f}\tLoss_G_all:{8:.5f}\tL1_loss_ave:{9:.6f}\tG_loss_ave:{10:.6f}\tD_loss_ave:{11:.6f}\tG_all_loss_ave:{12:.6f}\t\n".format(
            time.ctime(), e + 1, num, len(train_dataset) // (hp.train.batch_gan * len(device_ids)), iteration,
            loss_G_1, loss_G_2, loss_D, loss_G, total_loss_L1 / num, total_loss_G / num, total_loss_D / num,
                          total_loss_G_all / num)
        print(mesg)
        # print('aaaaaa')
        if (e + 1) % 1 == 0:
            model_mid_name = 'multi_epoch_' + str(e + 1) + '.pth'
            G.eval()
            G.cpu()
            model_mid_path_G = os.path.join('/workspace/model/rgan_SN_ff/G', model_mid_name)
            torch.save(G.state_dict(), model_mid_path_G)
            D.eval()
            D.cpu()
            model_mid_path_D = os.path.join('/workspace/model/rgan_SN_ff/D', model_mid_name)
            torch.save(D.state_dict(), model_mid_path_D)
            G.cuda(device_ids[0]).train()
            D.cuda(device_ids[0]).train()
        # if (e + 1) % 10 == 0:
        #     lr_factor = lr_factor/2
        #     print("iteration:", iteration)

if __name__ == '__main__':
    train()