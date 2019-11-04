import glob
import os
import librosa
import numpy as np
import time

from hparam import hparam as hp

print('start to prepare data for voicefilter')

os.makedirs(hp.data.train_path_crn, exist_ok=True)
os.makedirs(hp.data.train_path_mixed_crn, exist_ok=True)
os.makedirs(hp.data.train_path_reference_crn, exist_ok=True)
os.makedirs(hp.data.train_path_clean_crn, exist_ok=True)
os.makedirs(hp.data.train_path_noise_crn, exist_ok=True)

os.makedirs(hp.data.test_path_crn, exist_ok=True)
os.makedirs(hp.data.test_path_mixed_crn, exist_ok=True)
os.makedirs(hp.data.test_path_reference_crn, exist_ok=True)
os.makedirs(hp.data.test_path_clean_crn, exist_ok=True)
os.makedirs(hp.data.test_path_noise_crn, exist_ok=True)

rms = lambda y: np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def change_audio(audio_clean):
    len_speech = len(audio_clean)
    if len_speech <= (hp.data.sr):
        n_repeat = int(np.ceil(float(hp.data.sr) / float(len_speech)))
        audio_clean_mi = np.tile(audio_clean, n_repeat)
        audio_clean = audio_clean_mi[0: hp.data.sr]
    else:
        clean_onset = rs.randint(0, len_speech - hp.data.sr, 1)[0]
        clean_offset = clean_onset + hp.data.sr
        audio_clean = audio_clean[clean_onset: clean_offset]
    return audio_clean
# print('aa:', aa)
audio_path = glob.glob(os.path.dirname(os.path.abspath(hp.unprocessed_data)))
# print('audio_path:', audio_path)
total_speaker_num  = len(audio_path)
print('total_speaker_num:', total_speaker_num)
train_speaker_num = (total_speaker_num//10)*9
print('train_speaker_num:', train_speaker_num)

rs = np.random.RandomState(0)
# print('aaaaaa')
# print('audio_path:', audio_path)

for i, speaker in enumerate(audio_path):
    # print('aaaaaa')
    utterance_name = []
    print('i:', i)
    time_0 = time.time()
    for audio_name in os.listdir(speaker):
        if audio_name[-4: ] == '.WAV':
            utterance_name.append(audio_name)
    for j, audio_name in enumerate(os.listdir(speaker)):
        if audio_name[-4: ] == '.WAV':
            utterance_path = os.path.join(speaker, audio_name)
            audio_clean, sr = librosa.load(utterance_path, sr=hp.data.sr)
            audio_clean = change_audio(audio_clean)
            len_speech = len(audio_clean)
            packet_audio = audio_clean
            if hp.data.need_noise:
                noise_names = []
                for noise_name in os.listdir(hp.data.noise_path_all):
                    noise_names.append(noise_name)
                rand_noise_names_20 = rs.choice(noise_names, hp.data.num_noise_utterance_20)
                rand_noise_names_15 = rs.choice(noise_names, hp.data.num_noise_utterance_15)
                rand_noise_names_10 = rs.choice(noise_names, hp.data.num_noise_utterance_10)
                rand_noise_names_5 = rs.choice(noise_names, hp.data.num_noise_utterance_5)
                rand_noise_names_0 = rs.choice(noise_names, hp.data.num_noise_utterance_0)
                rand_noise_names__5 = rs.choice(noise_names, hp.data.num_noise_utterance__5)
                for k, rand_noise_name in enumerate(rand_noise_names_20):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10.**(float(20)/20.)
                    clean_scaling = target_snr/original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy"%(i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_20.npy"%(i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy"%(i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                        
                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy" % (i-train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_20.npy" % (i-train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy" % (i-train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_20.npy" % (i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                for k, rand_noise_name in enumerate(rand_noise_names_15):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10. ** (float(15) / 20.)
                    clean_scaling = target_snr / original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_15.npy" % (i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)

                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_15.npy" % (
                                                          i - train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_15.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                for k, rand_noise_name in enumerate(rand_noise_names_10):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10. ** (float(10) / 20.)
                    clean_scaling = target_snr / original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_10.npy" % (i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (
                                                  i - train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_10.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (
                                                  i - train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_10.npy" % (
                                                  i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)

                for k, rand_noise_name in enumerate(rand_noise_names_5):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10. ** (float(5) / 20.)
                    clean_scaling = target_snr / original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_5.npy" % (i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)

                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_5.npy" % (
                                                          i - train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                    
                for k, rand_noise_name in enumerate(rand_noise_names_0):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10. ** (float(0) / 20.)
                    clean_scaling = target_snr / original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_0.npy" % (i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)

                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d_0.npy" % (
                                                          i - train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d_0.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
                    
                for k, rand_noise_name in enumerate(rand_noise_names__5):
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    if len_noise <= len_speech:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    target_snr = 10. ** (float(-5) / 20.)
                    clean_scaling = target_snr / original_snr
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    noise_audio_mix = noise_audio_mix * alpha
                    audio_reference_name = rs.choice(utterance_name, 1)[0]
                    audio_reference_path = os.path.join(speaker, audio_reference_name)
                    audio_reference, sr = librosa.load(audio_reference_path, sr=hp.data.sr)
                    if i < train_speaker_num:
                        clean_path = os.path.join(hp.data.train_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (i, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.train_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d__5.npy" % (i, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.train_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (i, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.train_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (i, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)
    
                    else:
                        clean_path = os.path.join(hp.data.test_path_clean_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(clean_path, audio_clean_mix, sr=hp.data.sr)
                        np.save(clean_path, audio_clean_mix)
                        reference_path = os.path.join(hp.data.test_path_reference_crn,
                                                      "speaker%d_utterance%d_noise%d__5.npy" % (
                                                          i - train_speaker_num, j, k))
                        # librosa.output.write_wav(reference_path, audio_reference, sr=hp.data.sr)
                        np.save(reference_path, audio_reference)
                        mixed_path = os.path.join(hp.data.test_path_mixed_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(mixed_path, audio_mix, sr=hp.data.sr)
                        np.save(mixed_path, audio_mix)
                        noise_path = os.path.join(hp.data.test_path_noise_crn,
                                                  "speaker%d_utterance%d_noise%d__5.npy" % (
                                                      i - train_speaker_num, j, k))
                        # librosa.output.write_wav(noise_path, noise_audio_mix, sr=hp.data.sr)
                        np.save(noise_path, noise_audio_mix)

    time_1 = time.time()
    time_del = time_1 - time_0
    mesg = "{0}\tfinished\t{1}speakers\tcost_time:{2}".format(time.ctime(), i, time_del)
    print(mesg)
