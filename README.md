# PAGAN
## Dependencies
   python3.0, pytorch=1.0.1, glob, librosa, numpy, pypesq, pystoi
## Prepare Dataset
* Download Timit dataset  
Get Timit dataset at https://github.com/philipperemy/timit, contains a total of 6300 sentences, 630 speakers, we used these as clean dataset.  
* Download noise dataset  
Get noise dataset at  http://web.cse.ohiostate.edu/pnl/corpus/HuNonspeech/ HuCorpus.html or  http://home.ustc.edu.cn/˜xuyong62/demo/115noises.html.
* Edit config.yaml  
```
cd config
vim config.yaml
```
* preprocess ctc_data file
```
python data_for_1s.py
python data_rgan.py
```
## Train/Test PAGAN
```
python train_rgan_ft_lps_multi_SN_real_ori.py
python test_PAGAN.py
```
## Train/Test PAGAN_base
```
python train_PAGAN_base.py
python test_PAGAN_base.py
```
## Train/Test rgan_ff
```
python train_rgan_ft_lps_multi_SN_real_ori_ff.py
python test_rgan_ff.py
```
