import numpy as np
import torch
import os
import sndload as SL
from torch.utils.data import Dataset
import torchaudio.transforms as TXT
import sklearn.model_selection as SMS
from sklearn.preprocessing import OneHotEncoder

class MCBSet(Dataset):
    def __init__(self, datafolder="macaulay_out", max_ms = 99999, srate = 44100, basefolder = os.path.split(__file__)[0], set_type = "train", test_size = 0.2, tx = None, n_fft = 400, random_state = 3):
        curdir = os.path.split(__file__)[0]
        curpath = os.path.join(basefolder, datafolder)
        folders = [x for x in os.listdir(curpath) if os.path.isdir(os.path.join(curpath, x)) == True]
        sndpaths = []
        sndlabels = []
        
        for folder in folders:

            sndfolder = os.path.join(curpath, folder)
            cur_snds = [os.path.join(sndfolder,x) for x in os.listdir(sndfolder) if ".mp3" in x]
            cur_len = len(cur_snds)
            cur_labels = [folder] * cur_len
            sndpaths += cur_snds
            sndlabels += cur_labels
        self.enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        self.enc.fit(np.expand_dims(np.array(sndlabels),1))
        #print(sndpaths)
        s3 = SMS.StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
        spl = s3.split(sndpaths, sndlabels)
        cur_ds = list(spl)[0]
        train_idx = cur_ds[0]
        test_idx = cur_ds[1]
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.num_train = len(train_idx)
        self.num_test = len(test_idx)
        self.set_type = set_type
        max_samp = max_ms * 0.001 * srate
        self.fullpaths = sndpaths
        self.fulllabels = sndlabels
        self.max_ms = max_ms
        self.srate = srate
        self.tx = tx
        self.n_fft = n_fft
        self.max_samp = max_samp
        self.spect = None

        if self.tx == "spect":
            self.spect = TXT.Spectrogram(self.n_fft)
        if set_type == "train":
            self.cur_idx = self.train_idx
        else:
            self.cur_idx = self.test_idx

    def __len__(self):
        if self.set_type == "train":
            return self.num_train
        else:
            return self.num_test

    def __getitem__(self, idx):
        # return sound, label
        mapidx = self.cur_idx[idx]
        curpath = self.fullpaths[mapidx]
        curlabel = self.fulllabels[mapidx]
        retlabel = self.enc.transform([[curlabel]])
        retsnd = SL.sndloader(curpath, want_sr=None, want_bits=None, max_samp = self.max_samp, to_mono=True)
        if self.tx == "spect":
            retsnd = self.spect(retsnd)
        #print(retsnd, curlabel)
        return retsnd, retlabel

