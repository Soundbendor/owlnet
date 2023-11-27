import librosa as LR
import torch
import torchaudio as TA
from macaulay_set import MCBSet
from torch.utils.data import DataLoader
import sndload as SL
from torch import cuda
import os
def get_data():
    ext_drive = os.path.join(os.path.abspath(os.sep), 'media', 'dxk', 'tosh_ext')
    train_ds = MCBSet(set_type="train", basefolder=ext_drive, tx="spect", max_ms=1000)
    test_ds = MCBSet(set_type="test",basefolder=ext_drive, tx="spect", max_ms=1000)
    return train_ds, test_ds


def runner(to_train = True):
    device = 'cpu'
    dstx, dste = get_data()
    if torch.cuda.is_available() == True:
        device = 'cuda'
    trainer(dstx)

def trainer_batch(epoch_idx, dloader):
    for batch_idx, (ci,cl) in enumerate(dloader):
        print(ci)
        print(cl)

def trainer(train_data, bs = 16, epochs = 1):
    tdload = DataLoader(train_data, shuffle=True, batch_size = bs)
    for batch_idx in range(epochs):
        trainer_batch(batch_idx, tdload)

def viewer(train_data, bs=16, idx=0):
    tdload = DataLoader(train_data, shuffle=False, batch_size = bs)
    tf, tl = next(iter(tdload))
    return tf[idx], tl[idx]

if __name__ == "__main__":
    _trds, _teds = get_data()
    curtf, ltf = viewer(_trds)
    curte, lte = viewer(_teds)
    

#runner()
