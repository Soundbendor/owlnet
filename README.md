# owlnet
Owl detection/classification project

- `adboli.py` - implementation of architectures from Abdoli, S., Cardinal, P., Koerich, A.L. (2019). End-to-End Environmental Sound Classification using a 1D Convolutional Neural Network. https://arxiv.org/abs/1904.08990
- `birdset.py` - start at a dataloader for soundfiles
- `sndload.py` - sound loading helper
- `soundsplit.py` - split mp3 files into 10 second chunks using `mp3splt`
- `macaulay_get.py` - use the csv files (see `datasets` on server) for the macaulay dataset to get the actual mp3 files
- `main.py` - hacky barebones start at a training loop

