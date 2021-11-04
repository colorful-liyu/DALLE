from pathlib import Path
import time
from glob import glob
import os

import torch

from torch.utils.data import DataLoader
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, YttmTokenizer

# libraries needed for webdataset support
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

image_text_folder = "G:/dataset/CUB_birds/CUB_200_2011/images"
bpe_path = "dalle_pytorch/data/bpe_simple_vocab_16e6.txt"
TEXT_SEQ_LEN = 256
IMAGE_SIZE = 64
resize_ratio = 0.75
truncate_captions = True
is_shuffle = False

data_sampler = None
BATCH_SIZE = 1

ds = TextImageDataset(
        image_text_folder,
        text_len=TEXT_SEQ_LEN,
        image_size=IMAGE_SIZE,
        resize_ratio=resize_ratio,
        truncate_captions=truncate_captions,
        tokenizer=tokenizer,
        shuffle=is_shuffle,
    )

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

print('prepare to check dataset')

for i, (text, images) in enumerate(dl):
    print('='*15)
    print('idx is', i)
    print('the text is ', text)
    print('the images shape is ', images.shape)