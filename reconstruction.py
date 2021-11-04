import argparse
import os
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--vqgan_model_path', type=str, default = "pretrain_model/VQGANimagenet1024/vqgan.1024.model.ckpt",
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = "pretrain_model/VQGANimagenet1024/vqgan.1024.config.yml",
                   help='path to your trained VQGAN config. This should be a .yaml file.  (only valid when taming option is enabled)')

parser.add_argument('--image_folder', type = str, default = '../CUB_200_2011/CUB_200_2011/images',
                    help='path to your folder of images for learning the discrete VAE and its codebook')
# ../CUB_200_2011/CUB_200_2011/images

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--taming', dest='taming', action='store_true')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

vae_params = None

if args.taming:
    vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path).cuda()
elif vae_params is not None:
    vae = DiscreteVAE(**vae_params).cuda()
else:
    vae = OpenAIDiscreteVAE().cuda()

# generate images

image_size = vae.image_size
batch_size = args.batch_size


ds = ImageFolder(
    args.image_folder,
    T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
)

dl = DataLoader(ds, batch_size, shuffle = False, sampler=None)
amount = 0

for i, (images, _) in enumerate(dl):
    images = images.cuda()
    indices = vae.get_codebook_indices(images)
    rec = vae.decode(indices)

    outputs_dir = args.outputs_dir
    # os.mkdir(outputs_dir)
    img_save_dir = outputs_dir + '/' + str(i) + '.jpg'
    save_image(torch.cat((images, rec), 0), img_save_dir, normalize=True)
    amount += 1
print(f'created {amount} images at "{str(outputs_dir)}"')
