import torch
import argparse
import cv2
import numpy as np
import time
import os
import utils.custom_transforms as transformer
from utils.patch_transformer import Adversarial_Patch
from models.adversarial_models import AdversarialModels
from utils.dataloader import LoadFromImageFile
from utils.utils import makedirs, to_cuda_vars, format_time
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='Src/input_img')
parser.add_argument('--train_list', type=str, default='Src/list/test_list_orig.txt')

parser.add_argument('--encoder_path', type=str, default="mono_1024x320/encoder.pth")
parser.add_argument('--decoder_path', type=str, default="mono_1024x320/depth.pth")

parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=2)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=1)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--model', nargs='*', type=str, default='monodepth2', choices=['distill', 'monodepth2'], help='Model architecture')
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    train_transform = transformer.Compose([
        transformer.ResizeImage(h=args.height, w=args.width),
        transformer.ArrayToTensor()
    ])

    train_set = LoadFromImageFile(
        args.data_root,
        args.train_list,
        seed=args.seed,
        train=True,
        monocular=True,
        transform=train_transform,
        extension=".jpg"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True,
        drop_last=True
    )

    print('===============================')
    # Attacked Models
    models = AdversarialModels(args)
    models.load_weights()

    # Train
    for epoch in range(args.num_epochs):
        for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader), leave=False):
            with torch.autograd.detect_anomaly():
                sample = to_cuda_vars(sample)  # send item to gpu
                sample.update(models.get_original_disp(sample))  # get non-attacked disparity

                img, original_disp = sample['left'], sample['original_distill_disp']

                if 'monodepth2' in args.model:
                    est_disp = models.distill(img)
                    #print(f"img: {img}")
                    #print(f"est_disp: {est_disp}")
                    #print(f"original_disp: {original_disp}")
                    #print(f"Shapes: {img.shape, est_disp.shape, original_disp.shape}")


if __name__ == '__main__':
    main()
