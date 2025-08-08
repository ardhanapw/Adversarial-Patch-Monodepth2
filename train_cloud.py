import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

import comet_ml
import json

import argparse
import cv2
import numpy as np
import time
import os

import matplotlib.pyplot as plt

import utils.custom_transforms as transformer
from utils.patch_transformer import Adversarial_Patch
from models.adversarial_models import AdversarialModels
from utils.dataloader import LoadFromImageFile
from utils.utils import makedirs, to_cuda_vars, format_time
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='Src/left_images')
parser.add_argument('--train_list', type=str, default='Src/list/train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="models/guo/distill_model.ckpt")

parser.add_argument('--encoder_path', type=str, default="mono_1024x320/encoder.pth")
parser.add_argument('--decoder_path', type=str, default="mono_1024x320/depth.pth")

parser.add_argument('--height', type=int, help='input image height', default=320)
parser.add_argument('--width', type=int, help='input image width', default=1024)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=16)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=4)
#parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--lr', type=float, help='initial learning rate', default=3e-2)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=80)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=256)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='square')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file')
parser.add_argument('--mask_path', type=str, help='Initialize mask from file')
parser.add_argument('--export_adv_patch_path', type=str, help='Export generated adv patch and its mask', default='Dst/checkpoints')
parser.add_argument('--log_path', type=str, help='Tensorboard logging destination', default='Dst/runs/train_adversarial')
parser.add_argument('--log_interval', type=int, help='Tensorboard batch log interval', default=2000) #log every 2000 image

#this target_disp is for Guo et al. (2018) model, different disparity-depth conversion
#parser.add_argument('--target_disp', type=int, default=120)
#target_disp for Monodepth2 is depth = 1/target_disp, target_disp = softmaxxed output from the model
parser.add_argument('--target_disp', type=float, default=1/120)

parser.add_argument('--model', nargs='*', type=str, default='monodepth2', choices=['distill', 'monodepth2'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()

"""
def visualize_depth(img, disp):
    img_np = img.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))

    if img_np.dtype == np.float32 or img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    disp_np = disp.squeeze().detach().cpu().numpy()
    disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min())

    disp_uint8 = (disp_norm * 255).astype(np.uint8)


    disp_colored = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_MAGMA)
    stacked = np.vstack((original_bgr, disp_colored))

    cv2.imshow("Disparity", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def disp_to_depth(disp, min_depth, max_depth):
    #monodepth inference result (disparity) to depth
    #refer to the paper
    
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
"""

def init_tensorboard(log_path):
    return SummaryWriter(log_path)

def main():
    makedirs(args.export_adv_patch_path)
    
    with open("comet_ml_cred.json", "r") as f:
        comet_ml_cred = json.load(f)
        
    experiment = comet_ml.start(
                    api_key=comet_ml_cred["api_key"],
                    project_name="Generate adv patch on Monodepthv2"
                )

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    #Tensorboard
    writer = init_tensorboard(args.log_path)
    for key, value in vars(args).items():
        writer.add_text(key, str(value))

    #Augmentation
    train_transform = transformer.Compose([
        transformer.RandomHorizontalFlip(),
        transformer.RandomAugumentColor(),
        transformer.RandomScaleCrop(scale_range=(0.85, 1.0)),
        transformer.ResizeImage(h=args.height, w=args.width),
        transformer.ArrayToTensor()
    ])

    #Dataloader
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
    # Victim model
    models = AdversarialModels(args)
    models.load_weights()

    # patch, mask
    adv = Adversarial_Patch(
        patch_type=args.patch_shape,
        batch_size=args.batch_size,
        image_size=min(args.width, args.height),
        patch_size=args.patch_size,
        printability_file=args.print_file
    )

    if not args.patch_path:
        patch_cpu, mask_cpu = adv.initialize_patch_and_mask()
    else:
        patch_cpu, mask_cpu = adv.load_patch_and_mask_from_file(args.patch_path, args.mask_path, npy=True)
    patch_cpu.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    # Train
    print('===============================')
    print("Start training ...")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        ep_nps_loss, ep_tv_loss, ep_loss, ep_disp_loss = 0, 0, 0, 0
        ep_time = time.time()

        for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader), leave=False):
            with torch.autograd.detect_anomaly():
                sample = to_cuda_vars(sample)  # send item to gpu
                sample.update(models.get_original_disp(sample))  # get non-attacked disparity

                img, original_disp = sample['left'], sample['original_distill_disp']
                patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

                # transform patch
                patch_t, mask_t = adv.patch_transformer(
                    patch=patch,
                    mask=mask,
                    batch_size=args.batch_size,
                    img_size=(args.width, args.height),
                    do_rotate=True,
                    rand_loc=True,
                    random_size=True,
                    train=True
                )

                # apply transformed patch to clean image
                attacked_img = adv.patch_applier(img, patch_t, mask_t)
                #cv2.imshow("attacked", np.asarray(attacked_img_pil))
                #cv2.waitKey(0)

                # Loss
                disp_loss = torch.tensor(0.0, device=img.device)

                if 'distill' in args.model:
                    adv_target_distill_disp = adv.create_fake_disp(original_disp.detach(), mask_t[:, 0, :, :].detach(), args.target_disp)
                    est_disp = models.distill(attacked_img)
                    print("Est disp: ", est_disp)
                    distill_loss = torch.nn.functional.l1_loss(torch.mul(mask_t, adv_target_distill_disp), torch.mul(mask_t, est_disp)).mean()
                    disp_loss += distill_loss

                elif 'monodepth2' in args.model:
                    adv_target_distill_disp = adv.create_fake_disp(original_disp.detach(), mask_t[:, 0, :, :].detach(), args.target_disp)
                    est_disp = models.distill(attacked_img)
                    distill_loss = torch.nn.functional.l1_loss(torch.mul(mask_t, adv_target_distill_disp), torch.mul(mask_t, est_disp)).mean() #loss in the patch area
                    disp_loss += distill_loss

                nps_loss = adv.calculate_nps(patch)
                tv_loss = adv.calculate_tv(patch)

                loss = disp_loss + nps_loss * 0.01 + torch.max(tv_loss, torch.tensor(0.1).cuda()) * 2.5

                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_disp_loss += disp_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                if i_batch % int(args.log_interval/args.batch_size) == 0:
                #if i_batch % 27 == 0:               
                    iteration = len(train_loader) * epoch + i_batch
                    writer.add_scalar("total_loss", loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar("loss/disp_loss", disp_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar("loss/nps_loss", nps_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar("loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                    writer.add_scalar("misc/epoch", epoch, iteration)
                    writer.add_scalar("misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                    writer.add_image("patch", patch.detach().cpu().numpy(), iteration)
                del patch_t, loss, nps_loss, tv_loss, disp_loss
                torch.cuda.empty_cache()
                
        ep_loss = ep_loss/len(train_loader)
        ep_disp_loss = ep_disp_loss/len(train_loader)
        ep_nps_loss = ep_nps_loss/len(train_loader)
        ep_tv_loss = ep_tv_loss/len(train_loader)
        scheduler.step(ep_loss)
        
        total_time = time.time() - start_time
        print('===============================')
        print(f"Total training time: {format_time(int(total_time))}")
        print('Epoch: ', epoch)
        print('Total time: ', format_time(int(total_time)))
        print('epoch loss: ', ep_loss)
        print('disparity loss: ', ep_disp_loss)
        print('NPS loss: ', ep_nps_loss)
        print('TV loss: ', ep_tv_loss)
        np.save(args.export_adv_patch_path + '/epoch_{}_patch.npy'.format(str(epoch)), patch_cpu.data.numpy())
        np.save(args.export_adv_patch_path + '/epoch_{}_mask.npy'.format(str(epoch)), mask_cpu.data.numpy())
    
    experiment.end()

if __name__ == '__main__':
    main()
