import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import torch.nn.functional as F
import torchgeometry as tgm


class CreateFakeDisp(nn.Module):
    def __init__(self):
        super(CreateFakeDisp, self).__init__()

    def forward(self, original_disp, mask, target_disp):
        #print(f"mask array: {mask}")
        mask = mask[0, :, :]
        #print(f"mask element: {mask}")
        fake_disp = torch.mul((1 - mask), original_disp) + torch.mul(mask, target_disp)

        return fake_disp


class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, patch, mask):
        #print("Patch Applier")
        #print(img_batch.shape)
        #print(patch.shape)
        #print(mask.shape)
        patched_img_batch = torch.mul((1 - mask), img_batch) + torch.mul(mask, patch)
        return patched_img_batch


class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.05
        self.max_brightness = 0.05
        self.noise_factor = 0.10
        self.minangle = -20
        self.maxangle = 20
        self.minsize = 0.35
        self.maxsize = 0.45

        self.min_x_off = -200
        self.max_x_off = 200
        self.min_y_off = -80
        self.max_y_off = 80
        self.max_x_trans = 0.1
        self.min_x_trans = -0.1
        self.max_y_trans = 0.1
        self.min_y_trans = -0.1

    def normalize_transforms(self, transforms, W, H):
        theta = torch.zeros(transforms.shape[0], 2, 3).cuda()
        theta[:, 0, 0] = transforms[:, 0, 0]
        theta[:, 0, 1] = transforms[:, 0, 1]*H/W
        theta[:, 0, 2] = transforms[:, 0, 2]*2/W + theta[:, 0, 0] + theta[:, 0, 1] - 1

        theta[:, 1, 0] = transforms[:, 1, 0]*W/H
        theta[:, 1, 1] = transforms[:, 1, 1]
        theta[:, 1, 2] = transforms[:, 1, 2]*2/H + theta[:, 1, 0] + theta[:, 1, 1] - 1

        return theta
    
    def get_perspective_transform(self, src, dst):
        B = src.size(0)
        ones = torch.ones(B, 4, 1, device=src.device)
        zeros = torch.zeros_like(ones)

        x, y = src[:, :, 0:1], src[:, :, 1:2]
        u, v = dst[:, :, 0:1], dst[:, :, 1:2]

        A = torch.cat([
            torch.cat([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=2),
            torch.cat([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=2)
        ], dim=1)  # shape: (B, 8, 8)

        b = torch.cat([u, v], dim=1)  # shape: (B, 8, 1)

        h = torch.linalg.solve(A, b)  # shape: (B, 8, 1)
        h = h.view(B, 8)
        h = torch.cat([h, torch.ones(B, 1, device=src.device)], dim=1)
        return h.view(B, 3, 3)

    def forward(self, patch, mask, batch_size, img_size, do_rotate=True, rand_loc=True, random_size=True, do_perspective=True, train=True):
        device = patch.device
        # Determine size of padding
        #print("Patch Transformer")
        #pad = (img_size - patch.size(-1)) / 2
        #print(img_size)
        img_width, img_height = img_size
        pad_width = (img_width - patch.size(-1)) / 2
        pad_height = (img_height - patch.size(-1)) / 2
        
        # Make a batch of patches
        adv_batch = patch.expand(batch_size, -1, -1, -1)
        mask_batch = mask.expand(batch_size, -1, -1, -1)

        if train:
            contrast = torch.empty_like(adv_batch).uniform_(self.min_contrast, self.max_contrast)
            brightness = torch.empty_like(adv_batch).uniform_(self.min_brightness, self.max_brightness)
            noise = torch.empty_like(adv_batch).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise transformation, and then clamp
            adv_batch = adv_batch * contrast + brightness + noise
            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Pad patch and mask to image dimensions
        # mypad = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
        #mypad = nn.ConstantPad2d((160, 544, int(pad + 0.5), int(pad)), 0)
        mypad = nn.ConstantPad2d((int(pad_width + 0.5), int(pad_width), int(pad_height + 0.5), int(pad_height)), 0)
        
        adv_batch = mypad(adv_batch)
        mask_batch = mypad(mask_batch)

        # Rotation and rescaling transforms
        if do_rotate:
            angle = torch.FloatTensor(1).uniform_(self.minangle, self.maxangle).to(device).expand(batch_size)
        else:
            angle = torch.zeros(batch_size, device=device)

        # Resize
        current_patch_size = adv_batch.size(-2)
        if random_size:
            size = torch.FloatTensor(1).uniform_(self.minsize, self.maxsize).to(device)
            target_size = current_patch_size * (size ** 2)
        else:
            target_size = torch.tensor([current_patch_size * (0.4 ** 2)], device=device)
        scale = (target_size / current_patch_size).expand(batch_size)
        
        angle_rad = angle * math.pi / 180

        cos = torch.cos(angle_rad) * scale
        sin = torch.sin(angle_rad) * scale
        

        # Rotate
        center = torch.tensor([adv_batch.shape[3] / 2, adv_batch.shape[2] / 2], device=device).expand(batch_size, 2)

        rotation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation[:, 0, 0] = cos
        rotation[:, 0, 1] = -sin
        rotation[:, 1, 0] = sin
        rotation[:, 1, 1] = cos
        rotation[:, 0, 2] = center[:, 0] * (1 - cos) - center[:, 1] * sin
        rotation[:, 1, 2] = center[:, 1] * (1 - cos) + center[:, 0] * sin

        # Translation
        translation = torch.eye(3, 3).cuda()
        translation = translation.expand(batch_size, -1, -1).clone()

        if rand_loc:
            x_off = torch.FloatTensor(1).uniform_(self.min_x_off, self.max_x_off).to(device) / scale
            y_off = torch.FloatTensor(1).uniform_(self.min_y_off, self.max_y_off).to(device) / scale
            translation[:, 0, 2] = x_off
            translation[:, 1, 2] = y_off


        if do_perspective:
            # Perspective transform
            src_coord = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]], device=device).expand(batch_size, -1, -1)
            a_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            a_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            b_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            b_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            c_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            c_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            d_x = torch.FloatTensor(1).uniform_(self.min_x_trans, self.max_x_trans).to(device)
            d_y = torch.FloatTensor(1).uniform_(self.min_y_trans, self.max_y_trans).to(device)
            
            dst_coord = torch.tensor([[
                [0 + a_x, 0 + a_y],
                [1 + b_x, 0 + b_y],
                [0 + c_x, 1 + c_y],
                [1 + d_x, 1 + d_y],
            ]], device=device).expand(batch_size, -1, -1)


            perspective = self.get_perspective_transform(src_coord, dst_coord)
            M = rotation @ translation @ perspective
            
        else:
            M = rotation @ translation

        M_inv = torch.inverse(M)
        #theta = self.normalize_transforms(M_inv, W=512, H=256)
        theta = self.normalize_transforms(M_inv, W=1024, H=320)

        grid = F.affine_grid(theta, adv_batch.shape, align_corners=False)
        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        mask_batch_t = F.grid_sample(mask_batch, grid, align_corners=False)

        return adv_batch_t, mask_batch_t


class NonPrintabilityScore(nn.Module):
    def __init__(self, printability_file, patch_side):
        super(NonPrintabilityScore, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side), requires_grad=False)

    def forward(self, patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score/torch.numel(patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, patch):
        tvcomp1 = torch.sum(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]+0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(patch[:, 1:, :] - patch[:, :-1, :]+0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(patch)


class PatchFunction(object):
    def __init__(self):
        super(PatchFunction, self).__init__()

    def LoadPatchFromImage(self, image_path, mask_path):
        noise_size = np.floor(self.image_size * np.sqrt(self.patch_size))
        patch_image = np.array(cv2.imread(image_path)).astype(np.float32)
        patch_image = self.resize_img(patch_image, (int(noise_size), int(noise_size)))/128. - 1
        patch = torch.FloatTensor(np.array([patch_image.transpose(2, 0, 1)]))
        mask_image = np.array(cv2.imread(mask_path)).astype(np.float32)
        mask_image = self.resize_img(mask_image, (int(noise_size), int(noise_size)))/256.
        mask = torch.FloatTensor(np.array([mask_image.transpose(2, 0, 1)]))
        return patch, mask

    def LoadPatchFromNpy(self, image_path, mask_path):
        patch_image = np.load(image_path).astype(np.float32)
        mask_image = np.load(mask_path).astype(np.float32)
        patch = torch.FloatTensor(patch_image)
        mask = torch.FloatTensor(mask_image)
        return patch, mask

    def InitSquarePatch(self, image_size, patch_size):
        noise_dim = patch_size
        patch = torch.rand((3, noise_dim, noise_dim))
        mask = torch.full((3, noise_dim, noise_dim), 1, dtype=torch.float32)
        return patch, mask

    def InitCirclePatch(self, image_size, patch_size):
        patch, _ = self.InitSquarePatch(image_size, patch_size)
        mask = self.CreateCircularMask(patch.shape[-2], patch.shape[-1]).astype('float32')
        mask = torch.FloatTensor(np.array([mask, mask, mask]))
        return patch, mask

    def CreateCircularMask(self, w, h):
        center = [int(w/2), int(h/2)]
        radius = min(center[0], center[1], w-center[0], h-center[1])-2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def resize_img(self, img, size):
        width, height = size
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class Adversarial_Patch(PatchFunction):
    def __init__(self, patch_type, batch_size, image_size, patch_size, train=True, printability_file=None):
        super(Adversarial_Patch, self).__init__()
        self.patch_type = patch_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_transformer = PatchTransformer()
        self.patch_applier = PatchApplier()

        if train:
            self.printfile = printability_file
            self.create_fake_disp = CreateFakeDisp()
            self.calculate_nps = NonPrintabilityScore(self.printfile, self.patch_size).cuda()
            self.calculate_tv = TotalVariation().cuda()

    def initialize_patch_and_mask(self):
        if self.patch_type == 'square':
            patch, mask = self.InitSquarePatch(self.image_size, self.patch_size)
        elif self.patch_type == 'circle':
            patch, mask = self.InitCirclePatch(self.image_size, self.patch_size)
        print('=> Use "{}" patch with shape {}'.format(self.patch_type, patch.shape))
        return patch, mask

    def load_patch_and_mask_from_file(self, patch_path, mask_path, npy=True):
        patch, mask = self.LoadPatchFromNpy(patch_path, mask_path) if npy else self.LoadPatchFromImage(patch_path, mask_path)
        print('=> Load patch with shape {} from \'{}\''.format(patch.shape, patch_path))
        return patch, mask
