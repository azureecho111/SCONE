#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model import Generator, GlobalGenerator2, InceptionV3
from dataset import UnpairedDepthDataset
from PIL import Image
import numpy as np
from utils import channel2width

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str, help='name of this experiment')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where the model checkpoints are saved')
parser.add_argument('--results_dir', type=str, default='results', help='where to save result images')
parser.add_argument('--geom_name', type=str, default='feats2Geom', help='name of the geometry predictor')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--depthroot', type=str, default='', help='dataset of corresponding ground truth depth maps')

parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--geom_nc', type=int, default=3, help='number of channels of geometry data')
parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for the geometry loss')
parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
parser.add_argument('--midas', type=int, default=0, help='use midas depth map')

parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')

parser.add_argument('--mode', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

parser.add_argument('--predict_depth', type=int, default=0, help='run geometry prediction on the generated images')
parser.add_argument('--save_input', type=int, default=0, help='save input image')
parser.add_argument('--reconstruct', type=int, default=0, help='get reconstruction')
parser.add_argument('--how_many', type=int, default=100, help='number of images to test')

opt = parser.parse_args()
print(opt)

opt.no_flip = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

with torch.no_grad():
    # Networks

    net_G = 0
    net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
    net_G.cuda()

    net_GB = 0
    if opt.reconstruct == 1:
        net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
        net_GB.cuda()
        net_GB.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
        net_GB.eval()
    
    netGeom = 0
    if opt.predict_depth == 1:
        usename = opt.name
        if (len(opt.geom_name) > 0) and (os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))):
            usename = opt.geom_name
        myname = os.path.join(opt.checkpoints_dir, usename, 'netGeom_%s.pth' % opt.which_epoch)
        netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)

        netGeom.load_state_dict(torch.load(myname))
        netGeom.cuda()
        netGeom.eval()

        numclasses = opt.num_classes
        ### load pretrained inception
        net_recog = InceptionV3(opt.num_classes, False, use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
        net_recog.cuda()
        net_recog.eval()

    # Load state dicts
    net_G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    print('loaded', os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch))

    # Set model's test mode
    net_G.eval()

    
    transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC),
                   transforms.ToTensor()]


    test_data = UnpairedDepthDataset(opt.dataroot, '', opt, transforms_r=transforms_r, 
                mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

    dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

    ###################################

    ###### Testing######

    full_output_dir = os.path.join(opt.results_dir, opt.name)

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    for i, batch in enumerate(dataloader):
        # print(batch)
        if i > opt.how_many:
            break;
        img_r  = Variable(batch['r']).cuda()
        img_depth  = Variable(batch['depth']).cuda()
        real_A = img_r

        name = batch['name'][0]
        
        input_image = real_A
        print(input_image.shape)
        image = net_G(input_image)
        print("啊啊啊",image.shape)
        sketch_image = image.repeat_interleave(3, dim=1).squeeze()


        def tensor_to_pil_list(tensor: torch.Tensor, clip=True) -> list[Image.Image]:
            """
            将 CLIP Processor 输出的 Tensor 转换回 PIL Image 列表。
            - 如果输入形状为 (C, H, W)，返回包含 1 个 Image 的列表。
            - 如果输入形状为 (B, C, H, W)，返回包含 B 个 Image 的列表。
            """
            CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            if clip:
                if isinstance(tensor, Image.Image):
                    return [tensor]
                # print('本来应该',type(tensor))
                if tensor.ndim == 3:
                    # 如果是 (C, H, W)，添加一个 Batch 维度使其成为 (1, C, H, W)
                    tensor = tensor.unsqueeze(0)

                # 检查维度是否正确，确保是 (B, C, H, W)
                if tensor.ndim != 4 or tensor.shape[1] != 3:
                    raise ValueError(
                        f"Input tensor must be 3D (C, H, W) or 4D (B, C, H, W) with 3 channels, but got shape {tensor.shape}")

                # 使用 torch.unbind(dim=0) 将 Batch 维度拆分成一个包含 B 个 (C, H, W) tensor 的元组
                list_of_tensors = list(torch.unbind(tensor, dim=0))

                output_images = []

                # 遍历每个 (C, H, W) 图像 tensor
                for single_image_tensor in list_of_tensors:
                    # 确保 MEAN 和 STD 在同一设备上
                    current_mean = CLIP_MEAN.to(single_image_tensor.device)
                    current_std = CLIP_STD.to(single_image_tensor.device)

                    # 1. 反归一化： tensor = tensor * std + mean
                    restored_tensor = single_image_tensor * current_std + current_mean

                    # 2. 裁剪到 0-1 范围，防止溢出
                    restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

                    # 3. 转换维度： (C, H, W) -> (H, W, C)
                    #    并转换为 NumPy 数组 (0-255 整数)
                    image_numpy = (restored_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    # 4. 转换为 PIL Image
                    output_images.append(Image.fromarray(image_numpy))

                return output_images
            else:
                from torchvision.transforms.functional import to_pil_image
                if tensor.ndim == 3:
                    # 单张图片 [C, H, W]
                    return [to_pil_image(tensor)]
                else:
                    # 多张图片 [B, C, H, W]
                    images = []
                    for i in range(tensor.shape[0]):
                        images.append(to_pil_image(tensor[i]))
                    return images
        sketch_image = tensor_to_pil_list(sketch_image.detach(), False)[0]
        sketch_image.save(f"I_want_{name}.png", format='png')
        print(type(image), ' ', image.shape)
        save_image(image.data, full_output_dir+'/%s_out.png' % name)

        if (opt.predict_depth == 1):

            geom_input = image
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            _, geom_input = net_recog(geom_input)
            geom = netGeom(geom_input)
            geom = (geom+1)/2.0 ###[-1, 1] ---> [0, 1]

            input_img_fake = channel2width(geom)
            save_image(input_img_fake.data, full_output_dir+'/%s_geom.png' % name)

        if opt.reconstruct == 1:
            rec = net_GB(image)
            save_image(rec.data, full_output_dir+'/%s_rec.png' % name)

        if opt.save_input == 1:
            save_image(img_r, full_output_dir+'/%s_input.png' % name)

        sys.stdout.write('\rGenerated images %04d of %04d' % (i, opt.how_many))

    sys.stdout.write('\n')
    ###################################
