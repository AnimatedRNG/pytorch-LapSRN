#!/usr/bin/env python3

import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time
import pytorch_ssim
import os
import cv2
from sr_dataset import SRImageDataset, bicubic_downsample


def to_tensor(img):
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument(
        "--model", default="checkpoint/lapsrn_model_epoch_100.pth", type=str, help="model path")
    parser.add_argument("--dataset", default="test/", type=str,
                        help="dataset name, Default: Set5")
    parser.add_argument("--scale", default=4, type=int,
                        help="scale factor, Default: 4")

    opt = parser.parse_args()
    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if not cuda:
        model = torch.load(opt.model, map_location='cpu')["model"]
    else:
        model = torch.load(opt.model)["model"]

    avg_ssim_predicted = 0.0
    avg_ssim_bicubic = 0.0
    avg_elapsed_time = 0.0

    dataset = SRImageDataset(opt.dataset, 256, 256,
                             not cuda, bicubic_downsample, [lambda img: img])
    data_loader = DataLoader(dataset=dataset,
                             num_workers=0,
                             batch_size=1,
                             shuffle=False)
    model.eval()

    for batch in data_loader:
        input, label_x2, label_x4 = Variable(batch[0], requires_grad=False, volatile=True), \
            Variable(batch[1], requires_grad=False, volatile=True), \
            Variable(batch[2], requires_grad=False, volatile=True)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()

        bicubic_x4 = torch.nn.functional.interpolate(input,
                                                     label_x4.shape[2:],
                                                     None, mode='bicubic')

        bicubic_ssim = pytorch_ssim.ssim(label_x4, bicubic_x4)
        print("Bicubic ssim: {}".format(bicubic_ssim))
        avg_ssim_bicubic += bicubic_ssim

        start_time = time.time()
        HR_2x, HR_4x = model(input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        ssim_predicted = pytorch_ssim.ssim(label_x4, HR_4x)
        print("SSIM predicted: {}".format(ssim_predicted))
        avg_ssim_predicted += ssim_predicted

        cv2.imshow('label', label_x4.detach().cpu().numpy()[0, 0])
        cv2.imshow('bicubic', bicubic_x4.detach().cpu().numpy()[0, 0])
        cv2.imshow('nearest', cv2.resize(input.detach().cpu().numpy()[
                   0, 0], (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        cv2.imshow('predicted', HR_4x.detach().cpu().numpy()[0, 0])
        cv2.waitKey(0)
