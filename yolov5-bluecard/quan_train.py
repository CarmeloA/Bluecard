import argparse
import glob
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
from torchvision import transforms as tf
import MNN
MNNF = MNN.expr
nn = MNN.nn

logger = logging.getLogger(__name__)


def train(hyp,device):
    cuda = device.type != 'cpu'

    # v5 Model
    weights = './weights/ckpt_model_599_800_0.07873.pt'
    model = attempt_load(weights, map_location=device)

    # load MNN Model
    model_file = './weights/20201231_exp25_599_800_forT_bs128_320x320.mnn'
    net = nn.load_module_from_file(model_file, for_training=True)
    nn.compress.train_quant(net, quant_bits=8)
    mnn_opt = MNN.optim.SGD(1e-6, 0.9, 0)
    mnn_opt.append(net.parameters)
    net.train(True)


    # Image sizes
    gs = 32  # grid size (max stride)
    imgsz = 320
    batch_size = 128# verify imgsz are gs-multiples
    stride = [8,16,32]
    train_path = '/home/sysman/gate_Sample/VOCdevkit/VOC2017/ImageSets/train_5th_add.txt'
    opt = ''



    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs,opt,
                                            hyp=hyp, augment=True, cache=False, rect=False, rank=-1,
                                            world_size=1, workers=32)


    for epoch in range(5):  # epoch ------------------------------------------------------------------
        t0 = time.time()
        for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------
            t1 = time.time()
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            # Forward

            bs, c, h, w = imgs.shape

            # 20201231 by zlf
            # MNN forward
            data = MNNF.const(imgs.flatten().tolist(), [128, 3, 320, 320], MNNF.data_format.NCHW)
            predict = net.forward(data)
            predict.read()
            p1 = MNNF.Var.read(predict)
            p1 = torch.tensor(p1).cuda()
            x1, x2, x3 = torch.split(p1, [4800,1200,300], 1)

            x1 = x1.view(-1, 3, 109, 40, 40).permute(0, 1, 3, 4, 2).contiguous()
            x2 = x2.view(-1, 3, 109, 20, 20).permute(0, 1, 3, 4, 2).contiguous()
            x3 = x3.view(-1, 3, 109, 10, 10).permute(0, 1, 3, 4, 2).contiguous()
            x = [x1, x2, x3]

            loss1, loss_items1 = compute_loss(x, targets.to(device), model)
            loss1 = np.array(loss1.cpu())
            loss1 = MNNF.const(loss1.flatten().tolist(), [1], MNNF.data_format.NCHW)


            # Backward
            mnn_opt.step(loss1)
            t2 = time.time()
            print("[%d|%d|%d]train loss:%.5f,time:%.3f "%(epoch,i,len(dataloader)-1,loss1.read(),(t2-t1)))

        # save model
        file_name = './weights/%d_20201231test.mnn' % epoch
        net.train(False)
        predict = net.forward(MNNF.placeholder([1, 3, 192, 320], MNNF.NC4HW4))
        print("Save to " + file_name)
        MNNF.save([predict], file_name)
        print('Epoch:',(time.time()-t0))


        # end epoch ----------------------------------------------------------------------------------------------------
    # end training





if __name__ == '__main__':
    with open('./data/hyp.scratch.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    device = select_device('1', batch_size=128)
    train(hyp, device)
