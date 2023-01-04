from __future__ import print_function
import argparse
import os
import time
import resource
import gc
import logging
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #ban gpu
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from ast import literal_eval
import models
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
# Training settings
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# t1 = time.time()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread

def inference(str_path, x):
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))
    model = model(**model_config)
    model.load_state_dict(torch.load(str_path, map_location=torch.device('cpu'))['state_dict'])
    if args.cuda:
        model.cuda()
    model.eval()
    y = model(x)
    return 

async def ainference(model, x, name):
    y = model(x)
    return name

gc.collect()
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
soft = 42949672960 #bit 500MB*2


hard = soft
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


async def main():
    # 方法3 通过线程池
    num_model = 30
    num_thread = 20
    epoch = 10
    min_time = 100000
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=num_thread)

    str0 = './results/model_dir/alex-internet/checkpoint_0.pth.tar'
    str1 = './results/model_dir/alex-internet/checkpoint_1.pth.tar'
    str2 = './results/model_dir/alex-internet/checkpoint_2.pth.tar'
    str3 = './results/model_dir/alex-internet/checkpoint_3.pth.tar'
    str4 = './results/model_dir/alex-internet/checkpoint_4.pth.tar'
    str5 = './results/model_dir/alex-internet/checkpoint_5.pth.tar'
    str6 = './results/model_dir/alex-internet/checkpoint_6.pth.tar'
    str7 = './results/model_dir/alex-internet/checkpoint_7.pth.tar'
    str8 = './results/model_dir/alex-internet/checkpoint_8.pth.tar'
    str9 = './results/model_dir/alex-internet/checkpoint_9.pth.tar'
    str10 = './results/model_dir/alex-internet/checkpoint_10.pth.tar'
    str11 = './results/model_dir/alex-internet/checkpoint_11.pth.tar'
    str12 = './results/model_dir/alex-internet/checkpoint_12.pth.tar'
    str13 = './results/model_dir/alex-internet/checkpoint_13.pth.tar'
    str14 = './results/model_dir/alex-internet/checkpoint_14.pth.tar'
    str15 = './results/model_dir/alex-internet/checkpoint_15.pth.tar'
    str16 = './results/model_dir/alex-internet/checkpoint_16.pth.tar'
    str17 = './results/model_dir/alex-internet/checkpoint_17.pth.tar'
    str18 = './results/model_dir/alex-internet/checkpoint_18.pth.tar'
    str19 = './results/model_dir/alex-internet/checkpoint_19.pth.tar'
    str20 = './results/model_dir/alex-internet/checkpoint_20.pth.tar'
    str21 = './results/model_dir/alex-internet/checkpoint_21.pth.tar'
    str22 = './results/model_dir/alex-internet/checkpoint_22.pth.tar'
    str23 = './results/model_dir/alex-internet/checkpoint_23.pth.tar'
    str24 = './results/model_dir/alex-internet/checkpoint_24.pth.tar'
    str25 = './results/model_dir/alex-internet/checkpoint_25.pth.tar'
    str26 = './results/model_dir/alex-internet/checkpoint_26.pth.tar'
    str27 = './results/model_dir/alex-internet/checkpoint_27.pth.tar'
    str28 = './results/model_dir/alex-internet/checkpoint_28.pth.tar'
    str29 = './results/model_dir/alex-internet/checkpoint_29.pth.tar'
    # t22 = time.time()
   
    for j in range (1,num_model+1):
        os.system('sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')
        t1 = time.time()
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        data, target = torch.load("./data/test_load/alex_data.pt") 
        if args.cuda:
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        data = data.to(torch.device('cpu'))
        target = target.to(torch.device('cpu'))
        t2 = time.time()
        sum1 = t2-t1
        print("load data time: ", sum1)
        s_t = 0
        for k in range(epoch):
            threads = []
            s = time.time()
            for i in range(j):
                str_s = './results/model_dir/alex-internet/checkpoint_'+str(i)+'.pth.tar'
                threads.append(loop.run_in_executor(pool, inference, str_s, data))
            done,pending = await asyncio.wait(threads)
            e = time.time()
            tmp_time = e-s
            s_t = s_t + tmp_time
        print(str(j),s_t/epoch)


if __name__ == '__main__':
    # main()
    asyncio.run(main())



