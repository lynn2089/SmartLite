import sys
import os
import torch
import argparse
import data
import time
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import models
from ast import literal_eval
from clickhouse_driver import Client
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from torchvision.utils import save_image

# Training settings
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Example')

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
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

len_bit=2   #lengh of spliting
val_ptr=1   #lengh of filter

val_ptr_list = [1,2,4,8,16,32,64]

for val_ptr in val_ptr_list:
    # val_ptr = len_bit
    print("val_ptr: ", val_ptr)

    with open("./prune/"+str(args.model)+"_cnt_mrtx_epoch30.txt", "rb") as fp:
        cnt_mrtx = pickle.load(fp)

    with open("./prune/"+str(args.model)+"_cnt_mrtx_epoch40.txt", "rb") as fp2:
        cnt_mrtx_1 = pickle.load(fp2)

    cnt_mrtx = cnt_mrtx_1 - cnt_mrtx
    cnt_mrtx = cnt_mrtx.permute(0,2,3,1)

    prune_cnt_mrtx=[]
    cnt_mrtx = cnt_mrtx.reshape(cnt_mrtx.shape[0], cnt_mrtx.shape[1]*cnt_mrtx.shape[2]*cnt_mrtx.shape[3]).cpu().detach().numpy()

    # strategy 1: count the number of times the bit fips during training of the whole block
    for k in range (cnt_mrtx.shape[0]):
        prune_cnt_mrtx_1=[]
        i=0
        tmp_sum = 0
        for j in range(cnt_mrtx.shape[1]):
            if i < len_bit:
                tmp_sum = tmp_sum+cnt_mrtx[k][j]
                i=i+1
            else:
                prune_cnt_mrtx_1.append((tmp_sum,(j/len_bit)-1))
                i=0
                tmp_sum = 0
                tmp_sum = tmp_sum+cnt_mrtx[k][j]
                i=i+1
        prune_cnt_mrtx.append(prune_cnt_mrtx_1)
    prune_cnt_mrtx = np.array(prune_cnt_mrtx)
    prune_cnt_mrtx = prune_cnt_mrtx.tolist()

    idx = []
    for i in range(len(prune_cnt_mrtx)):
        prune_cnt_mrtx[i]= sorted(prune_cnt_mrtx[i])
    prune_cnt_mrtx = np.array(prune_cnt_mrtx)

    for i in range(prune_cnt_mrtx.shape[0]):
        idx_1=[]
        for j in range(prune_cnt_mrtx.shape[1]):
            if(prune_cnt_mrtx[i][j][0]<val_ptr):
                idx_1.append((prune_cnt_mrtx[i][j][1],0))   #level
            else:
                idx_1.append((prune_cnt_mrtx[i][j][1],1))
        idx.append(idx_1)
    idx = np.array(idx)

    '''
    # strategy 2: if all bits in the split bit-array are invalid, we can mark the whole bit-array as invalid
    idx=[]
    for k in range (cnt_mrtx.shape[0]):
        idx_1=[]
        i=0
        flag = 1
        for j in range(cnt_mrtx.shape[1]):
            if i < len_bit:
                # tmp_sum = tmp_sum+cnt_mrtx[k][j]
                if cnt_mrtx[k][j] < val_ptr:
                    flag = 0
                i=i+1
            else:
                # idx_1.append((tmp_sum,(j/len_bit)-1))
                if flag == 0:
                    idx_1.append(((j/len_bit)-1,0))
                else:
                    idx_1.append(((j/len_bit)-1,1))
                i=0
                flag = 1
                if cnt_mrtx[k][j] < val_ptr:
                    flag = 0
                i=i+1
        idx.append(idx_1)
    idx = np.array(idx)
    '''
    
    str_s = './results/'+str(args.save)+'/model_best.pth.tar'
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))
    model = model(**model_config)
    dict = torch.load(str_s, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(dict)

    if args.cuda:
        model.cuda()

    default_transform = {
        'train': get_transform(args.dataset,
                                input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                                input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
        if args.gpus and len(args.gpus) > 1:
            model = torch.nn.DataParallel(model, args.gpus)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (inputs, target) in enumerate(val_loader):
            if args.gpus is not None:
                target = target.cuda()

            if not training:
                with torch.no_grad():
                    input_var = Variable(inputs.type(args.type), volatile=not training)
                    target_var = Variable(target)
                    # compute output
                    
                    output = model(input_var)
            else:
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                output = model(input_var)


            loss = criterion(output, target_var)
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.data.copy_(p.org)
                optimizer.step()
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.org.copy_(p.data.clamp_(-1,1))
        return losses.avg, top1.avg, top5.avg

    def validate(data_loader, model, criterion, epoch):
        # switch to evaluate mode
        model.eval()
        return forward(data_loader, model, criterion, epoch,
                    training=False, optimizer=None)


    # pruning unnecessary parameters
    # weight_mtrx = model.layer3[0].conv2.weight.data #resnet
    weight_mtrx = model.conv1.weight.data  #alexnet/vgg
    h, w, c_out = weight_mtrx.shape[2], weight_mtrx.shape[3], weight_mtrx.shape[1]
    weight_mtrx = weight_mtrx.reshape(weight_mtrx.shape[0], weight_mtrx.shape[1]*weight_mtrx.shape[2]*weight_mtrx.shape[3]).cpu().detach().numpy()
    count = 0
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            if(int(idx[i][j][1])>0):
                idx_x = int(idx[i][j][0]*len_bit)
                for k in range(len_bit):
                    count = count+1
                    weight_mtrx[i][idx_x+k]=0
    print("the number of pruning: ", count)
    print("the ratio of pruning: ", count/(weight_mtrx.shape[0]*weight_mtrx.shape[1]))

    # test accuracy
    # model.layer3[0].conv2.weight.data = torch.tensor(weight_mtrx.reshape(weight_mtrx.shape[0], c_out, h, w)) #resnet
    model.conv1.weight.data = torch.tensor(weight_mtrx.reshape(weight_mtrx.shape[0], c_out, h, w)) #alexnet/vgg
    model=model.cuda()

    val_loss, val_prec1, val_prec5 = validate(
        val_loader, model, criterion, epoch=0)

    print('after prune acc: ', val_loss, val_prec1, val_prec5)






        







        






