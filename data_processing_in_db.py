import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import sys
sys.path.append('..')
import im2col as conveter
import numpy as np
import time
from clickhouse_driver import Client

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class BinarizeConv2dinDB(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2dinDB, self).__init__(*kargs, **kwargs)


    def forward(self, input):

        if input.size(1) != 3:
        # if input.size(1) == 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        # trandoramtion funtion from a tensor to multiple 1-dimensional arrays
        w1_cnn = self.weight
        w_f,w_c,w_h,w_w = self.weight.shape
        w1_cnn_train=conveter.stretchKernel(w1_cnn)
        x_b,_,x_h,x_w = input.shape
        h_out=int((x_h+2*self.padding[0]-w_h)/self.stride[0])+1
        w_out=h_out
        input = conveter.im2col(input.cpu(),(w_f,w_h,w_w,w_c),self.stride[0],(x_b,h_out,w_out,w_f))

        # bitwise compression
        input_int = []
        # kernel_int = []
        # length_int = []
        len_a = input.shape[0]
        len_b = w1_cnn_train.shape[1]
        for i in range (len_a):
            for j in range (len_b):
                test_a = input[i]
                test_b = w1_cnn_train[:,j]
                len_test_a = len(test_a)
                len_test_b = len(test_b)
                if len_test_a != len_test_b :
                    print("error")
                    exit()
                int_test_a, int_test_b, int_length = array2int(len_test_a, test_a, test_b)
                input_int.append([int(int_test_a), int(int_test_b), int(int_length)])
                # int_test_11, int_test_21, int_test_12, int_test_22, int_test_13, int_test_23, int_test_14, int_test_24= array2int_uint8(len_test_a, test_a, test_b)
                # input_int.append([int_test_11, int_test_21, int_test_12, int_test_22, int_test_13, int_test_23, int_test_14, int_test_24])
                # int_test_11, int_test_21, int_test_12, int_test_22= array2int_uint16(len_test_a, test_a, test_b)
                # input_int.append([int_test_11, int_test_21, int_test_12, int_test_22])

        input_int=np.array(input_int)
        np.save('input_int.npy',input_int)  #save binarized data list to current dir
        # np.save('input_int_uint8.npy',input_int)
        # np.save('input_int_5x5.npy',input_int)
        # np.save('input_int_7x7.npy',input_int)
        
        input_int=np.load('input_int.npy', allow_pickle=True) # load binarized data
        print(input_int.shape)
        len_all = input_int.shape[0]
        batch_index = 1
        len_tmp = int(len_all/batch_index)
        print("len_tmp: ",len_tmp)
        # input_int=input_int.tolist()
        input_int=input_int[:len_tmp,:2].tolist()
        
 
        # clickhouse
        db = Client(host='localhost')
        clean_bit_input = "DROP TABLE IF EXISTS bit_input;"
        db.execute(clean_bit_input)

        # test engine = MergeTree() with sparse index
        create_bit_input="CREATE TABLE IF NOT EXISTS bit_input (bit_1 UInt32, bit_2 UInt32, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
        # create_bit_input="CREATE TABLE IF NOT EXISTS bit_input (bit_1 UInt16, bit_2 UInt16, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
        # create_bit_input="CREATE TABLE IF NOT EXISTS bit_input (bit_1 UInt16, bit_2 UInt16, bit_3 UInt16, bit_4 UInt16" \
        #     ", PRIMARY KEY(bit_1, bit_2, bit_3, bit_4)) engine = MergeTree();" # test batch = 4
        # print("engine=mergetree")
        db.execute(create_bit_input)
        db.execute("insert into bit_input values ", input_int) 
        
        # test engine = Memory
        clean_bit_input_mem = "DROP TABLE IF EXISTS bit_output1;"
        db.execute(clean_bit_input_mem)
        create_bit_input_mem="CREATE TABLE IF NOT EXISTS bit_output1(bit_1 UInt32, bit_2 UInt32) engine = Memory;" 
        db.execute(create_bit_input_mem)
        db.execute("insert into bit_output1 values ", input_int)
        
        print("end create_feature_sql")

        return out


def array2int (len_array, array_input, array_kernel):
    sum_input = 0
    sum_kernel = 0
    tmp_length = 0 
    for i in range(len_array):
        t = len_array - i - 1
        if array_input[t] and array_kernel[t]:
            if array_input[t] > 0:
                sum_input = sum_input + pow(2, tmp_length)
            if array_kernel[t] > 0:
                sum_kernel = sum_kernel + pow(2, tmp_length)
            tmp_length = tmp_length + 1
            
    return sum_input, sum_kernel, tmp_length

def array2int_uint8 (len_array, array_input, array_kernel):
    sum_input_1, sum_kernel_1, sum_input_2, sum_kernel_2, sum_input_3, sum_kernel_3, sum_input_4, sum_kernel_4 = 0, 0, 0, 0, 0, 0 ,0 ,0
    len_array1, len_array2, len_array3 = 8, 16, 24
    tmp_length = 0
    for i in range(0,len_array1):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_1 = sum_input_1 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_1 = sum_kernel_1 + pow(2, tmp_length)
            tmp_length = tmp_length + 1
    tmp_length = 0
    for i in range(len_array1, len_array2):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_2 = sum_input_2 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_2 = sum_kernel_2 + pow(2, tmp_length)
            tmp_length = tmp_length + 1
    tmp_length = 0
    for i in range(len_array2, len_array3):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_3 = sum_input_3 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_3 = sum_kernel_3 + pow(2, tmp_length)    
            tmp_length = tmp_length + 1
    tmp_length = 0
    for i in range(len_array3, len_array):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_4 = sum_input_4 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_4 = sum_kernel_4 + pow(2, tmp_length)    
            tmp_length = tmp_length + 1 
    return sum_input_1, sum_kernel_1, sum_input_2, sum_kernel_2, sum_input_3, sum_kernel_3, sum_input_4, sum_kernel_4

def array2int_uint16 (len_array, array_input, array_kernel):
    sum_input_1, sum_kernel_1, sum_input_2, sum_kernel_2 = 0, 0, 0, 0
    len_array1 = 16
    tmp_length = 0
    for i in range(0,len_array1):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_1 = sum_input_1 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_1 = sum_kernel_1 + pow(2, tmp_length)
            tmp_length = tmp_length + 1
    tmp_length = 0
    for i in range(len_array1, len_array):
        if array_input[i] and array_kernel[i]:
            if array_input[i] > 0:
                sum_input_2 = sum_input_2 + pow(2, tmp_length)
            if array_kernel[i] > 0:
                sum_kernel_2 = sum_kernel_2 + pow(2, tmp_length)
            tmp_length = tmp_length + 1
    
    return sum_input_1, sum_kernel_1, sum_input_2, sum_kernel_2


def bin2int (array_input):
    input_int = []
    for j in range(array_input.shape[0]):
        sum_input = 0
        for i in range(array_input.shape[1]):
            if array_input[j][i] > 0:
                    sum_input = sum_input + pow(2, i)
        input_int.append([int(sum_input)])
    input_int=np.array(input_int)
    np.save('input_pooling_256.npy',input_int)
    return

def bin2int_linear (array_input):
    input_int = []
    len1 = 256
    for j in range(array_input.shape[0]):
        sum_input = 0
        sum_input_sht = 0
        for i in range(len1):
            if array_input[j][i] > 0:
                    sum_input = sum_input + pow(2, i)
        for i in range(len1,array_input.shape[1]):
            if array_input[j][i] > 0:
                    sum_input_sht = sum_input_sht + pow(2, i-len1)
        input_int.append([int(sum_input),int(sum_input_sht)])
    input_int=np.array(input_int)
    np.save('input_linear_weight_256.npy',input_int)
    return