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
import os
from clickhouse_driver import Client


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

'''
import psycopg2
class DB:
    conn=psycopg2.connect(database="bnn2sql", user="lqr" , password="panda9105",host="localhost")
    def connect(self):
        self.conn = psycopg2.connect(database="bnn2sql", user="lqr", password="panda9105", host="localhost")
    def executeone(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
        except psycopg2.OperationalError:
            self.conn.close()
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(sql)
        return cursor
    def executemany(self, sql, data):
        try:
            cursor = self.conn.cursor()
            # executemany方法的第二个参数是一个列表，每一个成员都是一个元组。
            cursor.executemany(sql, data)
        except psycopg2.OperationalError:
            self.conn.close()
            self.connect()
            cursor = self.conn.cursor()
            cursor.executemany(sql, data)
        return cursor
    def close(self):
        self.conn.close()
    def commit(self):
        self.conn.commit()
'''

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

# import torch.nn._functions as tnnf
import torch.nn.functional as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        # print("input shape: ", input.shape)
        # #1w batch pytorch
        # x = torch.sign(torch.rand(2560000, 320)-0.5)
        # print(x.shape)
        # len_all = x.shape[0]
        # batch_index = 16
        # len_tmp = int(len_all/batch_index)
        # input = x[:len_tmp,:256]
        # self.weight.data=self.weight.data[:,:256]
        # # input=input[:][:256]
        # print("input shape: ", input.shape)
        # print("self.weight shape: ", self.weight.shape)
        # t1 = time.time()

        #---table----
        # x = torch.sign(torch.rand(51200000, 320)-0.5)
        # print(x.shape)
        # t1 = time.time()
        # bin2int_linear(x)
        # t2 = time.time()
        # print("data time: ", t2-t1)
        # exit()
        # print("start")
        # bin2int_linear(self.weight.data)
        # print("end")
        # exit()

        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        # t2 = time.time()
        # print("linear time: ", t2-t1)
        # exit()
        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        # print("input: ")
        # print(input.dtype)
        # out1 = nn.functional.conv2d(input, self.weight, None, self.stride,
        #                            self.padding, self.dilation, self.groups)
        
        # print("weight: ", self.weight.size())
        # print("input: ", input.size())
        '''
        print('1')
        t1=time.time()
        tmp =  self.weight.reshape((720,3))
        tmp = tmp[:700]
        tmp = torch.cat([tmp,tmp],dim=0)
        # tmp = torch.cat([tmp,tmp[:600]],dim=0)
        # tmp = tmp.reshape((80,3,5,5))
        tmp = torch.cat([tmp,tmp],dim=0)
        tmp = torch.cat([tmp,tmp[:1120]],dim=0)
        tmp = tmp.reshape((80,3,7,7))
        self.weight = torch.nn.Parameter(tmp)
        print(self.weight.size())

        '''
       
        # self.stride = (3,3)
        # self.padding = 0
        # x = torch.sign(torch.rand(128000*3*8*8, 16)-0.5) #db
        # x = torch.sign(torch.rand(128000,3,32,32)-0.5)  #pytorch
        # print(x.shape)
        # len_all = x.shape[0]
        # batch_index = 1
        # len_tmp = int(len_all/batch_index)
        # input = x[:len_tmp]
        # t1 = time.time()
        # bin2int(input)
        # t2 = time.time()
        # print("data time: ", t2-t1)

        # len_all = input.shape[0]
        # batch_index = 16
        # len_tmp = int(len_all/batch_index)
        # input = input[:len_tmp]
        # print("self.weight shape: ", self.weight.shape)
        # self.avgpool = nn.AvgPool2d(4)
        # self.maxpool = nn.MaxPool2d(4)
        # t1 = time.time()
        
        # val_pool = self.maxpool(input)
        # t2 = time.time()
        # print("Pool time: ", t2-t1)
        # print(val_pool.shape)
        # exit()
        # cpu_num = 8
        # os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
        # os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        # os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
        # os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        # os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        # torch.set_num_threads(cpu_num)
        # print("cpu_num: ", cpu_num)
        
        # tmp = torch.rand(5, 3, 3, 3)-0.5
        # self.weight = torch.nn.Parameter(tmp)
        # input = torch.rand(8000, 3, 32, 32)
        # self.avgpool = nn.AvgPool2d(4)
        # self.maxpool = nn.MaxPool2d(4)
        # # print(input[:5])
        # t1 = time.time()
        # val_pool = self.avgpool(input)
        # t2 = time.time()
        # print("Pool time: ", t2-t1)
        # print(input.shape)
        # print(self.weight.shape)
        # print(self.dilation)
        # print(self.groups)
        # exit()
        # t1 = time.time()
        # print(self.weight.data[:10])
        # exit()
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

        # t2 = time.time()
        # print("Conv time: ", t2-t1)
        

        '''
        w1_cnn = self.weight
        # print("self.weight: ")
        # print(self.weight.shape)
        w_f,w_c,w_h,w_w = self.weight.shape
        w1_cnn_train=conveter.stretchKernel(w1_cnn)
        print('2')
        # print("input: ")
        # print(input.shape)
        x_b,_,x_h,x_w = input.shape
        h_out=int((x_h+2*self.padding[0]-w_h)/self.stride[0])+1
        w_out=h_out
        input = conveter.im2col(input.cpu(),(w_f,w_h,w_w,w_c),self.stride[0],(x_b,h_out,w_out,w_f))

        # conv1_output = input.dot(w1_cnn_train)
        t1=time.time()
        conv1_output = input.dot(w1_cnn_train)
        t2=time.time()
        t_mal=t2-t1
        # print("conv1_output: ")
        print("input: ", input.shape)
        print("w1_cnn_train: ", w1_cnn_train.shape)
        print("conv1_output: ", conv1_output.shape)
        print(time.time())
        # np.save('input_conv.npy',input)
        # np.save('weight_conv.npy',w1_cnn_train)
        # print(conv1_output[0])
        # print("out put: ")
        # out = out.permute(0,2,3,1).contiguous()
        # out = out.reshape((262144,80))
        # print(out[0])

        # print(self.bias)
        # print("time before(float): ", t_before)
        # print("time binary: ", t_binary)
        print("time mal: ", t_mal)
        # input_int=np.loadtxt('input_im2col_conv_256.npy') # load binarized data
        # print(input_int.shape)
        # exit()
        # im2col_conv(input, w1_cnn_train)
        # gemm_conv(input, w1_cnn_train)
        print("end conv")

        # x = torch.sign(torch.rand(128000*3*8*8, 16)-0.5) #db
        # input = x
        # im2col_pooling(input)
        # gemm_pooling(input)
        print("end pooling")

        # x = torch.sign(torch.rand(2560000, 320)-0.5)
        # input = x
        # x = torch.sign(torch.rand(10, 320)-0.5)
        # weight = x
        # # im2col_linear(input, weight)
        # gemm_linear(input, weight)
        print("end linear")
        exit()
        '''
        

        '''
        #----convert matrix to bitmap? how? compute a
        # len_bit = w_h * w_w * w_c
        ##1.二进制转化为整数 2.整数存进数组 3.数组存进表里
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
        # np.save('input_int.npy',input_int)  #save binarized data list to current dir
        # np.save('input_int.npy',input_int)
        # np.save('input_int_uint8.npy',input_int)
        np.save('input_int_7x7.npy',input_int)
        exit()
        '''

        '''
        input_int=np.load('input_int.npy', allow_pickle=True) # load binarized data
        print(input_int.shape)
        len_all = input_int.shape[0]
        batch_index = 1
        len_tmp = int(len_all/batch_index)
        print("len_tmp: ",len_tmp)
        # input_int=input_int.tolist()
        input_int=input_int[:len_tmp,:2].tolist()
        '''

        
        '''
        for i in range(len_tmp):
            row_list = []
            for j in range(batch_index):
                row_list.append(input_int[i+j*len_tmp][0])
                row_list.append(input_int[i+j*len_tmp][1])
            input_batch.append(row_list)
        '''

        # print(input_batch[:5])
        # exit()

        # exit()
        # print('1')
        # print(input_int[:10])

        # sum=0
        
        # for i in range(len(input_int)):
        #     if input_int[i][2] == 0:
        #         sum=sum+1
        # print(sum)
        # 去 0
        
        # kernel_int = []
        # len_a = w1_cnn_train.shape[1]
        # for i in range (len_a):
        #     test_a = w1_cnn_train[:,i]
        #     len_test_a = len(test_a)
        #     int_test_a = array2int(len_test_a, test_a)
        #     kernel_int.append([int(int_test_a)])
        
        '''
        # clickhouse
        t1 = time.time()
        db = Client(host='localhost')
        clean_bit_input = "DROP TABLE IF EXISTS bit_input;"
        db.execute(clean_bit_input)

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
        
        print(time.time()-t1)
        print("end create_feature_sql")
        '''
        
        # = -（2*popcount（a）- len（a）） len已知（从kernel的长度已知）
        '''
        # postgresql
        t1 = time.time()
        db = DB()
        clean_bit_input = "DROP TABLE IF EXISTS bit_input;"
        db.executeone(clean_bit_input)
        db.conn.commit()
        create_bit_input="CREATE TABLE IF NOT EXISTS bit_input (order_id integer, bit_a integer, bit_b integer, bit_len integer, PRIMARY KEY(order_id));"
        db.executeone(create_bit_input)
        db.conn.commit()
        db.executemany("insert into bit_input values(%s,%s,%s,%s) ", input_int[:2621440])
        db.conn.commit()
        print(time.time()-t1)
        print("end create_feature_sql")
        '''
        # exit()

        
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

def im2col_conv (array_input, array_weight):
    if(os.path.isfile("input_im2col_conv_256.npy")):
        os.remove("input_im2col_conv_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    t1 = time.time()
    batch_num = 128
    with open("input_im2col_conv_256.npy", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range (b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
                for j in range (array_weight.shape[1]):
                    for ind_j in range (array_input.shape[1]):
                        input_int.append([int(ind_j),array_input[i][ind_j],array_weight[ind_j][j]])
            input_int=np.array(input_int)
            np.savetxt(f, input_int)
    
    # np.save('input_im2col_conv_256.npy',input_int)
    print("data time: ", time.time()-t1)
    return

def gemm_conv (array_input, array_weight):
    if(os.path.isfile("input_gemm_conv_256.npy")):
        os.remove("input_gemm_conv_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    num_blocks = 3
    len1 = array_input.shape[1]/num_blocks
    t1 = time.time()
    batch_num = 128
    with open("input_gemm_conv_256.npy", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range (b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
                for j in range (array_weight.shape[1]):
                    for ind_j in range (int(len1)):
                        input_int.append([int(ind_j),array_input[i][0+ind_j*num_blocks],array_weight[0+ind_j*num_blocks][j],array_input[i][1+ind_j*num_blocks],array_weight[1+ind_j*num_blocks][j],array_input[i][2+ind_j*num_blocks],array_weight[2+ind_j*num_blocks][j]])
            input_int=np.array(input_int)
            np.save(f,input_int)
    print("data time: ", time.time()-t1)
    return

def im2col_pooling (array_input):
    if(os.path.isfile("input_im2col_pooling_256.npy")):
        os.remove("input_im2col_pooling_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    batch_num = 128
    with open("input_im2col_pooling_256.npy", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
                for j in range(array_input.shape[1]):
                    input_int.append([j,array_input[i][j]])
            input_int=np.array(input_int)
            np.save(f,input_int)
    # np.save('input_im2col_pooling_256.npy',input_int)  #save binarized data list to current dir
    return 

def im2col_linear (array_input, array_weight):
    if(os.path.isfile("input_im2col_linear_256.npy")):
        os.remove("input_im2col_linear_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    batch_num = 12800
    with open("input_im2col_linear_256.npy", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
                for j in range(array_weight.shape[0]):
                    for ind_j in range(array_input.shape[1]):
                        input_int.append([j,array_input[i][ind_j],array_weight[j][ind_j]])
            input_int=np.array(input_int)
            np.save(f,input_int)
    # np.save('input_im2col_linear_256.npy',input_int)  #save binarized data list to current dir
    return 

def gemm_pooling (array_input):
    if(os.path.isfile("input_gemm_pooling_256.npy")):
        os.remove("input_gemm_pooling_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    num_blocks = 3
    len1 = array_input.shape[1]/num_blocks
    batch_num = 128
    with open("input_gemm_pooling_256.npy", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
                for ind_j in range(int(len1)):
                    input_int.append([int(ind_j),array_input[i][0+ind_j*num_blocks],array_input[i][1+ind_j*num_blocks],array_input[i][2+ind_j*num_blocks]])
            input_int=np.array(input_int)
            np.save(f,input_int)
    # np.save('input_gemm_pooling_256.npy',input_int)  #save binarized data list to current dir
    return 

def gemm_linear (array_input, array_weight):
    if(os.path.isfile("input_gemm_linear_256.npz")):
        os.remove("input_gemm_linear_256.npz")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    num_blocks = 3
    len1 = array_input.shape[1]/num_blocks
    batch_num = 12800
    batch_size = int(array_input.shape[0]/batch_num)
    with open("input_gemm_linear_256.npz", "ab") as f:
        for b in range(batch_num):
            input_int = []
            for i in range(b*batch_size,(b+1)*batch_size):
                for j in range(array_weight.shape[0]):
                    for ind_j in range(int(len1)):
                        input_int.append([int(ind_j),int(array_input[i][0+ind_j*num_blocks]),array_weight[j][0+ind_j*num_blocks],array_input[i][1+ind_j*num_blocks],array_weight[j][1+ind_j*num_blocks],array_input[i][2+ind_j*num_blocks],array_weight[j][2+ind_j*num_blocks]])
            input_int=np.array(input_int).astype(int)
            np.save(f,input_int)
    # np.save('input_gemm_linear_256.npy',input_int)  #save binarized data list to current dir
    return 
    
        
       
        
    