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

# m2: dl2sql m4: smartlite
def db_test():
    x = torch.sign(torch.rand(1280000, 2)-0.5) #db
    input = x
    input_int = []
    for i in range(input.shape[0]):
        input_int.append([int(input[i][0]), int(input[i][1])])
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS test;"
    db.execute(clean_bit_input)
    create_bit_input="CREATE TABLE IF NOT EXISTS test (bit_1 Int8, bit_2 Int8, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into test values ", input_int) 
    return

def m2_conv ():
    print("m2_conv")
    array_input=np.load('m2_conv_cifar10_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m2_conv_cifar10_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_conv;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_conv (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[1]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[ind_j][j])])
        db.execute("insert into m2_conv values ", input_int)
    db.execute("OPTIMIZE TABLE m2_conv FINAL; ")
    db.execute("DROP TABLE IF EXISTS m2_conv_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m2_conv_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_conv;")
    print("time: ", time.time()-t1)
    return 

def m2_pool ():
    print("m2_pool")
    array_input=np.load('m2_pool_cifar10.npy', allow_pickle=True)
    array_input=array_input.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_pooling;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_pooling (bit_1 UInt32, bit_2 float, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input)
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_input.shape[1]):
                input_int.append([int(i),float(array_input[i][j])])
        db.execute("insert into m2_pooling values ", input_int)
    db.execute("OPTIMIZE TABLE m2_pooling FINAL; ")
    db.execute("DROP TABLE IF EXISTS m2_pooling_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m2_pooling_mem (bit_1 UInt32, bit_2 float) engine = Memory as select * from m2_pooling;")
    print("time: ", time.time()-t1)
    return 

def m2_linear ():
    print("m2_linear")
    array_input=np.load('m2_linear_cifar10_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m2_linear_cifar10_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_linear;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_linear (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[0]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[j][ind_j])])
        db.execute("insert into m2_linear values ", input_int)
    db.execute("OPTIMIZE TABLE m2_linear FINAL; ")
    db.execute("DROP TABLE IF EXISTS m2_linear_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m2_linear_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_linear;")
    print("time: ", time.time()-t1)
    return 

def m4_conv ():
    print("m4_conv")
    array_input=np.load('m4_conv_cifar10_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m4_conv_cifar10_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    input_int = m4_conv_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_conv_cifar10.npy',input_int)

    input_int=np.load('m4_conv_cifar10.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_conv;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_conv (bit_1 UInt32, bit_2 UInt32, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_conv values ", input_int)
    db.execute("OPTIMIZE TABLE m4_conv FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_conv_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_conv_mem (bit_1 UInt32, bit_2 UInt32) engine = Memory as select * from m4_conv;")
    print("time: ", time.time()-t1)
    return 

def m4_pool ():
    print("m4_pool")
    array_input=np.load('m4_pool_cifar10.npy', allow_pickle=True)
    array_input=array_input.tolist()
    input_int = m4_pool_generator(array_input)
    input_int=np.array(input_int)
    np.save('m4_pool_cifar10.npy',input_int)

    input_int=np.load('m4_pool_cifar10.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_pool;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_pool (bit_1 UInt16, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_pool values ", input_int)
    db.execute("OPTIMIZE TABLE m4_pool FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_pool_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_pool_mem (bit_1 UInt16) engine = Memory as select * from m4_pool;")
    print("time: ", time.time()-t1)
    return 

def m4_linear ():
    print("m4_linear")
    array_input=np.load('m4_linear_cifar10_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m4_linear_cifar10_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    input_int = m4_linear_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_linear_cifar10.npy',input_int)

    input_int=np.load('m4_linear_cifar10.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_linear;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_linear (bit_1 UInt256, bit_2 UInt256, bit_3 UInt64, bit_4 UInt64, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_linear values ", input_int)
    db.execute("OPTIMIZE TABLE m4_linear FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_linear_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_linear_mem (bit_1 UInt256, bit_2 UInt256, bit_3 UInt64, bit_4 UInt64) engine = Memory as select * from m4_linear;")
    print("time: ", time.time()-t1)
    return 

def m4_conv_generator (array_input, array_weight):
    input_int = []
    for i in range(array_input.shape[0]):
        sum_input = 0
        for ind_i in range(array_input.shape[1]):
            if array_input[i][ind_i] > 0:
                sum_input = sum_input + pow(2, ind_i)
        for j in range(array_weight.shape[1]):
            sum_weight = 0
            for ind_j in range(array_weight.shape[0]):
                if array_weight[ind_j][j] > 0:
                    sum_weight = sum_weight + pow(2, ind_j)
            input_int.append([int(sum_input),int(sum_weight)])
    return input_int

def m4_pool_generator (array_input):
    input_int = []
    for i in range(array_input.shape[0]):
        sum_input = 0
        for j in range(array_input.shape[1]):
            if array_input[i][j] > 0:
                sum_input = sum_input + pow(2, j)
        input_int.append([int(sum_input)])
    return input_int

def m4_linear_generator (array_input, array_weight):
    input_int = []
    len1=256
    for i in range(array_input.shape[0]):
        sum_input = 0
        sum_input_sht = 0
        for ind_i in range(len1):
            if array_input[i][ind_i] > 0:
                sum_input = sum_input + pow(2, ind_i)
        for ind_i in range(len1,array_input.shape[1]):
            if array_input[i][ind_i] > 0:
                sum_input_sht = sum_input_sht + pow(2, ind_i-len1)

        for j in range(array_weight.shape[0]):
            sum_weight = 0
            sum_weight_sht = 0
            for ind_j in range(len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight = sum_weight + pow(2, ind_j)
            for ind_j in range(len1, array_weight.shape[1]):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht = sum_weight_sht + pow(2, ind_j-len1)
            input_int.append([int(sum_input),int(sum_weight),int(sum_input_sht),int(sum_weight_sht)])
    return input_int

if __name__ == '__main__':
    #------------------cifar10---------------------#
    # m2_conv()
    # m2_pool()
    # m2_linear()
    input = torch.rand(500, 16, 3, 32, 32)
    avgpool = nn.AvgPool2d(4)
    maxpool = nn.MaxPool2d(4)
    # # print(input[:5])
    t1 = time.time()
    for i in range (500):
        val_pool = avgpool(input[i])
    t2 = time.time()
    print("Pool time: ", t2-t1)
    cpu_num = 8
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    print("cpu_num: ", cpu_num)
        
    input = torch.rand(10000, 16, 320)
    # print(x.shape)
    # len_all = x.shape[0]
    # len_tmp = int(len_all/batch_index)
    # input = x[:len_tmp,:256]
    # input = x
    weight = torch.rand(10, 320)
    # weight = x
    # t1 = time.time()
    # for i in range (10000):
    #     out = nn.functional.linear(input[i], weight)
    # # if not self.bias is None:
    # #         self.bias.org=self.bias.data.clone()
    # #         out += self.bias.view(1, -1).expand_as(out)
    # t2 = time.time()
    # print("linear time: ", t2-t1)
    exit()
    
        
'''
DROP TABLE IF EXISTS m2_conv;
DROP TABLE IF EXISTS m2_pooling;
DROP TABLE IF EXISTS m2_linear;
DROP TABLE IF EXISTS m4_conv;
DROP TABLE IF EXISTS m4_pool;
DROP TABLE IF EXISTS m4_linear;
'''