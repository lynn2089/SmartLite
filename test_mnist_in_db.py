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
def m2_conv ():
    print("m2_conv")
    array_input=np.load('m2_conv_mnist_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m2_conv_mnist_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_conv_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_conv_mnist (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1, bit_2, bit_3)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[1]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[ind_j][j])])
        db.execute("insert into m2_conv_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m2_conv_mnist FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_conv_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_conv_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_conv_mnist;")
    print("time: ", time.time()-t1)
    return 

def m2_pool ():
    print("m2_pool")
    array_input=np.load('m2_pool_mnist.npy', allow_pickle=True)
    array_input=array_input.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_pooling_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_pooling_mnist (bit_1 UInt32, bit_2 float, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input)
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_input.shape[1]):
                input_int.append([int(i),float(array_input[i][j])])
        db.execute("insert into m2_pooling_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m2_pooling_mnist FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_pooling_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_pooling_mem (bit_1 UInt32, bit_2 float) engine = Memory as select * from m2_pooling_mnist;")
    print("time: ", time.time()-t1)
    return 

def m2_linear ():
    print("m2_linear")
    array_input=np.load('m2_linear_mnist_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m2_linear_mnist_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_linear_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_linear_mnist (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1, bit_2, bit_3)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[0]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[j][ind_j])])
        db.execute("insert into m2_linear_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m2_linear_mnist FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_linear_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_linear_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_linear_mnist;")
    print("time: ", time.time()-t1)
    return 

def m4_conv ():
    print("m4_conv")
    array_input=np.load('m4_conv_mnist_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m4_conv_mnist_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    input_int = m4_conv_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_conv_mnist.npy',input_int)

    input_int=np.load('m4_conv_mnist.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_conv_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_conv_mnist (bit_1 UInt32, bit_2 UInt32, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_conv_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m4_conv_mnist FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_conv_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_conv_mem (bit_1 UInt32, bit_2 UInt32) engine = Memory as select * from m4_conv_mnist;")
    print("time: ", time.time()-t1)
    return 

def m4_pool ():
    print("m4_pool")
    array_input=np.load('m4_pool_mnist.npy', allow_pickle=True)
    array_input=array_input.tolist()
    input_int = m4_pool_generator(array_input)
    input_int=np.array(input_int)
    np.save('m4_pool_mnist.npy',input_int)

    input_int=np.load('m4_pool_mnist.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_pool_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_pool_mnist (bit_1 UInt8, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_pool_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m4_pool_mnist FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_pool_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_pool_mem (bit_1 UInt8) engine = Memory as select * from m4_pool_mnist;")
    print("time: ", time.time()-t1)
    return 

def m4_linear ():
    print("m4_linear")
    array_input=np.load('m4_linear_mnist_height.npy', allow_pickle=True)
    array_input=array_input.tolist()
    array_weight=np.load('m4_linear_mnist_weight.npy', allow_pickle=True)
    array_weight=array_weight.tolist()
    input_int = m4_linear_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_linear_mnist.npy',input_int)

    input_int=np.load('m4_linear_mnist.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_linear_mnist;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_linear_mnist (bit_1 UInt256, bit_2 UInt256, \
    bit_3 UInt256, bit_4 UInt256, \
    PRIMARY KEY(bit_1, bit_2, bit_3, bit_4)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_linear_mnist values ", input_int)
    db.execute("OPTIMIZE TABLE m4_linear_mnist FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_linear_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_linear_mem (bit_1 UInt256, bit_2 UInt256, \
    bit_3 UInt256, bit_4 UInt256 \
    ) engine = Memory as select * from m4_linear_mnist;")
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
        sum_input_sht_1 = 0
        for ind_i in range(len1):
            if array_input[i][ind_i] > 0:
                sum_input = sum_input + pow(2, ind_i)
        for ind_i in range(len1,2*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_1 = sum_input_sht_1 + pow(2, ind_i-len1)

        for j in range(array_weight.shape[0]):
            sum_weight = 0
            sum_weight_sht_1 = 0
            for ind_j in range(len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight = sum_weight + pow(2, ind_j)
            for ind_j in range(len1,2*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_1 = sum_weight_sht_1 + pow(2, ind_j-len1)
            input_int.append([int(sum_input),int(sum_weight), \
            int(sum_input_sht_1),int(sum_weight_sht_1)])
    return input_int
