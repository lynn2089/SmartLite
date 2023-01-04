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
    array_input = torch.rand(200704, 147)-0.5 #db
    array_weight = torch.rand(147,64)-0.5 #db
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_conv_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_conv_imagenet (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1, bit_2, bit_3)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[1]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[ind_j][j])])
        db.execute("insert into m2_conv_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m2_conv_imagenet FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_conv_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_conv_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_conv_imagenet;")
    print("time: ", time.time()-t1)
    return 

def m2_pool ():
    print("m2_pool")
    array_input = torch.rand(40410800, 16)-0.5 #db
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_pooling_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_pooling_imagenet (bit_1 UInt32, bit_2 float, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input)
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_input.shape[1]):
                input_int.append([int(i),float(array_input[i][j])])
        db.execute("insert into m2_pooling_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m2_pooling_imagenet FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_pooling_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_pooling_mem (bit_1 UInt32, bit_2 float) engine = Memory as select * from m2_pooling_imagenet;")
    print("time: ", time.time()-t1)
    return 

def m2_linear ():
    print("m2_linear")
    array_input = torch.rand(8000, 2048)-0.5 #db
    array_weight = torch.rand(10, 2048)-0.5 #db
    batch_num = 128
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m2_linear_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m2_linear_imagenet (bit_1 UInt32, bit_2 float, bit_3 float, PRIMARY KEY(bit_1, bit_2, bit_3)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[0]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),float(array_input[i][ind_j]),float(array_weight[j][ind_j])])
        db.execute("insert into m2_linear_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m2_linear_imagenet FINAL; ")
    # db.execute("DROP TABLE IF EXISTS m2_linear_mem; ")
    # db.execute("CREATE TABLE IF NOT EXISTS m2_linear_mem (bit_1 UInt32, bit_2 float, bit_3 float) engine = Memory as select * from m2_linear_imagenet;")
    print("time: ", time.time()-t1)
    return 

def m4_conv ():
    print("m4_conv")
    array_input = torch.sign(torch.rand(200704, 27)-0.5) #db
    array_weight = torch.sign(torch.rand(27, 4)-0.5) #db
    input_int = m4_conv_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_conv_imagenet.npy',input_int)

    input_int=np.load('m4_conv_imagenet.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_conv_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_conv_imagenet (bit_1 UInt32, bit_2 UInt32, PRIMARY KEY(bit_1, bit_2)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_conv_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m4_conv_imagenet FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_conv_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_conv_mem (bit_1 UInt32, bit_2 UInt32) engine = Memory as select * from m4_conv_imagenet;")
    print("time: ", time.time()-t1)
    return 

def m4_pool ():
    print("m4_pool")
    array_input = torch.sign(torch.rand(25088000, 16)-0.5) #db
    input_int = m4_pool_generator(array_input)
    input_int=np.array(input_int)
    np.save('m4_pool_imagenet.npy',input_int)

    input_int=np.load('m4_pool_imagenet.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_pool_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_pool_imagenet (bit_1 UInt16, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_pool_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m4_pool_imagenet FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_pool_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_pool_mem (bit_1 UInt16) engine = Memory as select * from m4_pool_imagenet;")
    print("time: ", time.time()-t1)
    return 

def m4_linear ():
    print("m4_linear")
    array_input = torch.sign(torch.rand(8000, 2048)-0.5) #db
    array_weight = torch.sign(torch.rand(10, 2048)-0.5) #db
    input_int = m4_linear_generator(array_input, array_weight)
    input_int=np.array(input_int)
    np.save('m4_linear_imagenet.npy',input_int)

    input_int=np.load('m4_linear_imagenet.npy', allow_pickle=True)
    input_int=input_int.tolist()
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS m4_linear_imagenet;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS m4_linear_imagenet (bit_1 UInt256, bit_2 UInt256, \
    bit_3 UInt256, bit_4 UInt256, \
    bit_5 UInt256, bit_6 UInt256, \
    bit_7 UInt256, bit_8 UInt256, \
    bit_9 UInt256, bit_10 UInt256, \
    bit_11 UInt256, bit_12 UInt256, \
    bit_13 UInt256, bit_14 UInt256, \
    bit_15 UInt256, bit_16 UInt256, \
    PRIMARY KEY(bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, \
    bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16)) engine = MergeTree();"
    db.execute(create_bit_input) 
    db.execute("insert into m4_linear_imagenet values ", input_int)
    db.execute("OPTIMIZE TABLE m4_linear_imagenet FINAL; ")
    db.execute("DROP TABLE IF EXISTS m4_linear_mem; ")
    db.execute("CREATE TABLE IF NOT EXISTS m4_linear_mem (bit_1 UInt256, bit_2 UInt256, \
    bit_3 UInt256, bit_4 UInt256, \
    bit_5 UInt256, bit_6 UInt256, \
    bit_7 UInt256, bit_8 UInt256, \
    bit_9 UInt256, bit_10 UInt256, \
    bit_11 UInt256, bit_12 UInt256, \
    bit_13 UInt256, bit_14 UInt256, \
    bit_15 UInt256, bit_16 UInt256 \
    ) engine = Memory as select * from m4_linear_imagenet;")
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
        sum_input_sht_2 = 0
        sum_input_sht_3 = 0
        sum_input_sht_4 = 0
        sum_input_sht_5 = 0
        sum_input_sht_6 = 0
        sum_input_sht_7 = 0
        for ind_i in range(len1):
            if array_input[i][ind_i] > 0:
                sum_input = sum_input + pow(2, ind_i)
        for ind_i in range(len1,2*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_1 = sum_input_sht_1 + pow(2, ind_i-len1)
        for ind_i in range(2*len1,3*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_2 = sum_input_sht_2 + pow(2, ind_i-2*len1)
        for ind_i in range(3*len1,4*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_3 = sum_input_sht_3 + pow(2, ind_i-3*len1)
        for ind_i in range(4*len1,5*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_4 = sum_input_sht_4 + pow(2, ind_i-4*len1)
        for ind_i in range(5*len1,6*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_5 = sum_input_sht_5 + pow(2, ind_i-5*len1)
        for ind_i in range(6*len1,7*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_6 = sum_input_sht_6 + pow(2, ind_i-6*len1)
        for ind_i in range(7*len1,8*len1):
            if array_input[i][ind_i] > 0:
                sum_input_sht_7 = sum_input_sht_7 + pow(2, ind_i-7*len1)

        for j in range(array_weight.shape[0]):
            sum_weight = 0
            sum_weight_sht_1 = 0
            sum_weight_sht_2 = 0
            sum_weight_sht_3 = 0
            sum_weight_sht_4 = 0
            sum_weight_sht_5 = 0
            sum_weight_sht_6 = 0
            sum_weight_sht_7 = 0
            for ind_j in range(len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight = sum_weight + pow(2, ind_j)
            for ind_j in range(len1,2*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_1 = sum_weight_sht_1 + pow(2, ind_j-len1)
            for ind_j in range(2*len1, 3*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_2 = sum_weight_sht_2 + pow(2, ind_j-2*len1)
            for ind_j in range(3*len1, 4*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_3 = sum_weight_sht_3 + pow(2, ind_j-3*len1)
            for ind_j in range(4*len1, 5*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_4 = sum_weight_sht_4 + pow(2, ind_j-4*len1)
            for ind_j in range(5*len1, 6*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_5 = sum_weight_sht_5 + pow(2, ind_j-5*len1)
            for ind_j in range(6*len1, 7*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_6 = sum_weight_sht_6 + pow(2, ind_j-6*len1)
            for ind_j in range(7*len1, 8*len1):
                if array_weight[j][ind_j] > 0:
                    sum_weight_sht_7 = sum_weight_sht_7 + pow(2, ind_j-7*len1)
            input_int.append([int(sum_input),int(sum_weight), \
            int(sum_input_sht_1),int(sum_weight_sht_1), \
            int(sum_input_sht_2),int(sum_weight_sht_2), \
            int(sum_input_sht_3),int(sum_weight_sht_3), \
            int(sum_input_sht_4),int(sum_weight_sht_4), \
            int(sum_input_sht_5),int(sum_weight_sht_5), \
            int(sum_input_sht_6),int(sum_weight_sht_6), \
            int(sum_input_sht_7),int(sum_weight_sht_7)])
    return input_int
