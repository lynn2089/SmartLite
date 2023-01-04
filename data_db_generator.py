import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import sys
sys.path.append('..')
import comm as conveter
import numpy as np
import time
import os
from clickhouse_driver import Client

def db_mcmm():
    input_int=np.load('input_int.npy', allow_pickle=True) # load binarized data
    print(input_int.shape)
    len_all = input_int.shape[0]
    batch_index = 1
    len_tmp = int(len_all/batch_index)
    print("len_tmp: ",len_tmp)
    # input_int=input_int.tolist()
    input_int=input_int[:len_tmp,:2].tolist()

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

    return 

def comm_pooling (array_input):
    if(os.path.isfile("input_comm_pooling_256.npy")):
        os.remove("input_comm_pooling_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    batch_num = 128
    # with open("input_comm_pooling_256.npy", "ab") as f:
    #     for b in range(batch_num):
    #         input_int = []
    #         for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
    #             for j in range(array_input.shape[1]):
    #                 input_int.append([i,array_input[i][j]])
    #         input_int=np.array(input_int)
    #         np.save(f,input_int)
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS comm_pooling;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS comm_pooling (bit_1 UInt32, bit_2 Int8, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input)
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_input.shape[1]):
                input_int.append([int(i),int(array_input[i][j])])
        db.execute("insert into comm_pooling values ", input_int)
    db.execute("OPTIMIZE TABLE comm_pooling FINAL; ")
    print("time: ", time.time()-t1)
    # np.save('input_comm_pooling_256.npy',input_int)  #save binarized data list to current dir
    return 

def comm_linear (array_input, array_weight):
    if(os.path.isfile("input_comm_linear_256.npy")):
        os.remove("input_comm_linear_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    batch_num = 1280
    # with open("input_comm_linear_256.npy", "ab") as f:
        # for b in range(batch_num):
        #     input_int = []
        #     for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
        #         for j in range(array_weight.shape[0]):
        #             for ind_j in range(array_input.shape[1]):
        #                 input_int.append([i,array_input[i][ind_j],array_weight[j][ind_j]])
            # input_int=np.array(input_int)
            # np.save(f,input_int)
    # np.save('input_comm_linear_256.npy',input_int)  #save binarized data list to current dir
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS comm_linear;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS comm_linear (bit_1 UInt32, bit_2 Int8, bit_3 Int8, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for j in range(array_weight.shape[0]):
                for ind_j in range(array_input.shape[1]):
                    input_int.append([int(i),int(array_input[i][ind_j]),int(array_weight[j][ind_j])])
        db.execute("insert into comm_linear values ", input_int)
    db.execute("OPTIMIZE TABLE comm_linear FINAL; ")
    print("time: ", time.time()-t1)
    return 

def mcmm_pooling (array_input):
    if(os.path.isfile("input_mcmm_pooling_256.npy")):
        os.remove("input_mcmm_pooling_256.npy")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    num_blocks = 3
    len1 = array_input.shape[1]/num_blocks
    batch_num = 128
    # print(int(array_input.shape[0]/batch_num))
    # exit()
    # with open("input_mcmm_pooling_256.npy", "ab") as f:
    #     for b in range(batch_num):
    #         input_int = []
    #         for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
    #             for ind_j in range(int(len1)):
    #                 input_int.append([int(i),array_input[i][0+ind_j*num_blocks],array_input[i][1+ind_j*num_blocks],array_input[i][2+ind_j*num_blocks]])
    #         input_int=np.array(input_int).astype(int)
    #         np.save(f,input_int)
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS mcmm_pooling;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS mcmm_pooling (bit_1 UInt32, bit_2 Int8, bit_3 Int8, bit_4 Int8, PRIMARY KEY(bit_1)) engine = MergeTree();"
    db.execute(create_bit_input) 
    for b in range(batch_num):
        input_int = []
        for i in range(b*int(array_input.shape[0]/batch_num),(b+1)*int(array_input.shape[0]/batch_num)):
            for ind_j in range(int(len1)):
                input_int.append([int(i),int(array_input[i][0+ind_j*num_blocks]),int(array_input[i][1+ind_j*num_blocks]),int(array_input[i][2+ind_j*num_blocks])])
        # input_int=np.array(input_int).astype(int)
        db.execute("insert into mcmm_pooling values ", input_int)
    db.execute("OPTIMIZE TABLE mcmm_pooling FINAL; ")
    print("time: ", time.time()-t1)
    return 

def mcmm_linear (array_input, array_weight):
    if(os.path.isfile("input_mcmm_linear_256.npz")):
        os.remove("input_mcmm_linear_256.npz")
        print("File Deleted successfully")
    else:
        print("File does not exist")
    num_blocks = 3
    len1 = array_input.shape[1]/num_blocks
    batch_num = 12800
    batch_size = int(array_input.shape[0]/batch_num)
    # with open("input_mcmm_linear_256.npz", "ab") as f:
    #     for b in range(batch_num):
    #         input_int = []
    #         for i in range(b*batch_size,(b+1)*batch_size):
    #             for j in range(array_weight.shape[0]):
    #                 for ind_j in range(int(len1)):
    #                     input_int.append([int(ind_j),int(array_input[i][0+ind_j*num_blocks]),array_weight[j][0+ind_j*num_blocks],array_input[i][1+ind_j*num_blocks],array_weight[j][1+ind_j*num_blocks],array_input[i][2+ind_j*num_blocks],array_weight[j][2+ind_j*num_blocks]])
    #         input_int=np.array(input_int).astype(int)
    #         np.save(f,input_int)
    # np.save('input_mcmm_linear_256.npy',input_int)  #save binarized data list to current dir
    t1 = time.time()
    db = Client(host='localhost')
    clean_bit_input = "DROP TABLE IF EXISTS mcmm_linear;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS mcmm_linear (bit_1 UInt16, bit_2 Int8, bit_3 Int8, bit_4 Int8, bit_5 Int8, bit_6 Int8, bit_7 Int8 ) engine = MergeTree() order by (bit_1);"
    db.execute(create_bit_input)
    for b in range(batch_num):
        input_int = []
        for i in range(b*batch_size,(b+1)*batch_size):
            for j in range(array_weight.shape[0]):
                for ind_j in range(int(len1)):
                    input_int.append([int(i),int(array_input[i][0+ind_j*num_blocks]),int(array_weight[j][0+ind_j*num_blocks]),int(array_input[i][1+ind_j*num_blocks]),int(array_weight[j][1+ind_j*num_blocks]),int(array_input[i][2+ind_j*num_blocks]),int(array_weight[j][2+ind_j*num_blocks])])
        # input_int=np.array(input_int).astype(int)
        db.execute("insert into mcmm_linear values ", input_int)
    print("time: ", time.time()-t1)
    db.execute("OPTIMIZE TABLE mcmm_linear FINAL; ")
    return 
    
    