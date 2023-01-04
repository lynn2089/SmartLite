from tkinter import S
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
import random
from clickhouse_driver import Client

def reshape():
    n=64
    
    # conv1
    h1=27
    kh = 11
    c_out = 96
    pool = 4 
    # len_feature_map = kh*kh*c_in
    len_feature_map = 256
    len_group=1
    len_all = len_feature_map*len_group

    cv1 = [] #value of conv1
    for k in range(n):
        cv1_1 = []
        cv1_2 = []
        for i in range(h1*h1):
            x = random.randint(0,pow(2,len_feature_map))
            cv1_1.append(x)
        for j in range(c_out):
            cv1_2.extend(cv1_1)
        cv1.extend(cv1_2)
    cv1 = np.array(cv1)

    bn1=[]
    for i in range(n*c_out):
        for j in range(h1*h1):
            bn1.append([i])
    bn1 = torch.tensor(bn1)
    pool1=[]
    for i in range(n*h1*h1*int(c_out/pool)):
        for j in range(pool):
            pool1.append([i])
    pool1 = torch.tensor(pool1)
    tmp = []
    for i in range(c_out):
        x = random.randint(0,pow(2,len_feature_map))
        for j in range(h1*h1):
            tmp.append(x)
    tmp = np.array(tmp)
    k1 = tmp
    for k in range (n-1):
        k1 = np.concatenate((k1,tmp),axis=0)

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS conv1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS conv1 (bn1 UInt16, pool1 UInt32, cv1 UInt256, k1 UInt256, PRIMARY KEY (bn1, pool1, cv1, k1)) engine = MergeTree();"
    db.execute(create_bit_input)
    conv1 = []
    for i in range (n*c_out*h1*h1):
        conv1.append([bn1[i],pool1[i],cv1[i],k1[i]])
    db.execute("insert into conv1 values ", conv1) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    db.execute("OPTIMIZE TABLE conv1 FINAL;") 
    db.execute("drop table if EXISTS conv1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv1_2(bn1 UInt16, pool1 UInt32, cv1 UInt256, k1 UInt256) engine = Memory as \
        SELECT bn1, pool1, cv1, k1 \
        from conv1;") 

    
    # conv2
    h2 = 13  #output featuremap
    kh = 5
    c_in = 96
    c_out = 256
    pool = 4 
    # len_feature_map = kh*kh*c_in
    len_feature_map = 256
    len_group=18
    len_all = len_feature_map*len_group

    x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
    cv2 = x #value of conv2
    for i in range (n-1):
        x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
        cv2 = torch.cat((cv2, x), 0)
    cn_id2=[]
    for i in range(n*h2*h2):
        for j in range(len_all):
            cn_id2.append([i])
    cn_id2 = torch.tensor(cn_id2)
    im_id = []
    for i in range(n):
        for j in range(h2*h2*len_all):
            im_id.append([i])
    im_id = torch.tensor(im_id)
    bin_id = []
    for i in range(n*h2*h2*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)

    data2 = torch.cat((bin_id, cn_id2, im_id, cv2), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data2 (bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8, PRIMARY KEY (bin_id, cn_id, im_id)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data2 values ", data2) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    
    kid2=[]
    for i in range(c_out):
        kid2.append([i])
    kid2 = torch.tensor(kid2)
    pool2=[]
    for i in range(int(c_out/pool)):
        for j in range(pool):
            pool2.append([i])
    pool2 = torch.tensor(pool2)
    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kernel2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kernel2 (pool2 UInt8, kid2 UInt16, kv2 Array(UInt256), PRIMARY KEY (pool2, kid2)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([pool2[i],kid2[i],kv2[i]])
    db.execute("insert into kernel2 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")

    db.execute("OPTIMIZE TABLE data2 FINAL;") 
    db.execute("drop table if EXISTS conv2_data2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv2_data2_2(bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8) engine = Memory as \
    SELECT bin_id, cn_id, im_id, cv2 \
    from data2;") 

    db.execute("OPTIMIZE TABLE kernel2 FINAL;") 
    db.execute("drop table if EXISTS kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kernel2_2(pool2 UInt8, kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT pool2, kid2, kv2 \
    from kernel2;") 

    db.execute("drop table if EXISTS conv2_kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv2_kernel2_2(pool2 UInt8, kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT pool2, kid2, kv2 \
    from kernel2;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv2_data2_2 group by bin_id) \
    group by cn_id2;") 

    db.execute("drop table if EXISTS conv2_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv2_t_l3 engine = Memory as select cn_id2, pool2, im_id2, kid2, data2, kv2 from t_l2, kernel2_2;") 

    # conv3
    # weight:  torch.Size([1152, 576, 3, 3])
    # input:  torch.Size([64, 576, 13, 13])
    # input:  torch.Size([64, 1152, 13, 13])
    h2 = 13  #output featuremap
    kh = 3
    c_in = 256
    c_out = 384
    # len_feature_map = kh*kh*c_in
    len_feature_map = 256
    len_group=9
    len_all = len_feature_map*len_group 

    x= torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
    cv2 = x #value of conv2
    for i in range (n-1):
        x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
        cv2 = torch.cat((cv2, x), 0)
    cn_id2=[]
    for i in range(n*h2*h2):
        for j in range(len_all):
            cn_id2.append([i])
    cn_id2 = torch.tensor(cn_id2)
    im_id = []
    for i in range(n):
        for j in range(h2*h2*len_all):
            im_id.append([i])
    im_id = torch.tensor(im_id)
    bin_id = []
    for i in range(n*h2*h2*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)

    data2 = torch.cat((bin_id, cn_id2, im_id, cv2), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data2 (bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8, PRIMARY KEY (bin_id, cn_id, im_id)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data2 values ", data2) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    kid2=[]
    for i in range(c_out):
        kid2.append([i])
    kid2 = torch.tensor(kid2)
    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kernel2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kernel2 (kid2 UInt16, kv2 Array(UInt256), PRIMARY KEY (kid2)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([kid2[i],kv2[i]])
    db.execute("insert into kernel2 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")


    db.execute("OPTIMIZE TABLE data2 FINAL;") 
    db.execute("drop table if EXISTS conv3_data2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv3_data2_2(bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8) engine = Memory as \
    SELECT bin_id, cn_id, im_id, cv2 \
    from data2;") 

    db.execute("OPTIMIZE TABLE kernel2 FINAL;") 
    db.execute("drop table if EXISTS kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kernel2_2(kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT kid2, kv2 \
    from kernel2;") 
    db.execute("drop table if EXISTS covn3_kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS covn3_kernel2_2(kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT kid2, kv2 \
    from kernel2;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv3_data2_2 group by bin_id) \
    group by cn_id2;") 

    db.execute("drop table if EXISTS conv3_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv3_t_l3 engine = Memory as select im_id2, kid2, data2, kv2 from t_l2, kernel2_2;") 

    # conv4 
    # weight:  torch.Size([768, 1152, 3, 3])
    # input:  torch.Size([64, 1152, 13, 13])
    # input:  torch.Size([64, 768, 13, 13])
    h2 = 13  #output featuremap
    kh = 3
    c_in = 384
    c_out = 384
    # len_feature_map = kh*kh*c_in
    len_feature_map = 256
    len_group=13
    len_all = len_feature_map*len_group
    
    
    x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
    cv2 = x #value of conv2
    for i in range (n-1):
        x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
        cv2 = torch.cat((cv2, x), 0)
    cn_id2=[]
    for i in range(n*h2*h2):
        for j in range(len_all):
            cn_id2.append([i])
    cn_id2 = torch.tensor(cn_id2)
    im_id = []
    for i in range(n):
        for j in range(h2*h2*len_all):
            im_id.append([i])
    im_id = torch.tensor(im_id)
    bin_id = []
    for i in range(n*h2*h2*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)

    data2 = torch.cat((bin_id, cn_id2, im_id, cv2), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data2 (bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8, PRIMARY KEY (bin_id, cn_id, im_id)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data2 values ", data2) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    kid2=[]
    for i in range(c_out):
        kid2.append([i])
    kid2 = torch.tensor(kid2)
    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kernel2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kernel2 (kid2 UInt16, kv2 Array(UInt256), PRIMARY KEY (kid2)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([kid2[i],kv2[i]])
    db.execute("insert into kernel2 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    db.execute("OPTIMIZE TABLE data2 FINAL;") 
    db.execute("drop table if EXISTS conv4_data2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv4_data2_2(bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8) engine = Memory as \
    SELECT bin_id, cn_id, im_id, cv2 \
    from data2;") 

    db.execute("OPTIMIZE TABLE kernel2 FINAL;") 
    db.execute("drop table if EXISTS kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kernel2_2(kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT kid2, kv2 \
    from kernel2;") 
    db.execute("drop table if EXISTS conv4_kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv4_kernel2_2(kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT kid2, kv2 \
    from kernel2;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv4_data2_2 group by bin_id) \
    group by cn_id2;") 

    db.execute("drop table if EXISTS conv4_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv4_t_l3 engine = Memory as select im_id2, kid2, data2, kv2 from t_l2, kernel2_2;") 

    # conv5
    h2 = 6  #output featuremap
    kh = 3
    c_in = 384
    c_out = 256
    pool = 4
    # len_feature_map = kh*kh*c_in
    len_feature_map = 256
    len_group=13
    len_all = len_feature_map*len_group
    
    
    x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
    cv2 = x #value of conv2
    for i in range (n-1):
        x = torch.sign(torch.sign(torch.rand(h2*h2*len_all, 1)-0.5)+1).int()
        cv2 = torch.cat((cv2, x), 0)
    cn_id2=[]
    for i in range(n*h2*h2):
        for j in range(len_all):
            cn_id2.append([i])
    cn_id2 = torch.tensor(cn_id2)
    im_id = []
    for i in range(n):
        for j in range(h2*h2*len_all):
            im_id.append([i])
    im_id = torch.tensor(im_id)
    bin_id = []
    for i in range(n*h2*h2*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)

    data2 = torch.cat((bin_id, cn_id2, im_id, cv2), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data2 (bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8, PRIMARY KEY (bin_id, cn_id, im_id)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data2 values ", data2) 
    print(time.time()-t1)
    print("end create_feature_sql")
    
    
    kid2=[]
    for i in range(c_out):
        kid2.append([i])
    kid2 = torch.tensor(kid2)
    pool2=[]
    for i in range(int(c_out/pool)):
        for j in range(pool):
            pool2.append([i])
    pool2 = torch.tensor(pool2)
    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kernel2;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kernel2 (pool2 UInt8, kid2 UInt16, kv2 Array(UInt256), PRIMARY KEY (pool2, kid2)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([pool2[i],kid2[i],kv2[i]])
    db.execute("insert into kernel2 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")

    db.execute("OPTIMIZE TABLE data2 FINAL;") 
    db.execute("drop table if EXISTS conv5_data2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv5_data2_2(bin_id UInt32, cn_id UInt32, im_id UInt8, cv2 UInt8) engine = Memory as \
    SELECT bin_id, cn_id, im_id, cv2 \
    from data2;") 

    db.execute("OPTIMIZE TABLE kernel2 FINAL;") 
    db.execute("drop table if EXISTS kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kernel2_2(pool2 UInt8, kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT pool2, kid2, kv2 \
    from kernel2;") 
    db.execute("drop table if EXISTS conv5_kernel2_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv5_kernel2_2(pool2 UInt8, kid2 UInt8, kv2 Array(UInt256)) engine = Memory as \
    SELECT pool2, kid2, kv2 \
    from kernel2;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv5_data2_2 group by bin_id) \
    group by cn_id2;") 

    db.execute("drop table if EXISTS conv5_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS conv5_t_l3 engine = Memory as select cn_id2, pool2, im_id2, kid2, data2, kv2 from t_l2, kernel2_2;") 

    # linear 1
    c_in = 256*6*6
    # len_feature_map = c_in
    len_feature_map = 256
    len_group=36
    c_out = 4096

    lv1 = torch.sign(torch.sign(torch.rand(n*c_in, 1)-0.5)+1).int()
    bin_id = []
    for i in range(n*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)
    ln_id1=[]
    for i in range(n):
        for j in range(c_in):
            ln_id1.append([i])
    ln_id1 = torch.tensor(ln_id1)
    data_ln1 = torch.cat((bin_id, ln_id1, lv1), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data_ln1 (bin_id UInt16, ln_id1 UInt8, lv1 UInt8, PRIMARY KEY (bin_id, ln_id1)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data_ln1 values ", data_ln1) 
    print(time.time()-t1)
    print("end create_feature_sql")

    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)
    
    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kerne_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kerne_ln1 (kv1 Array(UInt256), PRIMARY KEY (kv1)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([kv2[i]])
    db.execute("insert into kerne_ln1 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")

    db.execute("OPTIMIZE TABLE data_ln1 FINAL;") 
    db.execute("drop table if EXISTS linear1_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear1_ln1_2(bin_id UInt16, ln_id1 UInt16, lv1 UInt8) engine = Memory as \
    SELECT bin_id, ln_id1, lv1 \
    from data_ln1;") 

    db.execute("OPTIMIZE TABLE kerne_ln1 FINAL;") 
    db.execute("drop table if EXISTS kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 
    db.execute("drop table if EXISTS linear1_kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear1_kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear1_ln1_2 group by bin_id) \
    group by ln_id2;") 

    db.execute("drop table if EXISTS linear1_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear1_t_l3 engine = Memory as select data2, kv1 from t_l2, kerne_ln1_2;") 
    
    # linear 2
    c_in = 4096
    # len_feature_map = c_in
    len_feature_map = 256
    len_group=16
    c_out = 4096

    lv1 = torch.sign(torch.sign(torch.rand(n*c_in, 1)-0.5)+1).int()
    bin_id = []
    for i in range(n*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)
    ln_id1=[]
    for i in range(n):
        for j in range(c_in):
            ln_id1.append([i])
    ln_id1 = torch.tensor(ln_id1)
    data_ln1 = torch.cat((bin_id, ln_id1, lv1), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data_ln1 (bin_id UInt16, ln_id1 UInt8, lv1 UInt8, PRIMARY KEY (bin_id, ln_id1)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data_ln1 values ", data_ln1) 
    print(time.time()-t1)
    print("end create_feature_sql")

    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)
    
    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kerne_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kerne_ln1 (kv1 Array(UInt256), PRIMARY KEY (kv1)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([kv2[i]])
    db.execute("insert into kerne_ln1 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")

    db.execute("OPTIMIZE TABLE data_ln1 FINAL;") 
    db.execute("drop table if EXISTS linear2_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear2_ln1_2(bin_id UInt16, ln_id1 UInt16, lv1 UInt8) engine = Memory as \
    SELECT bin_id, ln_id1, lv1 \
    from data_ln1;") 

    db.execute("OPTIMIZE TABLE kerne_ln1 FINAL;") 
    db.execute("drop table if EXISTS kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 
    db.execute("drop table if EXISTS linear2_kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear2_kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 


    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear2_ln1_2 group by bin_id) \
    group by ln_id2;") 

    db.execute("drop table if EXISTS linear2_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear2_t_l3 engine = Memory as select data2, kv1 from t_l2, kerne_ln1_2;") 

    # linear 3
    c_in = 4096
    # len_feature_map = c_in
    len_feature_map = 256
    len_group=16
    c_out = 10

    lv1 = torch.sign(torch.sign(torch.rand(n*c_in, 1)-0.5)+1).int()
    bin_id = []
    for i in range(n*len_group):
        for j in range(len_feature_map):
            bin_id.append([i])
    bin_id = torch.tensor(bin_id)
    ln_id1=[]
    for i in range(n):
        for j in range(c_in):
            ln_id1.append([i])
    ln_id1 = torch.tensor(ln_id1)
    data_ln1 = torch.cat((bin_id, ln_id1, lv1), 1).tolist()

    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS data_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS data_ln1 (bin_id UInt16, ln_id1 UInt8, lv1 UInt8, PRIMARY KEY (bin_id, ln_id1)) engine = MergeTree();"
    db.execute(create_bit_input)
    db.execute("insert into data_ln1 values ", data_ln1) 
    print(time.time()-t1)
    print("end create_feature_sql")

    kv2=[]
    for i in range(c_out):
        kv2_pre = []
        for j in range(len_group):
            x = random.randint(0,pow(2,len_feature_map))
            kv2_pre.append(x)
        kv2.append(kv2_pre)
    kv2 = np.array(kv2)
    
    t1 = time.time()
    clean_bit_input = "DROP TABLE IF EXISTS kerne_ln1;"
    db.execute(clean_bit_input)  
    create_bit_input="CREATE TABLE IF NOT EXISTS kerne_ln1 (kv1 Array(UInt256), PRIMARY KEY (kv1)) engine = MergeTree();"
    db.execute(create_bit_input)
    kernel2 = []
    for i in range (c_out):
        kernel2.append([kv2[i]])
    db.execute("insert into kerne_ln1 values ", kernel2) 
    print(time.time()-t1)
    print("end create_feature_sql")

    db.execute("OPTIMIZE TABLE data_ln1 FINAL;") 
    db.execute("drop table if EXISTS linear3_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear3_ln1_2(bin_id UInt16, ln_id1 UInt16, lv1 UInt8) engine = Memory as \
    SELECT bin_id, ln_id1, lv1 \
    from data_ln1;") 

    db.execute("OPTIMIZE TABLE kerne_ln1 FINAL;") 
    db.execute("drop table if EXISTS kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 
    db.execute("drop table if EXISTS linear3_kerne_ln1_2;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear3_kerne_ln1_2(kv1 Array(UInt256)) engine = Memory as \
    SELECT kv1 \
    from kerne_ln1;") 

    db.execute("drop table if EXISTS t_l2;") 
    db.execute("CREATE TABLE IF NOT EXISTS t_l2 engine = Memory as \
    select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear3_ln1_2 group by bin_id) \
    group by ln_id2;") 

    db.execute("drop table if EXISTS linear3_t_l3;") 
    db.execute("CREATE TABLE IF NOT EXISTS linear3_t_l3 engine = Memory as select data2, kv1 from t_l2, kerne_ln1_2;") 
    return 

if __name__ == '__main__':
    db = Client(host='localhost')
    reshape()
    print("ok")