from tkinter import S
import argparse
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
import os
import collections


from clickhouse_driver import Client
db = Client(host='localhost')

import resource
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread
from psutil import *
import threading
from ast import literal_eval


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--num_model', metavar='num_model', default=1,
                    help='测试模型数量')

args = parser.parse_args()

# get current memory usage
def get_current_memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

tmp_mem=get_current_memory_usage()
before_used_mem = virtual_memory().used
max_used_mem = 0
kill_t = False
lock = threading.Lock()

def watch_mem():
    global max_used_mem
    while True:
        lock.acquire()
        flag = kill_t
        lock.release()
        if flag == True:
            break
        max_used_mem = max(max_used_mem, virtual_memory().used)
        time.sleep(0.1)
    return

# tasks schedular
def load(str_model,memory_model):
    if str_model in memory_model:
        return local_cost[str_model] # load the sub-model
    else:
        memory_model[str_model]=1
        return global_cost[str_model] # load the pre-trained model and sub-model

def schedular(tasks_R,memory_model):
    d=collections.Counter(tasks_R)
    cost=collections.defaultdict(int)
    tasks=collections.defaultdict(list)
    for task_name, times in enumerate(d):
        tasks[submodel[task_name]].append(task_name)
        load_cost=load(submodel[task_name],memory_model)
        cost[submodel[task_name]]+= load_cost*times # submodel is the dict between pre-trained model and sub-model
    cost=list(cost.items())
    cost.sort(key=lambda x:x[1])
    return cost,tasks
    

def run_alexnet_global(i):
    q0_1= "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
            from ( \
                select pool1, bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
                from model0_conv1_2 \
            ) \
            group by pool1 \
        ) \
        group by bn11;"
    q0_2 = " select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
                +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
                +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
                +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
                +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
                +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]))>2304?1:0 as val_cv2 \
                from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
                from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
                from ( \
                    select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model0_conv2_data2_2 group by bin_id) \
                group by cn_id2) as t_l2, model0_conv2_kernel2_2)) \
            group by cn_id2, pool2) \
        group by im_id22, kid22;"
    q0_3 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9]))>1152?1:0 as val_cv2 \
            from (select im_id2, kid2, data2, kv2 \
            from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model0_conv3_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model0_conv3_kernel2_2)) \
        group by im_id2, kid2;"

    q0_4 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
            from (select im_id2, kid2, data2, kv2 \
            from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model0_conv4_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model0_conv4_kernel2_2)) \
        group by im_id2, kid2;"
    q0_5 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
                +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
                +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
                +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
                +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
                from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
                from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
                from ( \
                    select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model0_conv5_data2_2 group by bin_id) \
                group by cn_id2) as t_l2, model0_conv5_kernel2_2)) \
            group by cn_id2, pool2) \
        group by im_id22, kid22;"
    q0_6 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
        +bitXor(data2[19], kv2[19])+bitXor(data2[20], kv2[20])+bitXor(data2[21], kv2[21]) \
        +bitXor(data2[22], kv2[22])+bitXor(data2[23], kv2[23])+bitXor(data2[24], kv2[24]) \
        +bitXor(data2[25], kv2[25])+bitXor(data2[26], kv2[26])+bitXor(data2[27], kv2[27]) \
        +bitXor(data2[28], kv2[28])+bitXor(data2[29], kv2[29])+bitXor(data2[30], kv2[30]) \
        +bitXor(data2[31], kv2[31])+bitXor(data2[32], kv2[32])+bitXor(data2[33], kv2[33]) \
        +bitXor(data2[34], kv2[34])+bitXor(data2[35], kv2[35])+bitXor(data2[36], kv2[36]))>4608?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear1_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear1_kerne_ln1_2);"
    q0_7 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear2_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear2_kerne_ln1_2);"
    q0_8 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear3_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear3_kerne_ln1_2);"

    db.execute(q0_1)
    db.execute(q0_2)
    db.execute(q0_3)
    db.execute(q0_4)
    db.execute(q0_5)
    db.execute(q0_6)
    db.execute(q0_7)
    db.execute(q0_8)

def run_alexnet_local(i):
    q0_6 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
        +bitXor(data2[19], kv2[19])+bitXor(data2[20], kv2[20])+bitXor(data2[21], kv2[21]) \
        +bitXor(data2[22], kv2[22])+bitXor(data2[23], kv2[23])+bitXor(data2[24], kv2[24]) \
        +bitXor(data2[25], kv2[25])+bitXor(data2[26], kv2[26])+bitXor(data2[27], kv2[27]) \
        +bitXor(data2[28], kv2[28])+bitXor(data2[29], kv2[29])+bitXor(data2[30], kv2[30]) \
        +bitXor(data2[31], kv2[31])+bitXor(data2[32], kv2[32])+bitXor(data2[33], kv2[33]) \
        +bitXor(data2[34], kv2[34])+bitXor(data2[35], kv2[35])+bitXor(data2[36], kv2[36]))>4608?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear1_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear1_kerne_ln1_2);"
    q0_7 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear2_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear2_kerne_ln1_2);"
    q0_8 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear3_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear3_kerne_ln1_2);"

    db.execute(q0_6)
    db.execute(q0_7)
    db.execute(q0_8)

str_linear = "select bitCount("
for i in range (1,129):
    str_linear = str_linear + "bitXor(data2["+str(i)+"], kv2["+str(i)+"])"
    if i!=128:
        str_linear = str_linear + "+"
str_linear = str_linear + ")>16384?1:0 as val_cv2 "

def run_vgg_global(i):
    q0_1 = "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
        from ( \
            select bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
            from model51_conv1_2 \
        ) \
        group by bn1;"
    q0_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            )>288?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv2_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv2_kernel2_2)) \
        group by im_id2, kid2;"
    q0_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            )>576?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv3_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv3_kernel2_2)) \
        group by im_id2, kid2;"
    q0_4 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
                +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
                )>576?1:0 as val_cv2 \
                from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
                from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
                from ( \
                    select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv4_data2_2 group by bin_id) \
                group by cn_id2) as t_l2, model51_conv4_kernel2_2)) \
            group by cn_id2, pool2) \
        group by im_id22, kid22;"
    q0_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9]) \
            )>1152?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv5_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv5_kernel2_2)) \
        group by im_id2, kid2;"
    q0_6 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
                +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
                +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
                +bitXor(data2[9], kv2[9]) \
                )>1152?1:0 as val_cv2 \
                from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
                from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
                from ( \
                    select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv6_data2_2 group by bin_id) \
                group by cn_id2) as t_l2, model51_conv6_kernel2_2)) \
            group by cn_id2, pool2) \
        group by im_id22, kid22;"
    q0_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9]) \
            )>1152?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv7_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv7_kernel2_2)) \
        group by im_id2, kid2;"
    q0_8 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv8_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv8_kernel2_2)) \
        group by im_id2, kid2;"
    q0_9 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv9_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv9_kernel2_2)) \
        group by im_id2, kid2;"
    q0_10 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv10_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv10_kernel2_2)) \
        group by im_id2, kid2;"
    q0_11 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv11_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv11_kernel2_2)) \
        group by im_id2, kid2;"
    q0_12 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv12_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv12_kernel2_2)) \
        group by im_id2, kid2;"
    q0_13 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
            )>2304?1:0 as val_cv2 \
            from (select cn_id2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv13_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, model51_conv13_kernel2_2)) \
        group by im_id2, kid2;"
    
    q0_14 = str_linear + "from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear1_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear1_kerne_ln1_2);"
    q0_15 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear2_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear2_kerne_ln1_2);"
    q0_16 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear3_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear3_kerne_ln1_2);"

    db.execute(q0_1)
    db.execute(q0_2)
    db.execute(q0_3)
    db.execute(q0_4)
    db.execute(q0_5)
    db.execute(q0_6)
    db.execute(q0_7)
    db.execute(q0_8)
    db.execute(q0_9)
    db.execute(q0_10)
    db.execute(q0_11)
    db.execute(q0_12)
    db.execute(q0_13)
    db.execute(q0_14)
    db.execute(q0_15)
    db.execute(q0_16)
   
def run_vgg_local(i):
    q0_14 = str_linear + "from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear1_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear1_kerne_ln1_2);"
    q0_15 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear2_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear2_kerne_ln1_2);"
    q0_16 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
        from (select data2, kv1 as kv2 \
        from (select groupArray(data1) as data2 \
        from ( \
            select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model"+str(i)+"_linear3_ln1_2 group by bin_id) \
        group by ln_id2) as t_l2, model"+str(i)+"_linear3_kerne_ln1_2);"

    db.execute(q0_14)
    db.execute(q0_15)
    db.execute(q0_16)
   
gc.collect()
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (576460752303423488*2*0.2, hard))
soft = 5368709120 
hard = soft
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

db.execute("set max_memory_usage = 5368709120")
# os.system('sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')

t = threading.Thread(target=watch_mem)
t.start()

async def main():
    
    num_model = int(args.num_model)
    num_thread = 20
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=num_thread)
    epoch = 1
    sum_time = 0

    # test different epoch
    for k in range(epoch):
        threads = []
        # for i in range(0,num_model):
        #     threads.append(loop.run_in_executor(pool, run_alexnet, i)) 
        #     threads.append(loop.run_in_executor(pool, run_vgg,i+51)) 

        # multi-model loading
        for cost, tasks in enumerate(schedular(tasks_R,memory_model)):
            for i in tasks[cost[0]]:
                if cost[0]=="alexnet":
                    if cost[0] in memory_model:
                        threads.append(loop.run_in_executor(pool, run_alexnet_local, i)) 
                    else:
                        memory_model[cost[0]]=1
                        threads.append(loop.run_in_executor(pool, run_alexnet_global, i)) 

                if cost[0]=="vgg":
                    if cost[0] in memory_model:
                        threads.append(loop.run_in_executor(pool, run_vgg_local, i)) 
                    else:
                        memory_model[cost[0]]=1
                        threads.append(loop.run_in_executor(pool, run_vgg_global, i)) 

        s = time.time()
        done,pending = await asyncio.wait(threads)
        e = time.time()
        tmp_time = e-s
        # if(tmp_time<min_time):
        #     min_time = tmp_time
        sum_time = sum_time + tmp_time
    global kill_t
    lock.acquire()
    kill_t = True
    lock.release()
    print("Model Number: {}, Memory Usage: {} MB, Time: {}".format(num_model*2, (max_used_mem - before_used_mem) / 1024 / 1024, sum_time/epoch))

if __name__ == '__main__':
    print("-----------------start-----------------")
    futures = [main()]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.wait(futures))
    print(time.time())
    print("-----------------end-----------------")
    
    