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
import os

from clickhouse_driver import Client
db = Client(host='localhost')

import resource
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread

import threading
from psutil import virtual_memory

max_used_mem = 0
kill_t = False
def watch_mem():
    global max_used_mem
    while True:
        lock.acquire()
        flag = kill_t
        lock.release()
        if flag == True:
            break
        max_used_mem = max(max_used_mem, virtual_memory().used)
        time.sleep(0.01)
    return

# smartlite lookup alexnet
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
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model0_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model0_linear1_kerne_ln1_2);"
q0_7 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
    +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model0_linear2_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model0_linear2_kerne_ln1_2);"
q0_8 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
    +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model0_linear3_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model0_linear3_kerne_ln1_2);"

str_linear_vgg_db = "select bitCount("
for i in range (1,129):
    str_linear_vgg_db = str_linear_vgg_db + "bitXor(data2["+str(i)+"], kv2["+str(i)+"])"
    if i!=128:
        str_linear_vgg_db = str_linear_vgg_db + "+"
str_linear_vgg_db = str_linear_vgg_db + ")>16384?1:0 as val_cv2 "

# smartlite lookup vgg
q1_1 = "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
    from ( \
        select bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
        from model51_conv1_2 \
    ) \
    group by bn1;"
q1_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>288?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from model51_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, model51_conv2_kernel2_2)) \
    group by im_id2, kid2;"
q1_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_4 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
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
q1_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_6 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
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
q1_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_8 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_9 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_10 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_11 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_12 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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
q1_13 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
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

q1_14 = str_linear_vgg_db + "from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model51_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model51_linear1_kerne_ln1_2);"
q1_15 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
    +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model51_linear2_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model51_linear2_kerne_ln1_2);"
q1_16 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
    +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16]))>2048?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from model51_linear3_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, model51_linear3_kerne_ln1_2);"

str_linear_resnet_db = "select bitCount("
for i in range (1,20):
    str_linear_resnet_db = str_linear_resnet_db + "bitXor(data2["+str(i)+"], kv2["+str(i)+"])"
    if i!=19:
        str_linear_resnet_db = str_linear_resnet_db + "+"
str_linear_resnet_db = str_linear_resnet_db + ")>2560?1:0 as val_cv2 "

#smartlite lookup resnet
q2_1 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>288?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv2_kernel2_2)) \
    group by im_id2, kid2;"

q2_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>288?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv2_kernel2_2)) \
    group by im_id2, kid2;"

q2_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>288?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv3_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv3_kernel2_2)) \
    group by im_id2, kid2;"
q2_4 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>288?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv4_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv4_kernel2_2)) \
    group by im_id2, kid2;"
q2_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        )>576?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv5_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv5_kernel2_2)) \
    group by im_id2, kid2;"
q2_6 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        )>576?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv6_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv6_kernel2_2)) \
    group by im_id2, kid2;"
q2_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        )>1408?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_samrtlite_conv7_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_samrtlite_conv7_kernel2_2)) \
    group by im_id2, kid2;"

q2_14 = str_linear_resnet_db + "from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelresnet_samrtlite_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelresnet_samrtlite_linear1_kerne_ln1_2);"
   
# dl2sql alexnet
q3_1= "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
    from ( \
        select sum(cv1 * k1)>0?1:0 as val, bn1 \
        from dl2sql_model0_conv1_2 \
        group by bn1 \
    ) \
    group by bn1;"
q3_2 = " select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model0_conv2_data2_2 as a inner join dl2sql_model0_conv2_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by cn_id, pool22) \
    group by im_id22, kid22;"
q3_3 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model0_conv3_data2_2 as a inner join dl2sql_model0_conv3_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"

q3_4 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model0_conv4_data2_2 as a inner join dl2sql_model0_conv4_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q3_5 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model0_conv5_data2_2 as a inner join dl2sql_model0_conv5_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by cn_id, pool22 \
    ) \
    group by im_id22, kid22;"

q3_6 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model0_linear1_ln1_2 as a inner join dl2sql_model0_linear1_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"
q3_7 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model0_linear2_ln1_2 as a inner join dl2sql_model0_linear2_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"
q3_8 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model0_linear3_ln1_2 as a inner join dl2sql_model0_linear3_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"

# dl2sql vgg
q4_1 = "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
    from ( \
        select sum(cv1 * k1)>0?1:0 as val, bn1 \
        from dl2sql_model251_conv1_2 \
        group by bn1 \
    ) \
    group by bn1;"
q4_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv2_data2_2 as a inner join dl2sql_model251_conv2_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv3_data2_2 as a inner join dl2sql_model251_conv3_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_4 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model251_conv4_data2_2 as a inner join dl2sql_model251_conv4_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by cn_id, pool22) \
    group by im_id22, kid22;"
q4_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv5_data2_2 as a inner join dl2sql_model251_conv5_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_6 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model251_conv6_data2_2 as a inner join dl2sql_model251_conv6_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by cn_id, pool22) \
    group by im_id22, kid22;"
q4_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv7_data2_2 as a inner join dl2sql_model251_conv7_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_8 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv8_data2_2 as a inner join dl2sql_model251_conv8_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_9 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv9_data2_2 as a inner join dl2sql_model251_conv9_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_10 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv10_data2_2 as a inner join dl2sql_model251_conv10_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_11 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv11_data2_2 as a inner join dl2sql_model251_conv11_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_12 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv12_data2_2 as a inner join dl2sql_model251_conv12_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q4_13 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_model251_conv13_data2_2 as a inner join dl2sql_model251_conv13_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"

q4_14 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model251_linear1_ln1_2 as a inner join dl2sql_model251_linear1_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"

q4_15 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model251_linear2_ln1_2 as a inner join dl2sql_model251_linear2_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"

q4_16 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_model251_linear3_ln1_2 as a inner join dl2sql_model251_linear3_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"

#dl2sql resnet
q5_1 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv2_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv2_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv2_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv2_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"

q5_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv3_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv3_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_4 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv4_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv4_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv5_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv5_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_6 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv6_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv6_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
        from dl2sql_modelresnet_dl2sql_conv7_data2_2 as a inner join dl2sql_modelresnet_dl2sql_conv7_kernel2_2 as b \
        on a.bin_id = b.orderid2 \
        group by cn_id, kid2 \
    ) \
    group by im_id2, kid2;"
q5_14 = "select sum(lv1*kv1)>0?1:0 \
    from dl2sql_modelresnet_dl2sql_linear2_ln1_2 as a inner join dl2sql_modelresnet_dl2sql_linear2_kerne_ln1_2 as b \
    on a.bin_id = b.bin_id \
    group by ln_id1, kid2;"

str_linear_vgg_prune = "select bitCount("
for i in range (1,99):
    str_linear_vgg_prune = str_linear_vgg_prune + "bitXor(data2["+str(i)+"], kv2["+str(i)+"])"
    if i!=98:
        str_linear_vgg_prune = str_linear_vgg_prune + "+"
str_linear_vgg_prune = str_linear_vgg_prune + ")>12713?1:0 as val_cv2 "

# pruned vgg
q6_1 = "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
    from ( \
        select bitCount(bitXor(cv1, k1))>140?1:0 as val, bn1 \
        from modelvgg16_prune_conv1_2 \
    ) \
    group by bn1;"
q6_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1]) \
        )>223?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv2_kernel2_2)) \
    group by im_id2, kid2;"
q6_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3]) \
        )>446?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv3_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv3_kernel2_2)) \
    group by im_id2, kid2;"
q6_4 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3]) \
            )>446?1:0 as val_cv2 \
            from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv4_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, modelvgg16_prune_conv4_kernel2_2)) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q6_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7]) \
        )>893?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv5_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv5_kernel2_2)) \
    group by im_id2, kid2;"
q6_6 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7]) \
            )>893?1:0 as val_cv2 \
            from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv6_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, modelvgg16_prune_conv6_kernel2_2)) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q6_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7]) \
        )>893?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv7_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv7_kernel2_2)) \
    group by im_id2, kid2;"
q6_8 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv8_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv8_kernel2_2)) \
    group by im_id2, kid2;"
q6_9 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv9_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv9_kernel2_2)) \
    group by im_id2, kid2;"
q6_10 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv10_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv10_kernel2_2)) \
    group by im_id2, kid2;"
q6_11 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv11_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv11_kernel2_2)) \
    group by im_id2, kid2;"
q6_12 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv12_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv12_kernel2_2)) \
    group by im_id2, kid2;"
q6_13 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        )>1787?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelvgg16_prune_conv13_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelvgg16_prune_conv13_kernel2_2)) \
    group by im_id2, kid2;"

q6_14 = str_linear_vgg_prune + "from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelvgg16_prune_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelvgg16_prune_linear1_kerne_ln1_2);"
q6_15 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12]))>1589?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelvgg16_prune_linear2_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelvgg16_prune_linear2_kerne_ln1_2);"
q6_16 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12]))>1589?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelvgg16_prune_linear3_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelvgg16_prune_linear3_kerne_ln1_2);"

# pruned alexnet
q7_1= "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
        from ( \
            select pool1, bitCount(bitXor(cv1, k1))>129?1:0 as val, bn1 \
            from modelalexnet_prune_conv1_2 \
        ) \
        group by pool1 \
    ) \
    group by bn11;"
q7_2 = " select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1645?1:0 as val_cv2 \
            from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelalexnet_prune_conv2_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, modelalexnet_prune_conv2_kernel2_2)) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q7_3 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6]))>822?1:0 as val_cv2 \
        from (select im_id2, kid2, data2, kv2 \
        from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelalexnet_prune_conv3_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelalexnet_prune_conv3_kernel2_2)) \
    group by im_id2, kid2;"

q7_4 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9]))>1233?1:0 as val_cv2 \
        from (select im_id2, kid2, data2, kv2 \
        from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelalexnet_prune_conv4_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelalexnet_prune_conv4_kernel2_2)) \
    group by im_id2, kid2;"
q7_5 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9]))>1233?1:0 as val_cv2 \
            from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
            from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
            from ( \
                select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelalexnet_prune_conv5_data2_2 group by bin_id) \
            group by cn_id2) as t_l2, modelalexnet_prune_conv5_kernel2_2)) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q7_6 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
    +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]) \
    +bitXor(data2[19], kv2[19])+bitXor(data2[20], kv2[20])+bitXor(data2[21], kv2[21]) \
    +bitXor(data2[22], kv2[22])+bitXor(data2[23], kv2[23])+bitXor(data2[24], kv2[24]) \
    +bitXor(data2[25], kv2[25]))>3290?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelalexnet_prune_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelalexnet_prune_linear1_kerne_ln1_2);"
q7_7 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    )>1462?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelalexnet_prune_linear2_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelalexnet_prune_linear2_kerne_ln1_2);"
q7_8 = "select bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    )>1462?1:0 as val_cv2 \
    from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelalexnet_prune_linear3_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelalexnet_prune_linear3_kerne_ln1_2);"

str_linear_resnet_prune = "select bitCount("
for i in range (1,15):
    str_linear_resnet_prune = str_linear_resnet_prune + "bitXor(data2["+str(i)+"], kv2["+str(i)+"])"
    if i!=14:
        str_linear_resnet_prune = str_linear_resnet_prune + "+"
str_linear_resnet_prune = str_linear_resnet_prune + ")>1955?1:0 as val_cv2 "

# pruned resnet
q8_1 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>220?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv2_kernel2_2)) \
    group by im_id2, kid2;"

q8_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>220?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv2_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv2_kernel2_2)) \
    group by im_id2, kid2;"

q8_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>220?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv3_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv3_kernel2_2)) \
    group by im_id2, kid2;"
q8_4 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        )>220?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv4_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv4_kernel2_2)) \
    group by im_id2, kid2;"
q8_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4]) \
        )>440?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv5_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv5_kernel2_2)) \
    group by im_id2, kid2;"
q8_6 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4]) \
        )>440?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv6_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv6_kernel2_2)) \
    group by im_id2, kid2;"
q8_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select cn_id2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        )>1075?1:0 as val_cv2 \
        from (select cn_id2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, groupBitAnd(cv2) as data1 from modelresnet_prune_conv7_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, modelresnet_prune_conv7_kernel2_2)) \
    group by im_id2, kid2;"

q8_14 = str_linear_resnet_prune + "from (select data2, kv1 as kv2 \
    from (select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, groupBitAnd(lv1) as data1 from modelresnet_prune_linear1_ln1_2 group by bin_id) \
    group by ln_id2) as t_l2, modelresnet_prune_linear1_kerne_ln1_2);"
 
# test memory usage
def get_current_memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


if __name__ == '__main__':
    print("-----------------start-----------------")
    # 
    # db_run_vgg / db_run_alexnet / db_run_resnet
    # 51/0/resnet_samrtlite

    # dl2sql_run_vgg / dl2sql_run_alexnet / dl2sql_run_resnet
    # 251/0/resnet_dl2sql

    # prune_run_vgg / prune_run_alexnet / prune_run_resnet
    # vgg16_prune /alexnet_prune/resnet_prune
    name = 'resnet_prune'
    db = Client(host='localhost')

    before_used_mem = virtual_memory().used
    lock = threading.Lock()
    t = threading.Thread(target=watch_mem)
    t.start()

    t1 = time.time()
    # smartlite lookup alexnet 
    db.execute(q0_1)
    db.execute(q0_2)
    db.execute(q0_3)
    db.execute(q0_4)
    db.execute(q0_5)
    db.execute(q0_6)
    db.execute(q0_7)
    db.execute(q0_8)

    # smartlite lookup vgg 
    # db.execute(q1_1)
    # db.execute(q1_2)
    # db.execute(q1_3)
    # db.execute(q1_4)
    # db.execute(q1_5)
    # db.execute(q1_6)
    # db.execute(q1_7)
    # db.execute(q1_8)
    # db.execute(q1_9)
    # db.execute(q1_10)
    # db.execute(q1_11)
    # db.execute(q1_12)
    # db.execute(q1_13)
    # db.execute(q1_14)
    # db.execute(q1_15)
    # db.execute(q1_16)

    # smartlite lookup resnet 
    # db.execute(q2_1)
    # db.execute(q2_2)
    # db.execute(q2_3)
    # db.execute(q2_4)
    # db.execute(q2_5)
    # db.execute(q2_6)
    # db.execute(q2_7)
    # db.execute(q2_14)


    # dl2sql alexnet 
    # db.execute(q3_1)
    # db.execute(q3_2)
    # db.execute(q3_3)
    # db.execute(q3_4)
    # db.execute(q3_5)
    # db.execute(q3_6)
    # db.execute(q3_7)
    # db.execute(q3_8)

    # dl2sql vgg
    # db.execute(q4_1)
    # db.execute(q4_2)
    # db.execute(q4_3)
    # db.execute(q4_4)
    # db.execute(q4_5)
    # db.execute(q4_6)
    # db.execute(q4_7)
    # db.execute(q4_8)
    # db.execute(q4_9)
    # db.execute(q4_10)
    # db.execute(q4_11)
    # db.execute(q4_12)
    # db.execute(q4_13)
    # db.execute(q4_14)
    # db.execute(q4_15)
    # db.execute(q4_16) 

    # dl2sql resnet 
    # db.execute(q5_1)
    # db.execute(q5_2)
    # db.execute(q5_3)
    # db.execute(q5_4)
    # db.execute(q5_5)
    # db.execute(q5_6)
    # db.execute(q5_7)
    # db.execute(q5_14)



    # smartlite pruned alexnet 
    # db.execute(q7_1)
    # db.execute(q7_2)
    # db.execute(q7_3)
    # db.execute(q7_4)
    # db.execute(q7_5)
    # db.execute(q7_6)
    # db.execute(q7_7)
    # db.execute(q7_8)
    

    # smartlite pruned vgg 
    # db.execute(q6_1)
    # db.execute(q6_2)
    # db.execute(q6_3)
    # db.execute(q6_4)
    # db.execute(q6_5)
    # db.execute(q6_6)
    # db.execute(q6_7)
    # db.execute(q6_8)
    # db.execute(q6_9)
    # db.execute(q6_10)
    # db.execute(q6_11)
    # db.execute(q6_12)
    # db.execute(q6_13)
    # db.execute(q6_14)
    # db.execute(q6_15)
    # db.execute(q6_16)

    # smartlite pruned resnet 
    # db.execute(q8_1)
    # db.execute(q8_2)
    # db.execute(q8_3)
    # db.execute(q8_4)
    # db.execute(q8_5)
    # db.execute(q8_6)
    # db.execute(q8_7)
    # db.execute(q8_14)


    t2 = time.time()

    lock.acquire()
    kill_t = True
    lock.release()
    t.join()

    print("Model name: {}, Memory Usage: {} MB, Time: {}".format(name,  (max_used_mem - before_used_mem) / 1024 / 1024, t2-t1))
    print("-----------------end-----------------")
    
    