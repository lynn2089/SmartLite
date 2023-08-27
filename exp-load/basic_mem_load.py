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

if __name__ == '__main__':
    print("-----------------start-----------------")
    db = Client(host='localhost')
 
    # os.system('sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')

    # q0_2 = "select sum(lv1*kv1)>0?1:0 \
    #     from dl2sql_modeltext_linear1_ln1_2 as a inner join dl2sql_modeltext_linear1_kerne_ln1_2 as b \
    #     on a.bin_id = b.bin_id \
    #     group by ln_id1, kid2;"
    t1=time.time()
    db.execute(q6_1)
    db.execute(q6_2)
    db.execute(q6_3)
    db.execute(q6_4)
    db.execute(q6_5)
    db.execute(q6_6)
    db.execute(q6_7)
    db.execute(q6_8)
    db.execute(q6_9)
    db.execute(q6_10)
    db.execute(q6_11)
    db.execute(q6_12)
    db.execute(q6_13)
    db.execute(q6_14)
    db.execute(q6_15)
    db.execute(q6_16)
    t2=time.time()

    print(t2-t1)
    print("-----------------end-----------------")
    
    