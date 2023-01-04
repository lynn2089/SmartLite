# An example for bitwise processing on neural network on DBMS
# q0 and q0_x are the instructions of model0
# q1 and q1_x are the instructions of model1

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
db = Client(host='localhost')

import resource
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread

q0 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
    from ( \
        select pool1, bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
        from model0_conv1_2 \
    ) \
    group by pool1 \
) \
group by bn11; \
select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv2_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]))>2304?1:0 as val_cv2 \
        from model0_conv2_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22; \
select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv3_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9]))>1152?1:0 as val_cv2 \
    from model0_conv3_t_l3) \
group by im_id2, kid2; \
select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv4_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
    from model0_conv4_t_l3) \
group by im_id2, kid2; \
select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv5_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
        from model0_conv5_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear1_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16])+bitXor(data2[17], kv1[17])+bitXor(data2[18], kv1[18]) \
    +bitXor(data2[19], kv1[19])+bitXor(data2[20], kv1[20])+bitXor(data2[21], kv1[21]) \
    +bitXor(data2[22], kv1[22])+bitXor(data2[23], kv1[23])+bitXor(data2[24], kv1[24]) \
    +bitXor(data2[25], kv1[25])+bitXor(data2[26], kv1[26])+bitXor(data2[27], kv1[27]) \
    +bitXor(data2[28], kv1[28])+bitXor(data2[29], kv1[29])+bitXor(data2[30], kv1[30]) \
    +bitXor(data2[31], kv1[31])+bitXor(data2[32], kv1[32])+bitXor(data2[33], kv1[33]) \
    +bitXor(data2[34], kv1[34])+bitXor(data2[35], kv1[35])+bitXor(data2[36], kv1[36]))>4608?1:0 as val_cv2 from model0_linear1_t_l3; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear2_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model0_linear2_t_l3; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear3_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model0_linear3_t_l3; \
"

q1 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
    from ( \
        select pool1, bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
        from model1_conv1_2 \
    ) \
    group by pool1 \
) \
group by bn11; \
select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv2_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]))>2304?1:0 as val_cv2 \
        from model1_conv2_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22; \
select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv3_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9]))>1152?1:0 as val_cv2 \
    from model1_conv3_t_l3) \
group by im_id2, kid2; \
select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv4_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
    from model1_conv4_t_l3) \
group by im_id2, kid2; \
select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv5_data2_2 group by bin_id \
) \
group by cn_id2; \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
        from model1_conv5_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear1_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16])+bitXor(data2[17], kv1[17])+bitXor(data2[18], kv1[18]) \
    +bitXor(data2[19], kv1[19])+bitXor(data2[20], kv1[20])+bitXor(data2[21], kv1[21]) \
    +bitXor(data2[22], kv1[22])+bitXor(data2[23], kv1[23])+bitXor(data2[24], kv1[24]) \
    +bitXor(data2[25], kv1[25])+bitXor(data2[26], kv1[26])+bitXor(data2[27], kv1[27]) \
    +bitXor(data2[28], kv1[28])+bitXor(data2[29], kv1[29])+bitXor(data2[30], kv1[30]) \
    +bitXor(data2[31], kv1[31])+bitXor(data2[32], kv1[32])+bitXor(data2[33], kv1[33]) \
    +bitXor(data2[34], kv1[34])+bitXor(data2[35], kv1[35])+bitXor(data2[36], kv1[36]))>4608?1:0 as val_cv2 from model1_linear1_t_l3; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear2_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model1_linear2_t_l3; \
select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear3_ln1_2 group by bin_id \
) \
group by ln_id2; \
select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model1_linear3_t_l3; \
"


q0_1 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
        from ( \
            select pool1, bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
            from model0_conv1_2 \
        ) \
        group by pool1 \
    ) \
    group by bn11;"
q0_2 = "select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv2_data2_2 group by bin_id \
    ) \
    group by cn_id2;"
q0_3 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
            +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]))>2304?1:0 as val_cv2 \
            from model0_conv2_t_l3 \
        ) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q0_4 = "select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv3_data2_2 group by bin_id \
    ) \
    group by cn_id2;"
q0_5 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9]))>1152?1:0 as val_cv2 \
        from model0_conv3_t_l3) \
    group by im_id2, kid2;"
q0_6 = "select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv4_data2_2 group by bin_id \
    ) \
    group by cn_id2;"
q0_7 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
    from ( \
        select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
        from model0_conv4_t_l3) \
    group by im_id2, kid2;"
q0_8 = "select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model0_conv5_data2_2 group by bin_id \
    ) \
    group by cn_id2;"
q0_9 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
    from ( \
        select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
        from ( \
            select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
            +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
            +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
            +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
            +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
            from model0_conv5_t_l3 \
        ) \
        group by cn_id2, pool2) \
    group by im_id22, kid22;"
q0_10 = "select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear1_ln1_2 group by bin_id \
    ) \
    group by ln_id2;"
q0_11 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
        +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
        +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
        +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
        +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
        +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16])+bitXor(data2[17], kv1[17])+bitXor(data2[18], kv1[18]) \
        +bitXor(data2[19], kv1[19])+bitXor(data2[20], kv1[20])+bitXor(data2[21], kv1[21]) \
        +bitXor(data2[22], kv1[22])+bitXor(data2[23], kv1[23])+bitXor(data2[24], kv1[24]) \
        +bitXor(data2[25], kv1[25])+bitXor(data2[26], kv1[26])+bitXor(data2[27], kv1[27]) \
        +bitXor(data2[28], kv1[28])+bitXor(data2[29], kv1[29])+bitXor(data2[30], kv1[30]) \
        +bitXor(data2[31], kv1[31])+bitXor(data2[32], kv1[32])+bitXor(data2[33], kv1[33]) \
        +bitXor(data2[34], kv1[34])+bitXor(data2[35], kv1[35])+bitXor(data2[36], kv1[36]))>4608?1:0 as val_cv2 from model0_linear1_t_l3;"
q0_12 = "select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear2_ln1_2 group by bin_id \
    ) \
    group by ln_id2;"
q0_13 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
        +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
        +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
        +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
        +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
        +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model0_linear2_t_l3;"
q0_14 = "select groupArray(data1) as data2 \
    from ( \
        select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model0_linear3_ln1_2 group by bin_id \
    ) \
    group by ln_id2;"
q0_15 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
        +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
        +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
        +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
        +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
        +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model0_linear3_t_l3;"

q1_1 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select max(val)>0?1:0 as val_pool, any(bn1) as bn11 \
    from ( \
        select pool1, bitCount(bitXor(cv1, k1))>181?1:0 as val, bn1 \
        from model1_conv1_2 \
    ) \
    group by pool1 \
) \
group by bn11;"
q1_2 = "select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv2_data2_2 group by bin_id \
) \
group by cn_id2;"
q1_3 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13])+bitXor(data2[14], kv2[14]) \
        +bitXor(data2[15], kv2[15])+bitXor(data2[16], kv2[16])+bitXor(data2[17], kv2[17])+bitXor(data2[18], kv2[18]))>2304?1:0 as val_cv2 \
        from model1_conv2_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22;"
q1_4 = "select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv3_data2_2 group by bin_id \
) \
group by cn_id2;"
q1_5 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9]))>1152?1:0 as val_cv2 \
    from model1_conv3_t_l3) \
group by im_id2, kid2;"
q1_6 = "select any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv4_data2_2 group by bin_id \
) \
group by cn_id2;"
q1_7 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
    +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
    +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
    +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
    +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
    from model1_conv4_t_l3) \
group by im_id2, kid2;"
q1_8 = "select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
from ( \
    select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from model1_conv5_data2_2 group by bin_id \
) \
group by cn_id2;"
q1_9 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, bitCount(bitXor(data2[1], kv2[1])+bitXor(data2[2], kv2[2]) \
        +bitXor(data2[3], kv2[3])+bitXor(data2[4], kv2[4])+bitXor(data2[5], kv2[5]) \
        +bitXor(data2[6], kv2[6])+bitXor(data2[7], kv2[7])+bitXor(data2[8], kv2[8]) \
        +bitXor(data2[9], kv2[9])+bitXor(data2[10], kv2[10])+bitXor(data2[11], kv2[11]) \
        +bitXor(data2[12], kv2[12])+bitXor(data2[13], kv2[13]))>1728?1:0 as val_cv2 \
        from model1_conv5_t_l3 \
    ) \
    group by cn_id2, pool2) \
group by im_id22, kid22;"
q1_10 = "select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear1_ln1_2 group by bin_id \
) \
group by ln_id2;"
q1_11 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16])+bitXor(data2[17], kv1[17])+bitXor(data2[18], kv1[18]) \
    +bitXor(data2[19], kv1[19])+bitXor(data2[20], kv1[20])+bitXor(data2[21], kv1[21]) \
    +bitXor(data2[22], kv1[22])+bitXor(data2[23], kv1[23])+bitXor(data2[24], kv1[24]) \
    +bitXor(data2[25], kv1[25])+bitXor(data2[26], kv1[26])+bitXor(data2[27], kv1[27]) \
    +bitXor(data2[28], kv1[28])+bitXor(data2[29], kv1[29])+bitXor(data2[30], kv1[30]) \
    +bitXor(data2[31], kv1[31])+bitXor(data2[32], kv1[32])+bitXor(data2[33], kv1[33]) \
    +bitXor(data2[34], kv1[34])+bitXor(data2[35], kv1[35])+bitXor(data2[36], kv1[36]))>4608?1:0 as val_cv2 from model1_linear1_t_l3;"
q1_12 = "select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear2_ln1_2 group by bin_id \
) \
group by ln_id2;"
q1_13 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model1_linear2_t_l3;"
q1_14 = "select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from model1_linear3_ln1_2 group by bin_id \
) \
group by ln_id2;"
q1_15 = "select bitCount(bitXor(data2[1], kv1[1])+bitXor(data2[2], kv1[2]) \
    +bitXor(data2[3], kv1[3])+bitXor(data2[4], kv1[4])+bitXor(data2[5], kv1[5]) \
    +bitXor(data2[6], kv1[6])+bitXor(data2[7], kv1[7])+bitXor(data2[8], kv1[8]) \
    +bitXor(data2[9], kv1[9])+bitXor(data2[10], kv1[10])+bitXor(data2[11], kv1[11]) \
    +bitXor(data2[12], kv1[12])+bitXor(data2[13], kv1[13])+bitXor(data2[14], kv1[14]) \
    +bitXor(data2[15], kv1[15])+bitXor(data2[16], kv1[16]))>2048?1:0 as val_cv2 from model1_linear3_t_l3;"

def run_0():
    db = Client(host='localhost')
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
def run_1():
    db = Client(host='localhost')
    db.execute(q1_1)
    db.execute(q1_2)
    db.execute(q1_3)
    db.execute(q1_4)
    db.execute(q1_5)
    db.execute(q1_6)
    db.execute(q1_7)
    db.execute(q1_8)
    db.execute(q1_9)
    db.execute(q1_10)
    db.execute(q1_11)
    db.execute(q1_12)
    db.execute(q1_13)
    db.execute(q1_14)
    db.execute(q1_15)

gc.collect()
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
soft = 4194304000  
hard = soft
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

async def main():
    num_thread = 10
    epoch = 10
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=num_thread)
    for j in range(1,31):
        min_time = 100
        for k in range(epoch):
            threads = []
            for i in range(j):
                threads.append(loop.run_in_executor(pool, run_0)) 
                threads.append(loop.run_in_executor(pool, run_1)) 
            s = time.time()
            done,pending = await asyncio.wait(threads)
            e = time.time()
            tmp_time = e-s
            if(tmp_time<min_time):
                min_time = tmp_time
        print(min_time)

if __name__ == '__main__':
    # main()
    asyncio.run(main())
    
    