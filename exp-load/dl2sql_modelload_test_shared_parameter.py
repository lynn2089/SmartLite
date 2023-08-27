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

from clickhouse_driver import Client
db = Client(host='localhost')

import resource
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread

parser = argparse.ArgumentParser(description='load')

parser.add_argument('--num_model', metavar='num_model', default=1,
                    help='the number of models')
args = parser.parse_args()

def run_alexnet(i):
    db = Client(host='localhost')
    q0_1= "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
        from ( \
            select sum(cv1 * k1)>0?1:0 as val, bn1 \
            from dl2sql_model0_conv1_2 \
            group by bn1 \
        ) \
        group by bn1;"
    q0_2 = " select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
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
    q0_3 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model0_conv3_data2_2 as a inner join dl2sql_model0_conv3_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"

    q0_4 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model0_conv4_data2_2 as a inner join dl2sql_model0_conv4_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_5 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
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
    
    q0_6 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear1_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear1_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"
    q0_7 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear2_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear2_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"
    q0_8 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear3_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear3_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"

    db.execute(q0_1)
    db.execute(q0_2)
    db.execute(q0_3)
    db.execute(q0_4)
    db.execute(q0_5)
    db.execute(q0_6)
    db.execute(q0_7)
    db.execute(q0_8)


def run_vgg(i):
    db = Client(host='localhost')
    q0_1 = "select (((avg(val))/stddevSamp(val))+0.00005)>0?1:0 \
        from ( \
            select sum(cv1 * k1)>0?1:0 as val, bn1 \
            from dl2sql_model51_conv1_2 \
            group by bn1 \
        ) \
        group by bn1;"
    q0_2 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv2_data2_2 as a inner join dl2sql_model51_conv2_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_3 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv3_data2_2 as a inner join dl2sql_model51_conv3_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_4 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
                from dl2sql_model51_conv4_data2_2 as a inner join dl2sql_model51_conv4_kernel2_2 as b \
                on a.bin_id = b.orderid2 \
                group by cn_id, kid2 \
            ) \
            group by cn_id, pool22) \
        group by im_id22, kid22;"
    q0_5 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv5_data2_2 as a inner join dl2sql_model51_conv5_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_6 = "select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
        from ( \
            select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
            from ( \
                select any(im_id) as im_id2, any(pool2) as pool22, cn_id, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
                from dl2sql_model51_conv6_data2_2 as a inner join dl2sql_model51_conv6_kernel2_2 as b \
                on a.bin_id = b.orderid2 \
                group by cn_id, kid2 \
            ) \
            group by cn_id, pool22) \
        group by im_id22, kid22;"
    q0_7 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv7_data2_2 as a inner join dl2sql_model51_conv7_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_8 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv8_data2_2 as a inner join dl2sql_model51_conv8_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_9 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv9_data2_2 as a inner join dl2sql_model51_conv9_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_10 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv10_data2_2 as a inner join dl2sql_model51_conv10_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_11 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv11_data2_2 as a inner join dl2sql_model51_conv11_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_12 = " select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv12_data2_2 as a inner join dl2sql_model51_conv12_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    q0_13 = "select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
        from ( \
            select any(im_id) as im_id2, kid2, sum(cv2*kv2)>0?1:0 as val_cv2 \
            from dl2sql_model51_conv13_data2_2 as a inner join dl2sql_model51_conv13_kernel2_2 as b \
            on a.bin_id = b.orderid2 \
            group by cn_id, kid2 \
        ) \
        group by im_id2, kid2;"
    
    q0_14 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear1_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear1_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"
    
    q0_15 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear2_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear2_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"
    
    q0_16 = "select sum(lv1*kv1)>0?1:0 \
        from dl2sql_model"+str(i)+"_linear3_ln1_2 as a inner join dl2sql_model"+str(i)+"_linear3_kerne_ln1_2 as b \
        on a.bin_id = b.bin_id \
        group by ln_id1, kid2;"

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
   
gc.collect()
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (576460752303423488*2*0.2, hard))
soft = 5368709120  #5gb
hard = soft
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

db.execute("set max_memory_usage = 21474836480")
# os.system('sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')
# os.system('echo panda9105 | sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')

# get_current_memory_usage
def get_current_memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

async def main():
    num_model = int(args.num_model)
    num_thread = 20
    loop = asyncio._get_running_loop()
    pool = ThreadPoolExecutor(max_workers=num_thread)
    epoch = 1
    sum_time = 0
    sum_mem = 0
    for k in range(epoch):
        threads = []
        for i in range(num_model):
            threads.append(loop.run_in_executor(pool, run_alexnet, i)) 
            threads.append(loop.run_in_executor(pool, run_vgg,i+51)) 
        s = time.time()
        done,pending = await asyncio.wait(threads)
        e = time.time()
        tmp_time = e-s
        # tmp_mem = get_current_memory_usage()
        # if(tmp_time<min_time):
        #     min_time = tmp_time
        sum_time = sum_time + tmp_time
        # sum_mem = sum_mem + tmp_mem
    print("Model Number: {}, Memory Usage: {} MB, Time: {}".format(num_model*2, get_current_memory_usage(), sum_time/epoch))

if __name__ == '__main__':
    print("-----------------start-----------------")
    futures = [main()]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.wait(futures))
    print(time.time())
    print("-----------------end-----------------")
    
    