#!/usr/bin/python3

import sys
import json
from clickhouse_driver import Client
db = Client(host='localhost')
if __name__ == '__main__':
    for line in sys.stdin:
        value = json.loads(line)
        first_arg = int(value['argument_1'])
        second_arg = value['argument_2']
        db.execute(q0)
        db.execute(q1)
        db.execute(q2)
        db.execute(q3)
        db.execute(q4)
        db.execute(q5)
        result = {db.execute(q6)}
        print(json.dumps(result), end='\n')
        sys.stdout.flush()



q0 = " CREATE TABLE IF NOT EXISTS conv3_data2_2 engine = Memory as \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
        +lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
        +lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
        +lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
        +lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])+lookup(data2[14], kv2[14]) \
        +lookup(data2[15], kv2[15])+lookup(data2[16], kv2[16])+lookup(data2[17], kv2[17])+lookup(data2[18], kv2[18])>2304?1:0 as val_cv2 \
        from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(",first_arg,") as data1 from ",second_arg," group by bin_id) \
        group by cn_id2) as t_l2, conv2_kernel2_2)) \
    group by cn_id2, pool2) \
group by im_id22, kid22;" 


q1 = " CREATE TABLE IF NOT EXISTS conv4_data2_2 engine = Memory as \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
    +lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
    +lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
    +lookup(data2[9], kv2[9])>1152?1:0 as val_cv2 \
    from (select im_id2, kid2, data2, kv2 \
    from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv3_data2_2 group by bin_id) \
    group by cn_id2) as t_l2, covn3_kernel2_2)) \
group by im_id2, kid2;" 


q2 = "CREATE TABLE IF NOT EXISTS conv5_data2_2 engine = Memory as \
select (((avg(val_cv2))/stddevSamp(val_cv2))+0.00005)>0?1:0 \
from ( \
    select im_id2, kid2, lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
    +lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
    +lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
    +lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
    +lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])>1728?1:0 as val_cv2 \
    from (select im_id2, kid2, data2, kv2 \
    from (select any(im_id1) as im_id2, groupArray(data1) as data2 \
    from ( \
        select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv4_data2_2 group by bin_id) \
    group by cn_id2) as t_l2, conv4_kernel2_2)) \
group by im_id2, kid2;" 

q3 = "CREATE TABLE IF NOT EXISTS linear1_ln1_2 engine = Memory as \
select (((avg(val_pool))/stddevSamp(val_pool))+0.00005)>0?1:0 \
from ( \
    select any(im_id2) as im_id22, any(kid2) as kid22, max(val_cv2)>0?1:0 as val_pool \
    from ( \
        select cn_id2, pool2, im_id2, kid2, lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
        +lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
        +lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
        +lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
        +lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])>1728?1:0 as val_cv2 \
        from (select cn_id2, pool2, im_id2, kid2, data2, kv2 \
        from (select cn_id2, any(im_id1) as im_id2, groupArray(data1) as data2 \
        from ( \
            select any(cn_id) as cn_id2, any(im_id) as im_id1, sumbin(cv2) as data1 from conv5_data2_2 group by bin_id) \
        group by cn_id2) as t_l2, conv5_kernel2_2)) \
    group by cn_id2, pool2) \
group by im_id22, kid22;"

q4 = "CREATE TABLE IF NOT EXISTS linear2_ln1_2 engine = Memory as \
select lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
+lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
+lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
+lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
+lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])+lookup(data2[14], kv2[14]) \
+lookup(data2[15], kv2[15])+lookup(data2[16], kv2[16])+lookup(data2[17], kv2[17])+lookup(data2[18], kv2[18]) \
+lookup(data2[19], kv2[19])+lookup(data2[20], kv2[20])+lookup(data2[21], kv2[21]) \
+lookup(data2[22], kv2[22])+lookup(data2[23], kv2[23])+lookup(data2[24], kv2[24]) \
+lookup(data2[25], kv2[25])+lookup(data2[26], kv2[26])+lookup(data2[27], kv2[27]) \
+lookup(data2[28], kv2[28])+lookup(data2[29], kv2[29])+lookup(data2[30], kv2[30]) \
+lookup(data2[31], kv2[31])+lookup(data2[32], kv2[32])+lookup(data2[33], kv2[33]) \
+lookup(data2[34], kv2[34])+lookup(data2[35], kv2[35])+lookup(data2[36], kv2[36])>4608?1:0 as val_cv2 \
from (select data2, kv1 as kv2 \
from (select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear1_ln1_2 group by bin_id) \
group by ln_id2) as t_l2, linear1_kerne_ln1_2);"

q5 = "CREATE TABLE IF NOT EXISTS linear3_ln1_2 engine = Memory as \
select lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
+lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
+lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
+lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
+lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])+lookup(data2[14], kv2[14]) \
+lookup(data2[15], kv2[15])+lookup(data2[16], kv2[16])>2048?1:0 as val_cv2 \
from (select data2, kv1 as kv2 \
from (select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear2_ln1_2 group by bin_id) \
group by ln_id2) as t_l2, linear2_kerne_ln1_2);" 

q6 =  "select lookup(data2[1], kv2[1])+lookup(data2[2], kv2[2]) \
+lookup(data2[3], kv2[3])+lookup(data2[4], kv2[4])+lookup(data2[5], kv2[5]) \
+lookup(data2[6], kv2[6])+lookup(data2[7], kv2[7])+lookup(data2[8], kv2[8]) \
+lookup(data2[9], kv2[9])+lookup(data2[10], kv2[10])+lookup(data2[11], kv2[11]) \
+lookup(data2[12], kv2[12])+lookup(data2[13], kv2[13])+lookup(data2[14], kv2[14]) \
+lookup(data2[15], kv2[15])+lookup(data2[16], kv2[16])>2048?1:0 as val_cv2 \
from (select data2, kv1 as kv2 \
from (select groupArray(data1) as data2 \
from ( \
    select any(ln_id1) as ln_id2, sumbin(lv1) as data1 from linear3_ln1_2 group by bin_id) \
group by ln_id2) as t_l2, linear3_kerne_ln1_2);" 








