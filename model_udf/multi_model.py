#!/usr/bin/python3

import sys
import json
import resource
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from threading import Thread
from clickhouse_driver import Client
db = Client(host='localhost')
if __name__ == '__main__':
    for line in sys.stdin:
        value = json.loads(line)
        first_arg = int(value['argument_1'])
        second_arg = value['argument_2']
        result = {asyncio.run(load(second_arg))}
        print(json.dumps(result), end='\n')
        sys.stdout.flush()

async def load(first_arg,second_arg):
    num_model = second_arg
    num_thread = 10
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=num_thread)
    for j in range(1,num_model+1):
        # os.system('sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')
        # os.system('echo panda9105 | sudo sh -c \'echo 3 > /proc/sys/vm/drop_caches\'')
        threads = []
        for i in range(j):
            threads.append(loop.run_in_executor(pool, run_1, i, first_arg)) 
        done,pending = await asyncio.wait(threads)

def run_1(i,first_arg):
    model = "alexnet_"+i
    db.execute("select model_",first_arg,"(value,",model,");")

