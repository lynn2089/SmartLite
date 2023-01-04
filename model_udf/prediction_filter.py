#!/usr/bin/python3

import sys
import json
import bitstring
from clickhouse_driver import Client
db = Client(host='localhost')
if __name__ == '__main__':
    for line in sys.stdin:
        value = json.loads(line)
        first_arg = int(value['argument_1'])
        second_arg = int(value['argument_2'])
        third_arg = int(value['argument_3'])
        len = first_arg.bit_length()/2
        field_1 = bitstring.BitArray(first_arg)
        field_1_a = field_1[0:len].uint
        field_1_b = field_1[len:len*2].uint
        field_2 = bitstring.BitArray(second_arg)
        field_2_a = field_2[0:len].uint
        field_2_b = field_2[len:len*2].uint
        q0 = "select Lookup("+field_1_a+","+field_1_a+");"
        q1 = "select Lookup("+field_1_b+","+field_1_b+");"
        res = db.execute(q0)
        if (res>third_arg):
            result = {0}
        else:
            if (res+db.execute(q1)>third_arg):
                result = {0}
            else:
                result = {1}
        print(json.dumps(result), end='\n')
        sys.stdout.flush()





