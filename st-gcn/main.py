# -*- coding: utf-8 -*-
# @Time     : 2022/2/17 14:08
# @Author   : xiehl
# @Software : PyCharm
# ---------------------------
#!/usr/bin/env python
import argparse
import sys

# torchlight
# import torchlight
# from torchlight import import_class
from torchlight import import_class




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    #
    processors = dict()
    processors['prediction'] = import_class('processor.prediction.REC_Processor')
    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    print("1", arg.processor)
    print("2", processors[arg.processor])
    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
