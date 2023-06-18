#!/usr/bin/env python3

import logging
import sys

import impprof


logging.basicConfig(format="%(asctime)s [%(levelname)s:%(name)s] - %(message)s")
logging.getLogger("impprof").setLevel(logging.DEBUG)

with impprof.Monitor() as mon:
    __import__(sys.argv[1])

for i in mon.imports.values():
    print(i)
