#!/usr/bin/env python3

import importlib
import logging
import sys

import impprof


logging.basicConfig(format="%(asctime)s [%(levelname)s:%(name)s] - %(message)s")
logging.getLogger("impprof").setLevel(logging.DEBUG)

with impprof.Monitor() as mon:
    importlib.import_module(sys.argv[1])

for i in mon.imports.values():
    print(i)
