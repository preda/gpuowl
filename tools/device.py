#!/usr/bin/python3

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List
from datetime import datetime

drm = '/sys/class/drm/'

def deviceList():
    return sorted([int(d[4:]) for d in os.listdir(drm) if re.match(r'^card\d+$', d)])

def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ''
    

def device(d):
    return drm + f'card{d}/device/'
    
def uid(d):
    return read(device(d) + 'unique_id')


if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <unique-id>\nE.g. {sys.argv[0]} 780c28c172da0000\nreturns the id of the device with the given unique-id')
    sys.exit(1)

# print(f'"{sys.argv[1]}"')
    
for d in deviceList():
    if uid(d) == sys.argv[1]:
        print(d)
        exit(0)
