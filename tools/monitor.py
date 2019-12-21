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

def hwmonPath(device):
    hwmonBase = device + 'hwmon/'
    return hwmonBase + os.listdir(hwmonBase)[0] + '/'
    
def read(path):
    with open(path) as f:
        return f.read().strip()

def readInt(path, scale=1e-6):
    return int(float(read(path)) * scale + 0.5)

@dataclass
class Gpu:
    uid: str
    temps: List[int]
    fan: int 
    power: int
    sclk: int
    mclk: int
    voltage: int

def readGpu(d: int):
    device = drm + f'card{d}/device/'
    uid = read(device + 'unique_id')
        
    hwmon = hwmonPath(device)
    temps = [(int(read(hwmon + f'temp{i}_input')) + 500) // 1000 for i in range(1, 4)]
    fan = readInt(hwmon + 'fan1_input', 1)
    power = readInt(hwmon + 'power1_average')
    sclk = readInt(hwmon + 'freq1_input')
    mclk = readInt(hwmon + 'freq2_input')
    voltage = readInt(hwmon + 'in0_input', 1)
    return Gpu(uid=uid, temps=temps, fan=fan, power=power, sclk=sclk, mclk=mclk, voltage=voltage)

def printInfo(devices):
    print('GPU UID            VDD   SCLK MCLK PWR  FAN  Temp      ' + str(datetime.now()))
    for d in devices:
        gpu = readGpu(d)
        temps = '/'.join((str(x) for x in gpu.temps))
        print('%(card)d %(uid)s %(voltage)dmV %(sclk)s %(mclk)s %(power)3dW %(fan)4d %(temps)sC' %
              dict(gpu.__dict__, card=d, temps=temps))
    print()
    
devices = deviceList()

sleep = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[1] == '-t' else None

printInfo(devices)
while sleep:
    time.sleep(sleep)
    printInfo(devices)
