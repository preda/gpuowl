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
    try:
        return hwmonBase + os.listdir(hwmonBase)[0] + '/'
    except FileNotFoundError:
        return None
    
def read(path):
    with open(path) as f:
        return f.read().strip()

def readOr(path, fallback=''):
    try:
        return read(path)
    except FileNotFoundError:
        return fallback

def readInt(path, scale=1e-6):
    return int(float(readOr(path, '0')) * scale + 0.5)

@dataclass
class Gpu:
    uid: str
    temps: List[int]
    fan: int 
    power: int
    sclk: int
    mclk: int
    voltage: int
    pcieRead: int
    pcieWrite: int
    pcieErr: int
    memBusy: int
    memUsedGB: float
    pciId: str

def readGpu(d: int, readSlow = False):
    device = drm + f'card{d}/device/'
    uid = readOr(device + 'unique_id', '----------------')
    pcieRead, pcieWrite, pcieSize = map(int, read(device + 'pcie_bw').split()) if readSlow else (0, 0, 0)
    pcieErr = readInt(device + 'pcie_replay_count', 1)
    memBusy = readInt(device + 'mem_busy_percent', 1)
    memUsed = readInt(device + 'mem_info_vram_used', 1)
    memUsedGB = memUsed * (1.0 / (1024 * 1024 * 1024))
    # pciId = read(device + 'thermal_throttling_logging').split()[0]
    pciId = read(device + 'uevent').split('PCI_SLOT_NAME=')[1].split()[0].lstrip('0000:')
    
    hwmon = hwmonPath(device)
    if hwmon:
        temps = [(int(read(hwmon + f'temp{i}_input')) + 500) // 1000 for i in range(1, 4)]
        fan = readInt(hwmon + 'fan1_input', 1)
        power = readInt(hwmon + 'power1_average')
        sclk = readInt(hwmon + 'freq1_input')
        mclk = readInt(hwmon + 'freq2_input')
        voltage = readInt(hwmon + 'in0_input', 1)
    else:
        temps = [0, 0, 0]
        fan = 0
        power = 0
        sclk = 0
        mclk = 0
        voltage = 0
    return Gpu(uid=uid, temps=temps, fan=fan, power=power, sclk=sclk, mclk=mclk, voltage=voltage,
               pcieRead=pcieRead, pcieWrite=pcieWrite, pcieErr=pcieErr, memBusy=memBusy, memUsedGB=memUsedGB, pciId=pciId)

def printInfo(devices, readSlow):
    print(datetime.now())    
    print('# PCI     UID              VDD   SCLK MCLK Mem-used Mem-busy PWR  FAN  Temp     PCIeErr' + (' PCIe R/W' if readSlow else '') + ' ')
    for d in devices:
        gpu = readGpu(d, readSlow)
        temps = '/'.join((str(x) for x in gpu.temps))
        print(('%(card)d %(pciId)s %(uid)s %(voltage)dmV %(sclk)4d %(mclk)4d %(memUsedGB)5.2fGB    %(memBusy)2d%%    %(power)3dW %(fan)4d %(temps)s %(pcieErr)7d' + (' %(pcieRead)d/%(pcieWrite)d' if readSlow else ''))
              % dict(gpu.__dict__, card=d, temps=temps))
    
devices = deviceList()

readSlow = len(sys.argv) >= 2 and sys.argv[1] == '-s'

sleep = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[1] == '-t' else None

printInfo(devices, readSlow)
while sleep:
    time.sleep(sleep)
    print('\n', datetime.now())
    printInfo(devices, readSlow)
