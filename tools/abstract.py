#!/usr/bin/python3

import sys

for line in sys.stdin:
    if line.startswith('\tv_') or line.startswith('\ts_') or line.startswith('\tglobal_') or line.startswith('\tds_'):
        print(line.strip().split()[0])
    else:
        print(line, end='')
