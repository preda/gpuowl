#!/usr/bin/python3

import sys

cases = []
for line in sys.stdin:
    (exp, f, bits, b1, b2) = map(int, line.strip().split(','))
    cases.append((b1, b2, exp, f))
    #print("%15d %15d %15d %d" % (b2, b1, exp, f))

cases.sort()

tasks = []
def task(b1, b2, exp, f):
    #print("# %d %d %d %d" % (ob1, ob2, exp, f))
    tasks.append((b1, b2, exp))

def roundUp(x, m):
    return ((x - 1) // m + 1) * m

def gen(limitB1, limitB2, stepB1, stepB2):
    for (b1, b2, exp, f) in cases:
        if b1 <= limitB1 and b2 <= limitB2:
            task(roundUp(b1, stepB1), roundUp(b2, stepB2), exp, f)

gen(10000, 100000, 5000, 50000)

tasks.sort()
for b1, b2, exp in tasks:
    print("B1=%d,B2=%d;PFactor=%d" % (b1, b2, exp))

#gen(10000)
#gen(20000)
#gen(40000)
