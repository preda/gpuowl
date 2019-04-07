#!/usr/bin/python3

import sys

cases = []
for line in sys.stdin:
    (exp, f, bits, b1, b2) = map(int, line.strip().split(','))
    cases.append((b1, b2, exp, f))
    #print("%15d %15d %15d %d" % (b2, b1, exp, f))

cases.sort()

def task(b1, b2, exp, f):
    #print("# %d %d %d %d" % (ob1, ob2, exp, f))
    print("B1=%d,B2=%d;PFactor=%d" % (b1, b2, exp))

def nextMul(x, m):
    return ((x - 1) // m + 1) * m

def gen(base, mul = 30):
    for (b1, b2, exp, f) in cases:
        if b1 < 5 * base and b2 < 5 * base * mul:
            b1 = nextMul(b1, base)
            b2 = nextMul(max(b2, b1 + 1), base * mul)
            task(b1, b2, exp, f)

#gen(10000)
gen(20000)
gen(40000)
