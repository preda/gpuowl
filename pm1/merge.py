#!/usr/bin/env python3

import sys

prev = 0
primes = []
for line in sys.stdin:
    (z, p) = map(int, line.strip().split())
    if z != prev:
        if primes:
            print(prev, ' '.join(map(str, sorted(primes))))
        prev = z
        primes = []
    primes.append(p)
    
if primes:
    print(prev, ' '.join(map(str, sorted(primes))))
