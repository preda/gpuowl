#!/usr/bin/env python3

from math import *

# Table of values of Dickman's "rho" function for argument from 2 in steps of 1/20.
# Was generated in SageMath: [dickman_rho(x/20.0) for x in range(40,142)]
rhotab = [
 # 2
 0.306852819440055, 0.282765004395792,
 0.260405780162154, 0.239642788276221,
 0.220357137908328, 0.202441664262192,
 0.185799461593866, 0.170342639724018,
 0.155991263872504, 0.142672445952511,
 0.130319561832251, 0.118871574006370,
 0.108272442976271, 0.0984706136794386,
 0.0894185657243129, 0.0810724181216677,
 0.0733915807625995, 0.0663384461579859,
 0.0598781159863707, 0.0539781578442059,
 # 3
 0.0486083882911316, 0.0437373330511146,
 0.0393229695406371, 0.0353240987411619,
 0.0317034445117801, 0.0284272153221808,
 0.0254647238733285, 0.0227880556511908,
 0.0203717790604077, 0.0181926910596145,
 0.0162295932432360, 0.0144630941418387,
 0.0128754341866765, 0.0114503303359322,
 0.0101728378150057, 0.00902922680011186,
 0.00800687218838523, 0.00709415486039758,
 0.00628037306181464, 0.00555566271730628,
 # 4
 0.00491092564776083, 0.00433777522517762,
 0.00382858617381395, 0.00337652538864193,
 0.00297547478958152, 0.00261995369508530,
 0.00230505051439257, 0.00202636249613307,
 0.00177994246481535, 0.00156225163688919,
 0.00137011774112811, 0.00120069777918906,
 0.00105144485543239, 0.000920078583646128,
 0.000804558644792605, 0.000703061126353299,
 0.000613957321970095, 0.000535794711233811,
 0.000467279874773688, 0.000407263130174890,
 # 5
 0.000354724700456040, 0.000308762228684552,
 0.000268578998820779, 0.000233472107922766,
 0.000202821534805516, 0.000176080503619378,
 0.000152766994802780, 0.000132456257345164,
 0.000114774196621564, 0.0000993915292610416,
 0.0000860186111205116, 0.0000744008568854185,
 0.0000643146804615109, 0.0000555638944463892,
 0.0000479765148133912, 0.0000414019237006278,
 0.0000357083490382522, 0.0000307806248038908,
 0.0000265182000840266, 0.0000228333689341654,
 # 6
 0.0000196496963539553, 0.0000169006186225834,
 0.0000145282003166539, 0.0000124820385512393,
 0.0000107183044508680, 9.19890566611241e-6,
 7.89075437420041e-6, 6.76512728089460e-6,
 5.79710594495074e-6, 4.96508729255373e-6,
 4.25035551717139e-6, 3.63670770345000e-6,
 3.11012649979137e-6, 2.65849401629250e-6,
 2.27134186228307e-6, 1.93963287719169e-6,
 1.65557066379923e-6, 1.41243351587104e-6,
 1.20442975270958e-6, 1.02657183986121e-6,
 # 7
 8.74566995329392e-7, 7.44722260394541e-7, 0
]

def interpolate(tab, x):
    assert(x >= 0)
    ix = int(x)
    # print(x, ix)
    return tab[-1] if ix + 1 >= len(tab) else (tab[ix] + (x - ix) * (tab[ix + 1] - tab[ix]))

def rho(x):    
    return 1 if x <= 1 else (1 - log(x) if x < 2 else interpolate(rhotab, (x-2)*20))

def integral(a, b, f, STEPS = 20):
    w = b - a
    assert(w >= 0)
    if w == 0:
        return 0
    step = w / STEPS
    return step * sum([f(a + step * (0.5 + i)) for i in range(STEPS)])

def miu(a, b):
    return rho(a) + integral(a - b, a - 1, lambda t: rho(t)/(a-t))

def pm1(exponent, factoredTo, B1, B2):
    B2 = max(B1, B2) # B2 can't be lower than B1.
    takeAwayBits = log2(exponent) + 1
    SLICE_WIDTH = 0.25
    MIDDLE_SHIFT = log2(1 + 2**SLICE_WIDTH) - 1
    bitsFactor = factoredTo + MIDDLE_SHIFT - takeAwayBits
    bitsB1 = log2(B1)
    bitsB2 = log2(B2)
    alpha = bitsFactor / bitsB1
    beta = bitsB2 / bitsB1
    EPSILON = 1e-7
    sum = 0
    p = 1
    nSlice = factoredTo / SLICE_WIDTH + 0.5
    while p >= EPSILON:
        p = miu(alpha, beta) / nSlice
        sum += p
        alpha += SLICE_WIDTH / bitsB1
        nSlice += 1
    return -expm1(-sum)

def nPrimesBetween(B1, B2):
    return max(0, B2/log(B2) - B1/log(B1))

def workForBounds(B1, B2, factorB1=1.1, factorB2=1.35):
    # 1.442 is an approximation of log(powerSmooth(N))/N, the bitlen-expansion of powerSmooth().
    # 0.85 is an estimation of the ratio of primes remaining after "pairing" in second stage.
    return B1 * 1.442 * factorB1 + nPrimesBetween(B1, B2) * 0.85 * factorB2

def gain(exponent, factoredTo, B1, B2):
    return pm1(exponent, factoredTo, B1, B2) - workForBounds(B1, B2) / exponent

def walkGain(exponent, factoredTo):
    print(f'Exponent {exponent} factored to {factoredTo}: ', end='')
    B1 = 1000000
    B2 = 30000000
    stepB1 = 50000
    stepB2 = 500000
    while True:
        best = gain(exponent, factoredTo, B1, B2)
        bestB1 = B1
        bestB2 = B2
        # p = pm1(exponent, factoredTo, B1, B2)
        # print(p * 100, best * 100, B1, B2)
        for (tryB1, tryB2) in [(B1 - stepB1, B2), (B1 + stepB1, B2), (B1, B2 - stepB2), (B1, B2 + stepB2)]:
            if tryB1 <= 0 or tryB2 <= tryB1:
                continue
            p = gain(exponent, factoredTo, tryB1, tryB2)
            if p > best:
                (best, bestB1, bestB2) = (p, tryB1, tryB2)
        if bestB1 == B1 and bestB2 == B2:
            break
        (B1, B2) = (bestB1, bestB2)
    #print('')
    p = pm1(exponent, factoredTo, B1, B2)
    print(p * 100, best * 100, B1, B2)



walkGain(330000000, 76)

walkGain(100000000, 77)

walkGain(100000000, 78)
