#!/usr/bin/env python3

from math import *

# Table of values of Dickman's "rho" function for argument from 2 in steps of 1/20.
# Was generated in SageMath: [dickman_rho(2 + x/20.0) for x in range(280)]
rhotab = [
#2
0.306852819440055, 0.282765004395792, 0.260405780162154, 0.239642788276221, 0.220357137908328, 0.202441664262192, 0.185799461593866, 0.170342639724018, 0.155991263872504, 0.142672445952511,
0.130319561832251, 0.118871574006370, 0.108272442976271, 0.0984706136794386, 0.0894185657243129, 0.0810724181216677, 0.0733915807625995, 0.0663384461579859, 0.0598781159863707, 0.0539781578442059,
#3
0.0486083882911316, 0.0437373330511146, 0.0393229695406371, 0.0353240987411619, 0.0317034445117801, 0.0284272153221808, 0.0254647238733285, 0.0227880556511908, 0.0203717790604077, 0.0181926910596145,
0.0162295932432360, 0.0144630941418387, 0.0128754341866765, 0.0114503303359322, 0.0101728378150057, 0.00902922680011186, 0.00800687218838523, 0.00709415486039758, 0.00628037306181464, 0.00555566271730628,
#4
0.00491092564776083, 0.00433777522517762, 0.00382858617381395, 0.00337652538864193, 0.00297547478958152, 0.00261995369508530, 0.00230505051439257, 0.00202636249613307, 0.00177994246481535, 0.00156225163688919,
0.00137011774112811, 0.00120069777918906, 0.00105144485543239, 0.000920078583646128, 0.000804558644792605, 0.000703061126353299, 0.000613957321970095, 0.000535794711233811, 0.000467279874773688, 0.000407263130174890,
#5
0.000354724700456040, 0.000308762228684552, 0.000268578998820779, 0.000233472107922766, 0.000202821534805516, 0.000176080503619378, 0.000152766994802780, 0.000132456257345164, 0.000114774196621564, 0.0000993915292610416,
0.0000860186111205116, 0.0000744008568854185, 0.0000643146804615109, 0.0000555638944463892, 0.0000479765148133912, 0.0000414019237006278, 0.0000357083490382522, 0.0000307806248038908, 0.0000265182000840266, 0.0000228333689341654,
#6
0.0000196496963539553, 0.0000169006186225834, 0.0000145282003166539, 0.0000124820385512393, 0.0000107183044508680, 9.19890566611241e-6, 7.89075437420041e-6, 6.76512728089460e-6, 5.79710594495074e-6, 4.96508729255373e-6,
4.25035551717139e-6, 3.63670770345000e-6, 3.11012649979137e-6, 2.65849401629250e-6, 2.27134186228307e-6, 1.93963287719169e-6, 1.65557066379923e-6, 1.41243351587104e-6, 1.20442975270958e-6, 1.02657183986121e-6,
#7
8.74566995329392e-7, 7.44722260394541e-7, 6.33862255545582e-7, 5.39258025342825e-7, 4.58565512804405e-7, 3.89772368391109e-7, 3.31151972577348e-7, 2.81223703587451e-7, 2.38718612981323e-7, 2.02549784558224e-7,
1.71786749203399e-7, 1.45633412099219e-7, 1.23409021080502e-7, 1.04531767460094e-7, 8.85046647687321e-8, 7.49033977199179e-8, 6.33658743306062e-8, 5.35832493603539e-8, 4.52922178102003e-8, 3.82684037781748e-8,
#8
3.23206930422610e-8, 2.72863777994286e-8, 2.30269994373198e-8, 1.94247904820595e-8, 1.63796304411581e-8, 1.38064422807221e-8, 1.16329666668818e-8, 9.79786000820215e-9, 8.24906997200364e-9, 6.94244869879648e-9,
5.84056956293623e-9, 4.91171815795476e-9, 4.12903233557698e-9, 3.46976969515950e-9, 2.91468398787199e-9, 2.44749453802384e-9, 2.05443505293307e-9, 1.72387014435469e-9, 1.44596956306737e-9, 1.21243159178189e-9,
#9
1.01624828273784e-9, 8.51506293255724e-10, 7.13217989231916e-10, 5.97178273686798e-10, 4.99843271868294e-10, 4.18227580146182e-10, 3.49817276438660e-10, 2.92496307733140e-10, 2.44484226227652e-10, 2.04283548915435e-10,
1.70635273863534e-10, 1.42481306624186e-10, 1.18932737801671e-10, 9.92430725748863e-11, 8.27856490334434e-11, 6.90345980053579e-11, 5.75487956079478e-11, 4.79583435743883e-11, 3.99531836601083e-11, 3.32735129630055e-11,
#10
2.77017183772596e-11, 2.30555919904645e-11, 1.91826261797451e-11, 1.59552184492373e-11, 1.32666425229607e-11, 1.10276645918872e-11, 9.16370253824348e-12, 7.61244195636034e-12, 6.32183630823821e-12, 5.24842997441282e-12,
4.35595260905192e-12, 3.61414135533970e-12, 2.99775435412426e-12, 2.48574478117179e-12, 2.06056954190735e-12, 1.70761087761789e-12, 1.41469261268532e-12, 1.17167569925493e-12, 9.70120179176324e-13, 8.03002755355921e-13,
#11
6.64480907032201e-13, 5.49695947730361e-13, 4.54608654601190e-13, 3.75862130571052e-13, 3.10667427553834e-13, 2.56708186340823e-13, 2.12061158957008e-13, 1.75129990979628e-13, 1.44590070306053e-13, 1.19342608376890e-13,
9.84764210448520e-14, 8.12361284968988e-14, 6.69957047626884e-14, 5.52364839983536e-14, 4.55288784872365e-14, 3.75171868260434e-14, 3.09069739955730e-14, 2.54545912496319e-14, 2.09584757642051e-14, 1.72519300955857e-14,
#12
1.41971316501794e-14, 1.16801642038076e-14, 9.60689839298851e-15, 7.89957718055663e-15, 6.49398653148027e-15, 5.33711172323687e-15, 4.38519652833446e-15, 3.60213650413600e-15, 2.95814927457727e-15, 2.42867438017647e-15,
1.99346333303212e-15, 1.63582721456795e-15, 1.34201472284939e-15, 1.10069820297832e-15, 9.02549036511458e-16, 7.39886955899583e-16, 6.06390497499970e-16, 4.96858003320228e-16, 4.07010403543137e-16, 3.33328522514641e-16,
#13
2.72918903047290e-16, 2.23403181509686e-16, 1.82826905742816e-16, 1.49584399704446e-16, 1.22356868095946e-16, 1.00061422004550e-16, 8.18091101788785e-17, 6.68703743742468e-17, 5.46466232309370e-17, 4.46468473170557e-17,
3.64683865173660e-17, 2.97811167122010e-17, 2.43144513286369e-17, 1.98466595514452e-17, 1.61960906400940e-17, 1.32139661280568e-17, 1.07784613453433e-17, 8.78984690826589e-18, 7.16650138491662e-18, 5.84163977794677e-18,
#14
4.76063001400521e-18, 3.87879232126172e-18, 3.15959506343337e-18, 2.57317598320038e-18, 2.09513046990837e-18, 1.70551888483764e-18, 1.38805354722395e-18, 1.12943303162933e-18, 9.18797221060242e-19, 7.47281322095490e-19,
6.07650960951011e-19, 4.94003693444398e-19, 4.01524901266115e-19, 3.26288213964971e-19, 2.65092374707276e-19, 2.15327927385602e-19, 1.74868299982827e-19, 1.41980841083036e-19, 1.15254171584394e-19, 9.35388736783942e-20,
#15
7.58990800429806e-20, 6.15729693405857e-20, 4.99405370840484e-20, 4.04973081615272e-20, 3.28329006413784e-20, 2.66135496324475e-20, 2.15678629328980e-20, 1.74752135068077e-20, 1.41562828504629e-20, 1.14653584509271e-20,
9.28406140589761e-21, 7.51623982263034e-21, 6.08381226695129e-21, 4.92338527497562e-21, 3.98350139454904e-21, 3.22240072043320e-21, 2.60620051521272e-21, 2.10741515728752e-21, 1.70375305656048e-21, 1.37713892323882e-21,
#20
2.2354265870871718e-27]

def interpolate(tab, x):
    assert(x >= 0)
    ix = int(x)
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

# Probabiliy of first stage success.
def pFirstStage(a):
    return rho(a)

# Probability of second stage success
def pSecondStage(a, b):
    return integral(a - b, a - 1, lambda t: rho(t)/(a-t))

# See "Some Integer Factorization Algorithms using Elliptic Curves", R. P. Brent, page 3.
# https://maths-people.anu.edu.au/~brent/pd/rpb102.pdf
# Also "Speeding up Integer Multiplication and Factorization", A. Kruppa, chapter 5.3.3 (page 102).
def miu(a, b):
    # We approximate the two events as disjoint, thus we simply add them.
    return pFirstStage(a) + pSecondStage(a, b)

# Approximation of the number of primes <= n.
# The correction term "-1.06" was determined experimentally to improve the approximation.
def primepi(n):
    return n / (log(n) - 1.06)

def nPrimesBetween(B1, B2):
    assert(B2 >= B1)
    return primepi(B2) - primepi(B1)

def workForBounds(B1, B2, factorB1=1.2, factorB2=1.35):
    # 1.442 is an approximation of log(powerSmooth(N))/N, the bitlen-expansion of powerSmooth().
    # 0.85 is an estimation of the ratio of primes remaining after "pairing" in second stage.
    return (B1 * 1.442 * factorB1, nPrimesBetween(B1, B2) * 0.85 * factorB2)

def fmtBound(b):
    s = f'{b//1000000}M' if not b % 1000000 else f'{b/1000000:1}M' if b > 1000000 else f'{b//1000}K' if not b % 1000 else f'{b/1000}' if b > 1000 else f'{b}'
    return f'{s:>4}'
    
# steps of approx 10%
niceStep = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 45, 50, 55, 60, 65, 70, 80, 90]

# use nice round values for bounds.
def nextNiceNumber(value):
    ret = 1
    while value >= niceStep[-1]:
        value //= 10
        ret *= 10
    for n in niceStep:
        if n > value:
            return n * ret
    assert(False)

class PM1:
    SLICE_WIDTH = 0.25
    MIDDLE_SHIFT = log2(1 + 2**SLICE_WIDTH) - 1

    def __init__(self, exponent, factoredTo):
        self.exponent = exponent
        self.factoredTo = factoredTo

        # Bits removed because of special form of factors 2*k*exponent + 1
        self.takeAwayBits = log2(exponent) + 1
        
    def pm1(self, B1, B2):
        B2 = max(B1, B2) # B2 can't be lower than B1.
        bitsB1 = log2(B1)
        bitsB2 = log2(B2)
        alpha = (self.factoredTo + self.MIDDLE_SHIFT - self.takeAwayBits) / bitsB1
        alphaStep = self.SLICE_WIDTH / bitsB1
        beta = bitsB2 / bitsB1
        
        sum1 = 0
        sum2 = 0
        invSliceProb = self.factoredTo / self.SLICE_WIDTH + 0.5
        p = 1

        while p >= 1e-8:
            sliceProb = 1 / invSliceProb
            p1 = pFirstStage(alpha) * sliceProb
            p2 = pSecondStage(alpha, beta) * sliceProb
            sum1 += p1
            sum2 += p2
            p = p1 + p2
            alpha += alphaStep
            invSliceProb += 1
        # print(f'stopped at {nSlice * SLICE_WIDTH}')
        return (-expm1(-sum1), -expm1(-sum2))

    # return tuple (benefit, work) expressed as a ratio of one PRP test
    def gain(self, B1, B2):
        (p1, p2) = self.pm1(B1, B2)
        (w1, w2) = workForBounds(B1, B2)
        p = p1 + p2
        # the formula below models one GCD after first stage, one GCD in the middle of second stage, and one GCD at the end.
        w = (w1 + (1 - p1 - p2/4) * w2) * (1 / self.exponent)
        return (p, w)

    def walk(self, *, debug=None, B1=None, B2=None):
        fixB1, fixB2 = B1, B2
        debug and print(f'Exponent {self.exponent} factored to {self.factoredTo}:')
        B1 = fixB1 if fixB1 else nextNiceNumber(int(self.exponent / 1000))
        B2 = max(B1, fixB2 if fixB2 else nextNiceNumber(int(self.exponent / 100)))
        
        smallB1, smallB2 = 0, 0
        midB1, midB2 = 0, 0
        
        (p, w) = self.gain(B1, B2)

        while True:
            stepB1 = nextNiceNumber(B1) - B1
            stepB2 = nextNiceNumber(B2) - B2
            (p1, w1) = (p, w + 1) if fixB1 else self.gain(B1 + stepB1, B2)
            (p2, w2) = (p, w + 1) if fixB2 else self.gain(B1, B2 + stepB2)

            assert(w1 > w and w2 > w and p1 >= p and p2 >= p)
            r1 = (p1 - p) / (w1 - w)
            r2 = (p2 - p) / (w2 - w)

            # first time both rates go under 1 marks the point of diminishing returns from P-1; save max-efficient bounds.
            isSmallPoint = r1 < 1 and r2 < 1 and not smallB1
            if isSmallPoint:
                smallB1 = B1
                smallB2 = B2

            # some intermediate bounds between max-efficient and good-factor-finding
            isMidPoint = r1 < .5 and r2 < .5 and not midB1
            if isMidPoint:
                midB1 = B1
                midB2 = B2

            isBigPoint = r1 < 1 and r2 < 1 and p1 <= w1 and p2 <= w2
            
            debug and print(f'{fmtBound(B1)}, {fmtBound(B2)} : (p={p*100:.3f}%, work={w*100:.3f}%), B1 step {r1:.3f}={(p1-p)*100:.4f}/{(w1-w)*100:.4f}, B2 step {r2:.3f}={(p2-p)*100:.4f}/{(w2-w)*100:.4f}', '[MIN]' if isSmallPoint else '[MID]' if isMidPoint else '[BIG]' if isBigPoint else '')

            if isBigPoint:
                break

            if r1 > r2:
                B1 += stepB1
                (p, w) = (p1, w1)
            else:
                B2 += stepB2
                (p, w) = (p2, w2)

        return ((smallB1, smallB2) if smallB1 else None, (midB1, midB2) if midB1 else None, (B1, B2))

    def printResult(self, bounds, label):
        if bounds:
            B1, B2 = bounds
            p1, p2 = self.pm1(B1, B2)
            w1, w2 = workForBounds(B1, B2)
            _, w = self.gain(B1, B2)
            w1, w2 = w1/self.exponent, w2/self.exponent
            print(f'{label}: B1={fmtBound(B1)}, B2={fmtBound(B2)} : p={100*(p1+p2):.2f}% ({100*p1:.2f}% + {100*p2:.2f}%), work={100*w:.2f}% ({w1*100:.2f}% + {w2*100:.2f}%)')

    def walkBounds(self, **named):
        for bounds, label in zip(self.walk(**named), ('[MIN]', '[MID]', '[BIG]')):
            self.printResult(bounds, label)

def walk(exponent, factored, **named):
    PM1(exponent, factored).walkBounds(**named)
            
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: pm1.py <exponent> <factoredTo> [-verbose] [-B1 <fixed B1>] [-B2 <fixed B2>]')
        print('Example: pm1.py 100000000 77')
        exit(1)
        
    exponent = int(sys.argv[1])
    factored = int(sys.argv[2])
    debug = False
    fixedB1, fixedB2 = None, None
    
    args = sys.argv[3:]
    while args:
        # print(args[0])
        if args[0] == '-verbose':
            debug = True
            args = args[1:]
        elif args[0] == '-B1':
            fixedB1 = int(args[1])
            args = args[2:]
        elif args[0] == '-B2':
            fixedB2 = int(args[1])
            args = args[2:]
        else:
            print('Unrecognized argument "f{args[0]}"')
            args = args[1:]
            
    walk(exponent, factored, debug=debug, B1=fixedB1, B2=fixedB2)
