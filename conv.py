#!/usr/bin/python3

import numpy
import math

M31 = 2**31 - 1
M61 = 2**61 - 1

def mod(x):
    (a, b) = x
    return (a % M31, b % M31)

def mod61(x):
    (a, b) = x
    return (a % M61, b % M61)

def rmul(x, y):
    (a, b) = x
    (c, d) = y

    k1 = a *  (c + d)
    k2 = c *  (b - a)
    k3 = d * -(b + a)

    re = k1 + k3
    im = k1 + k2
    return (re, im)

def mmul(x, y):
    return mod(rmul(x, y))


#z = (2**15, 0x2997cb94)
z = (2**16, 0x4b94532f)

def pow2(z, k):
    for i in range(k):
        z = mmul(z, z)
    return z
        
for i in range(32):
    z = mmul(z, z)
    print(i+1, '%x %x'%z)

z61 = (2**31, 0xe5718ad1b2a95b8)
z61 = (2**30, 0x6caa56e1cae315a)

x = z61
for i in range(62):
    x = mod61(rmul(x, x))
    print(i + 1, '%x %x' % x)


    
tau = math.pi * 2

fft = numpy.fft.fft
ifft = numpy.fft.ifft

#vs = [6, 3, 7, 1, 1, 0, 2, 3]

vs = [2, 1, 3, 0, 0, 0, 0, 0]

ws = [0, 5, 2, 7, 4, 1, 6, 3]

bitlen = [3, 2, 3, 2, 2, 3, 2, 2]


def reduce(x):
    return (x & 0x7fffffff) + (x >> 31)

def weight(vs, ws, r):
    out = []
    for (x, w) in zip(vs, ws):
        shift = r * w % 31
        x = reduce(x << shift)
        out.append(x)
    return out

def weight2(vs, ws, r):
    return [x * (r ** (w / 8)) for (x, w) in zip(vs, ws)]

def toComplex(vs):
    return [a + 1j * b for (a, b) in zip(vs[0::2], vs[1::2])]

def conv2(vs):
    weight2(vs, ws, 2)
    vs = fft(vs)
    vs = [x * x for x in vs]
    vs = ifft(vs)
    return weight2(vs, ws, 1/2)

def conv3(vs):
    vs = fft(vs)
    vs = [x * x for x in vs]
    return ifft(vs)

def conjugate(x):
    (a, b) = x
    return mod((a, -b))

def rX(a, b):
    (ar, ai) = a
    (br, bi) = b
    return ((ar + br, ai + bi), (ar - br, ai - bi))

def mX(a, b):
    (a, b) = rX(a, b)
    return (mod(a), mod(b))
    
def fft4Core(v, W, X, mul):
    (a, b, c, d) = v
    (a, c) = X(a, c)
    (b, d) = X(b, d)
    d = mul(d, W)
    #d *= -1j
    (a, b) = X(a, b)
    (c, d) = X(c, d)
    return (a, b, c, d)

def fft4(v, W, X, mul):
    (a, b, c, d) = fft4Core(v, W, X, mul)
    (b, c) = (c, b)
    return (a, b, c, d)

def fft8(v, W, X, mul):
    #w = math.e ** (-1j * tau / 8)    
    (a, b, c, d, e, f, g, h) = v
    (a, e) = X(a, e)
    (b, f) = X(b, f)
    (c, g) = X(c, g)
    (d, h) = X(d, h)
    f = mul(f, W)
    W2 = mul(W, W)
    g = mul(g, W2)
    h = mul(h, mul(W2, W))
    (a, b, c, d) = fft4Core((a, b, c, d), W2, X, mul)
    (e, f, g, h) = fft4Core((e, f, g, h), W2, X, mul)
    return (a, e, c, g, b, f, d, h)

def div2(x):
    (a, b) = x
    return (a * 2**30 % M31, b * 2**30 % M31)

def fft8a(v, W, X, mul):
    v = tuple(zip(*v))[0]
    v = zip(v[::2], v[1::2])
    v = fft4(v, mul(W, W), X, mul)
    
    print(v)
    
    v = list(v)
    (a, b) = v[0]
    v[0] = mod((a + b, a - b))
    
    a = v[1]
    b = conjugate(v[3])
    #b = v[3]
    
    (a, b) = X(a, b)
    (wa, wb) = W
    b = mul(b, mod(conjugate((wb, wa))))
    (a, b) = X(a, b)
    b = conjugate(b)
    v[1] = div2(a)
    v[3] = div2(b)
    return v
    

def ifft8r(v):
    v = map(lambda x : complex(x).conjugate(), v)
    v = fft8(v)
    v = map(lambda x : complex(x).conjugate() / 8, v)
    return list(v)

def conj(v): return ((x, -y) for (x, y) in v)

def ifft8(v, W, X, mul, div8):    
    v = conj(v)
    v = fft8(v, W, X, mul)
    v = conj(v)
    return tuple(((div8(a), div8(b)) for (a, b) in v))
#return tuple(((a * 2**28 % M31, b * 2**28 % M31) for (a, b) in v))

def rsq(x):
    (a, b) = x
    return ((a + b) * (a - b), a * b * 2)

def sq(x):
    return mod(rsq(x))

def pow(x, k):
    y = x
    for i in range(k - 1):
        y = mmul(y, x)
    return y

w8 = (2**15, 2**15)

#for i in range(8): print(i, pow(w8, i))


w8i = (w8[0], -w8[1])
print(w8, mmul(w8, w8), mmul(w8i, w8i), pow(w8, 7))
w8inv = pow(w8, 7)

#print(w8i, w8inv, mmul(w8i, w8i), mmul(w8inv, w8inv))

v = (6, 3, 7, 1, 8, 4, 2, 5)
v = tuple(zip(v, [0]*8))
print(v)
a = fft8(v, w8, mX, mmul)
print(a)

aa = fft8a(v, w8, mX, mmul)
print(aa)

exit(0)

def div8(x): return x * 2**28 % M31

b = ifft8(a, w8, mX, mmul, div8)
print(b)

a2 = tuple(map(sq, a))
print(a2)
bb = ifft8(a2, w8, mX, mmul, div8)
print(bb)


print("\nreal\n")

rw8 = (2**(-1/2), -2**(-1/2))
ra = fft8(v, rw8, rX, rmul)
print(ra)

rb = ifft8(ra, rw8, rX, rmul, lambda x : x / 8)
print(rb)

ra2 = tuple(map(rsq, ra))
print(a2)
rbb = ifft8(ra2, rw8, rX, rmul, lambda x : x / 8)
print(rbb)



exit(0)

v = [5, 7, 2, 1]
print(fft4(v))
print(list(fft(v)))

v = [6, 3, 7, 1, 1, 0, 2, 3]
print(fft8(v))
print(list(fft(v)))
a = fft8(v)

print(ifft8(a))
print(list(ifft(a)))


z0 = (2**15, 0x2997cb94)
z  = (2**16, 0x4b94532f)





def pow2(z, k):
    for i in range(k):
        z = mmul(z, z, M31)
    return z
        
for i in range(32):
    z = mmul(z, z, 2**31 - 1)
    print(i+1, '%x %x'%z)

#print(fft8(vs, math.e ** (-1j * tau / 8)))
#print(vs)
#print(fft(vs))


#print(conv(vs))


#print(conv2(vs + ([0] * 8)))
#cs = toComplex(vs)
#bs = weight(vs, ws, 4)
#print(bs, toComplex(bs))

