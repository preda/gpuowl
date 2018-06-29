// gpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#define DEVICE static __device__
#define KERNEL static __global__
#define DUAL static __device__ __host__

#include "LowGpu.h"
#include "args.h"
#include "common.h"

#include <math.h>
#include <cufft.h>

#define CC(what) assert((what) == cudaSuccess)

pair<vector<double>, vector<double>> genWeights(uint E, uint N) {
  vector<double> aTab, iTab;
  aTab.reserve(N);
  iTab.reserve(N);

  int baseBits = E / N;
  auto iN = 1 / (long double) N;

  for (int k = 0; k < N; ++k) {
    int bits = bitlen(N, E, k);
    assert(bits == baseBits || bits == baseBits + 1);
    long double a  = exp2l(extra(N, E, k) * iN);
    long double ia = 1 / (N * a);
    aTab.push_back((bits == baseBits) ? a  : -a);
    iTab.push_back((bits == baseBits) ? ia : -ia);
  }
  assert(aTab.size() == N && iTab.size() == N);
  return std::make_pair(aTab, iTab);
}

DEVICE int carryStep(long x, long *carry, int bits) {
  x += *carry;
  int w = lowBits(x, bits);
  *carry = (x - w) >> bits;
  return w;
}

DEVICE uint signBit(double a) { return __double_as_longlong(a) < 0; } // return (a < 0);
DEVICE uint bitlen(uint base, double a) { return base + signBit(a); }
DEVICE long unweight(double x, double w) { return __double2ll_rn(x * fabs(w)); }

DEVICE double2 mul(double2 a, double2 b) { return double2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
DEVICE double2 sq(double2 a) { return double2{(a.x + a.y) * (a.x - a.y), 2 * a.x * a.y}; }

template<uint mul>
DEVICE int unweightAndCarry(uint base, double u, double w, long *carry) {
  return carryStep(mul * unweight(u, w), carry, bitlen(base, w));
}

/*
DEVICE int2 unweightAndCarry(uint base, uint mul, double2 u, long *carry, double2 w) {
  int a = carryStep(mul * unweight(u.x, w.x), carry, bitlen(base, w.x));
  int b = carryStep(mul * unweight(u.y, w.y), carry, bitlen(base, w.y));
  return int2{a, b};
}

DEVICE double2 carryAndWeight(uint base, int2 u, long carry, double2 w) {
  double a = carryStep(u.x, &carry, bitlen(base, w.x)) * fabs(w.x);
  double b = (u.y + carry) * fabs(w.y);
  return double2{a, b};
}
*/

DEVICE double carryAndWeight(uint base, int u, double w, long *carry) {
  return fabs(w) * carryStep(u, carry, bitlen(base, w));
}

template<uint mul>
KERNEL void carryA(uint base, const double4 *in, int4 *out, long *carryOut, const double4 *weight) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  long carry = 0;
  double4 u = in[id];
  double4 w = weight[id];
  int a = unweightAndCarry<mul>(base, u.x, w.x, &carry);
  int b = unweightAndCarry<mul>(base, u.y, w.y, &carry);
  int c = unweightAndCarry<mul>(base, u.z, w.z, &carry);
  int d = unweightAndCarry<mul>(base, u.w, w.w, &carry);
  out[id] = int4{a, b, c, d};
  u32 size = gridDim.x * blockDim.x;
  u32 next = (id + 1 == size) ? 0 : (id + 1);
  carryOut[next] = carry;
}

DEVICE void carryBcore(uint base, const int4 *in, long carry, double4 *out, const double4 *weight) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  int4 u = in[id];
  double4 w = weight[id];
  double a = carryAndWeight(base, u.x, w.x, &carry);
  double b = carryAndWeight(base, u.y, w.y, &carry);
  double c = carryAndWeight(base, u.z, w.z, &carry);
  double d = fabs(w.w) * (u.w + carry); // sink final carry into 'd'.
  out[id] = double4{a, b, c, d};
}

KERNEL void carryB(uint base, const int4 *in, const long *carryIn, double4 *out, const double4 *weight) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  carryBcore(base, in, carryIn[id], out, weight);
}

KERNEL void preWeight(uint base, const int4 *in, const double4 *weight, double4 *out) {
  carryBcore(base, in, 0, out, weight);
}

KERNEL void carryFinal(uint base, int4 *io, const double4 *weight, const long *carryIn) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  int4 u = io[id];
  double4 w = weight[id];
  long carry = carryIn[id];
  int a = carryStep(u.x, &carry, bitlen(base, w.x));
  int b = carryStep(u.y, &carry, bitlen(base, w.y));
  int c = carryStep(u.z, &carry, bitlen(base, w.z));
  int d = u.w + carry;
  io[id] = int4{a, b, c, d};
}

/*
KERNEL void readResidue(uint N2, const int2 *in, int2 *out) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  u32 p = (id >= 32) ? id - 32 : (N2 - 32 + id);
  out[id] = in[p];
}
*/

/*
// like carryA, but with mul3.
KERNEL void carryM(uint base, const double2 *in, int2 *out, long *carryOut, const double2 *weights) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  long carry = 0;
  out[id] = unweightAndCarry(base, 3, in[id], &carry, weights[id]);
  u32 size = gridDim.x * blockDim.x;
  u32 next = (id + 1 == size) ? 0 : (id + 1);
  carryOut[next] = carry;
}

KERNEL void carryB(uint base, const int2 *in, const long *carryIn, double2 *out, const double2 *weights) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  out[id] = carryAndWeight(base, in[id], carryIn[in], weights[id]);
}
*/

DEVICE double sq(double x) { return x * x; }

KERNEL void square(double2 *io) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
    u32 size = gridDim.x * blockDim.x;
    io[0].x    = sq(io[0].x);
    io[size].x = sq(io[size].x);
  } else {
    io[id]     = sq(io[id]);
  }
}

KERNEL void multiply(const double2 *in, double2 *io) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
    u32 size = gridDim.x * blockDim.x;
    io[0].x    = io[0].x * in[0].x;
    io[size].x = io[size].x * in[size].x;
  } else {
    io[id]     = mul(io[id], in[id]);
  }
}

//out[0]: in1==in2. out[1]: in1 != 0.
KERNEL void compare(const int2 *in1, const int2 *in2, int *out) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  int2 a = in1[id], b = in2[id];
  if (a.x != b.x || a.y != b.y) {
    out[0] = false;
  } else if (a.x != 0 || a.y != 0) {
    out[1] = true;
  }
}

static pair<bool, vector<int>> power2357(int n) {
  std::vector<int> v;
  for (int b : {2, 3, 5, 7}) {
    int k = 0;
    while (n % b == 0) { n /= b; ++k; }
    v.push_back(k);
  }
  return make_pair(n == 1, v);
}

class CudaGpu : public LowGpu<int *> {
  cufftHandle plan1, plan2;
  vector<int> goodData, goodCheck;
  double *bufA, *bufI;
  void *bufBig1, *bufBig2, *bufBig3;
  int *bufSmall;
  
protected:
  vector<int> readOut(int *&buf) {
    vector<int> ret(N);
    CC(cudaMemcpy(ret.data(), buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    return ret;
  }

  void writeIn(const vector<int> &data, int *&buf) {
    assert(data.size() == N);
    CC(cudaMemcpy(buf, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));    
  }

  void modSqLoop(int *&bufIn, int *&bufOut, int nIters, bool doMul3) {
    u32 baseBits = E / N;
    preWeight<<<N/4/256, 256>>>(baseBits, (int4 *) bufIn, (double4 *) bufA, (double4 *) bufBig1);

    for (int i = 0; i < nIters - 1; ++i) {
      CC(cufftExecD2Z(plan1, (double *) bufBig1, (double2 *) bufBig2));
      square<<<N/2/256, 256>>>((double2 *) bufBig2);
      CC(cufftExecZ2D(plan2, (double2 *) bufBig2, (double *) bufBig1));
      carryA<1><<<N/4/256, 256>>>(baseBits, (double4 *) bufBig1, (int4 *) bufOut, (long *) bufBig2, (double4 *) bufI);

      // log("%d : %016llx\n", i, bufResidue(bufOut, 0));
      
      carryB<<<N/4/256, 256>>>(baseBits, (int4 *) bufOut, (long *) bufBig2, (double4 *) bufBig1, (double4 *) bufA);
    }

    CC(cufftExecD2Z(plan1, (double *) bufBig1, (double2 *) bufBig2));
    square<<<N/2/256, 256>>>((double2 *) bufBig2);
    CC(cufftExecZ2D(plan2, (double2 *) bufBig2, (double *) bufBig1));

    /*
    double tmp[16];
    cudaMemcpy(tmp, bufBig1, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    log("%f %f %f %f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    */
    
    if (doMul3) {
      carryA<3><<<N/4/256, 256>>>(baseBits, (double4 *) bufBig1, (int4 *) bufOut, (long *) bufBig2, (double4 *) bufI);
    } else {
      carryA<1><<<N/4/256, 256>>>(baseBits, (double4 *) bufBig1, (int4 *) bufOut, (long *) bufBig2, (double4 *) bufI);
    }

    /*
    int tmp2[16];
    cudaMemcpy(tmp2, bufOut, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    log("%d %d %d %d\n", tmp2[0], tmp2[1], tmp2[2], tmp2[3]);
    */
    carryFinal<<<N/4/256, 256>>>(baseBits, (int4 *) bufOut, (double4 *) bufI, (long *) bufBig2);
  }

  void modMul(int *&bufIn, int *&bufIo, bool doMul3) {
    u32 baseBits = E / N;
    preWeight<<<N/4/256, 256>>>(baseBits, (int4 *) bufIn, (double4 *) bufA, (double4 *) bufBig1);
    CC(cufftExecD2Z(plan1, (double *) bufBig1, (double2 *) bufBig2));
    preWeight<<<N/4/256, 256>>>(baseBits, (int4 *) bufIo, (double4 *) bufA, (double4 *) bufBig1);
    CC(cufftExecD2Z(plan1, (double *) bufBig1, (double2 *) bufBig3));
    multiply<<<N/2/256, 256>>>((double2 *) bufBig3, (double2 *) bufBig2);
    CC(cufftExecZ2D(plan2, (double2 *) bufBig2, (double *) bufBig1));
    if (doMul3) {
      carryA<3><<<N/4/256, 256>>>(baseBits, (double4 *) bufBig1, (int4 *) bufIo, (long *)bufBig2, (double4 *) bufI);
    } else {
      carryA<1><<<N/4/256, 256>>>(baseBits, (double4 *) bufBig1, (int4 *) bufIo, (long *)bufBig2, (double4 *) bufI);
    }
    carryFinal<<<N/4/256, 256>>>(baseBits, (int4 *) bufIo, (double4 *) bufI, (long *) bufBig2);
  }

  bool equalNotZero(int *&buf1, int *&buf2, u32 deltaOffset) {
    assert(deltaOffset == 0);
    int data[2] = {true, false};
    CC(cudaMemcpyAsync(bufSmall, data, 2 * sizeof(int), cudaMemcpyHostToDevice, 0));
    compare<<<N/2/256, 256>>>((int2 *) buf1, (int2 *) buf2, (int *) bufSmall);
    CC(cudaMemcpy(data, bufSmall, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    bool equal = data[0];
    bool notZero = data[1];
    if (!equal) {
      u64 res1 = checkResidue();
      u64 res2 = bufResidue(bufAux, 0);
      log("check %d %d %016llx %016llx\n", (int)equal, (int)notZero, res1, res2);
      if (res1 == res2) {
        vector<int> check(N);
        vector<int> aux(N);
        cudaMemcpy(check.data(), bufCheck, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(aux.data(), bufAux, N * sizeof(int), cudaMemcpyDeviceToHost);
        int nDiff = 0;
        for (int i = 0; i < N; ++i) {
          if (check[i] != aux[i]) {
            ++nDiff;
            log("diff %d: %d %d\n", i, check[i], aux[i]);
            if (nDiff > 20) { break; }
          }
        }
      }
    }
    
    return equal && notZero;
  }
  
public:
  CudaGpu(u32 E, u32 N) :
    LowGpu(E, N)
  {
    assert(N % 1024 == 0);
    CC(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    
    {
      auto weights = genWeights(E, N);
      CC(cudaMalloc((void **)&bufA, N * sizeof(double)));
      CC(cudaMalloc((void **)&bufI, N * sizeof(double)));
      CC(cudaMalloc((void **)&bufSmall, 128 * sizeof(int)));
      CC(cudaMemcpy(bufA, weights.first.data(),  N * sizeof(double), cudaMemcpyHostToDevice));
      CC(cudaMemcpy(bufI, weights.second.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    CC(cufftPlan1d(&plan1, N, CUFFT_D2Z, 1));
    CC(cufftPlan1d(&plan2, N, CUFFT_Z2D, 1));
        
    CC(cudaMalloc((void **)&bufData,  N * sizeof(int)));
    CC(cudaMalloc((void **)&bufCheck, N * sizeof(int)));
    CC(cudaMalloc((void **)&bufAux,   N * sizeof(int)));
    CC(cudaMalloc(&bufBig1, (N + 1) * sizeof(double)));
    CC(cudaMalloc(&bufBig2, (N + 1) * sizeof(double)));
    CC(cudaMalloc(&bufBig3, (N + 1) * sizeof(double)));
    CC(cudaMalloc((void **)&bufSmall, 128 * sizeof(int)));
  }

  ~CudaGpu() {
    for (void *p : initializer_list<void *>{bufData, bufCheck, bufAux, bufA, bufI, bufBig1, bufBig2, bufBig3, bufSmall}) { CC(cudaFree(p)); }
    CC(cufftDestroy(plan1));
    CC(cufftDestroy(plan2));
  }

  static int round4096(int x) { return ((x - 1) / 4096 + 1) * 4096; }
  static int roundPow(int x, int step) {
    assert(x > 0 && (x % step == 0));
    while(!power2357(x).first) { x += step; }
    assert(x > 0);
    return x;
  }
  
  static u32 fftSize(u32 E, int argSize) {
    if (argSize > 20) { return roundPow(round4096(argSize), 4096); } // user forced FFT size.
    
    int fft = roundPow(round4096(E / 18.1f), 4096); // a rather conservative bits-per-word.
    while (argSize > 0) { fft = roundPow(fft + 4096, 4096); --argSize; }
    while (argSize < 0) { fft = roundPow(fft - 4096, -4096); ++argSize; }
    return fft;
  }
  
  static unique_ptr<Gpu> make(u32 E, Args &args) {
    u32 fft = fftSize(E, args.fftSize);
    vector<int> v = power2357(fft).second;
    log("Exponent %u using FFT %dK (2^%d * 3^%d * 5^%d * 7^%d)\n", E, fft/1024, v[0], v[1], v[2], v[3]);
    return unique_ptr<Gpu>(new CudaGpu(E, fft));
  }
  
  void finish() { cudaDeviceSynchronize(); }

  void commit() {
    assert(offsetData == 0 && offsetCheck == 0);
    goodData = readOut(bufData);
    goodCheck = readOut(bufCheck);
  }

  void rollback() {
    writeIn(goodData, bufData);
    writeIn(goodCheck, bufCheck);
  }

  u64 bufResidue(int *&buf, u32 offset) {
    assert(offset == 0);
    // readResidue<<<1, 64>>>(N/2, (int2 *) bufData, (int2 *) bufSmall);
    vector<int> words(128);
    CC(cudaMemcpyAsync(words.data(), buf + (N - 64), 64 * sizeof(int), cudaMemcpyDeviceToHost));
    CC(cudaMemcpy(words.data() + 64, buf, 64 * sizeof(int), cudaMemcpyDeviceToHost));
    return residueFromRaw(words, 0);
  }
};
