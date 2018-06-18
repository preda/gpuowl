// cudaOwl, a CUDA PRP Mersenne primality tester.
// Copyright (C) 2017-2018 Mihai Preda.

#define DEVICE static __device__
#define GLOBAL static __global__
#define DUAL static __device__ __host__

#include "checkpoint.h"
#include "common.h"

#include <cufft.h>
#include <cstdio>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

/*
namespace std { template<> struct default_delete<FILE> { void operator()(FILE *f) { if (f != nullptr) { fclose(f); } } }; }
unique_ptr<FILE> open(const string &name, const char *mode) { return unique_ptr<FILE>(fopen(name.c_str(), mode)); }
*/

u32 step(u32 N, u32 E) { return N - (E % N); }
u32 extra(u32 N, u32 E, u32 k) { return u64(step(N, E)) * k % N; }
bool isBigWord(u32 N, u32 E, u32 k) { return extra(N, E, k) + step(N, E) < N; }
u32 bitlen(u32 N, u32 E, u32 k) { return E / N + isBigWord(N, E, k); }

void genWeights(int E, int N, double *aTab, double *iTab) {
  double *pa = aTab;
  double *pi = iTab;

  int baseBits = E / N;
  auto iN = 1 / (long double) N;

  for (int k = 0; k < N; ++k) {
    int bits = bitlen(N, E, k);
    assert(bits == baseBits || bits == baseBits + 1);
    auto a  = exp2l(extra(N, E, k) * iN);
    auto ia = 1 / (4 * N * a);
    *pa++ = (bits == baseBits) ? a  : -a;
    *pi++ = (bits == baseBits) ? ia : -ia;
  }
}

DEVICE int carryStep(long x, long *carry, int bits) {
  x += *carry;
  int w = lowBits(x, bits);
  *carry = (x - w) >> bits;
  return w;
}

DEVICE uint signBit(double a) { return __double_as_longlong(a) >> 63; }
DEVICE uint bitlen(uint base, double a) { return base + signBit(a); }
DEVICE long unweight(double x, double w) { return __double2ll_rn(x * fabs(w)); }

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

GLOBAL void carryA(uint base, const double2 *in, int2 *out, long *carryOut, const double2 *weights) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  long carry = 0;
  out[id] = unweightAndCarry(base, 1, in[id], &carry, weights[id]);
  carryOut[id] = carry;
}

GLOBAL void carryB(uint base, const int2 *in, const long *carryIn, double2 *out, const double2 *weights) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  u32 size = gridDim.x * blockDim.x;
  u32 prev = id ? (id - 1) : (size - 1);
  out[id] = carryAndWeight(base, in[id], carryIn[prev], weights[id]);
}

vector<string> stringArgs(int argc, char **argv) {
  vector<string> out;
  for (int i = 1; i < argc; ++i) { out.push_back(string(argv[i])); }
  return out;
}

vector<int> getSizeTable() {
  if (auto fi = open("ffts.txt", "r")) {
    vector<int> ret;
    int s = 0;
    char c;
    while (fscanf(fi.get(), "%d%c", &s, &c)  == 2) {
      int mul = c == 'M' ? 1024 * 1024 : c == 'K' ? 1024 : 1;
      ret.push_back(s * mul);
    }
    return ret;
  } else {
    return {2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024};
  }
}

u32 defaultFFTSize(u32 E, int step) {
  const float maxBitsPerWord = E <= 77600000 ? 18.5f : E <= 153000000 ? 18.25f : 18;
  const int targetFFT = int(E / maxBitsPerWord + 0.5f);
  vector<int> sizes = getSizeTable();
  assert(!sizes.empty() && targetFFT <= sizes.back());
  int i = 0;
  while (sizes[i] < targetFFT) { ++i; }
  assert(i < sizes.size() && targetFFT <= sizes[i]);
  return sizes[min(max(0, i + step), int(sizes.size() - 1))];
}

u32 getFFTSize(u32 E, const string &userSizeStr) {
  if (userSizeStr.size() < 1) { return defaultFFTSize(E, 0); }
  if (userSizeStr[0] == '-' || userSizeStr[0] == '+') {
    return defaultFFTSize(E, atoi(userSizeStr.c_str()));
  }
  int unit = (userSizeStr.back() == 'K') ? 1024 : (userSizeStr.back() == 'M') ? 1024*1024 : 1;
  return atoi(userSizeStr.c_str()) * unit;
}

int main(int argc, char **argv) {
  auto args = stringArgs(argc, argv);
  if (args.size() == 0 || args.size() > 2 || (args.size() >= 1 && (args[0] == "-h" || args[0] == "--help"))) {
    printf(R"""(Usage: cudaOwl <exponent> [<FFT-size>]
Examples:
cudaOwl 85000001        : use the default FFT size for the exponent.
cudaOwl 85000001 5000K  : use 5000K FFT.
cudaOwl 85000001 -1     : use FFT size one-step-down from default.
)""");
    return 0;
  }

  const u32 E = atoi(args[0].c_str());
  const u32 fftSize = getFFTSize(E, (args.size() >= 2) ? args[1] : "");
  printf("Exponent %d, FFT %dK, %.2f bits-per-word\n", E, fftSize / 1024, E / float(fftSize));
}
