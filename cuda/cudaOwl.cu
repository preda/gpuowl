#include <cufft.h>

#include <cstdio>
#include <cassert>
#include <memory>

#include <string>
#include <vector>

typedef unsigned char byte;
typedef long long i64;
typedef unsigned long long u64;
typedef int      i32;
typedef unsigned u32;
typedef unsigned __int128 u128;

using namespace std; // std::string, std::pair, std::vector, std::unique_ptr;

namespace std { template<> struct default_delete<FILE> { void operator()(FILE *f) { if (f != nullptr) { fclose(f); } } }; }

unique_ptr<FILE> open(const string &name, const char *mode) { return unique_ptr<FILE>(fopen(name.c_str(), mode)); }

int extra(unsigned N, unsigned E, unsigned k) {
  assert(E % N);
  u32 step = N - (E % N);
  return u64(step) * k % N;
}

bool isBigWord(unsigned N, unsigned E, unsigned k) {
  u32 step = N - (E % N); 
  return extra(N, E, k) + step < N;
  // return extra(N, E, k) < extra(N, E, k + 1);
}

int bitlen(int N, int E, int k) { return E / N + isBigWord(N, E, k); }

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

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

#define TAU (2 * M_PIl)

double2 root1(u32 k, u32 N) { return double2{double(cosl(- TAU / N * k)), double(sinl(- TAU / N * k))}; }

double2 *trig(double2 *p, int n, int B) {
  for (int i = 0; i < n; ++i) { *p++ = root1(i, B); }
  return p;
}

vector<double2> genSquareTrig(int N) {
  vector<double2> tab(N / 4);
  trig(tab.data(), N / 4, N);
  return tab;
}  
  /*
  const int size = H / 2 + W;
  vector<double2> tab(size);
  double2 *end = tab.data();
  end = trig(end, H / 2, H * 2);
  end = trig(end, W, W * H * 2);
  assert(end - tab.data() == size);
  return tab;
  */

#define DEVICE static __device__
#define GLOBAL static __global__

DEVICE int lowBits(int u, int bits) { return (u << (32 - bits)) >> (32 - bits); }

DEVICE int carryStep(long x, long *carry, int bits) {
  x += *carry;
  int w = lowBits(x, bits);
  *carry = (x - w) >> bits;
  return w;
}

DEVICE uint signBit(double a) { return __double_as_longlong(a) >> 63; }
DEVICE uint bitlen(uint baseLen, double a) { return baseLen + signBit(a); }
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

DEVICE double2 conjugate(double2 u) { return double2{u.x, -u.y}; }
DEVICE double2 swap(double2 u) { return double2{u.y, u.x}; }
DEVICE double2 mul(double2 a, double2 b) { return double2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
DEVICE double2 mul(double2 a, uint k) { return double2{a.x * k, a.y * k}; }
DEVICE double2 sq(double2 a) { return double2{(a.x + a.y) * (a.x - a.y), 2 * a.x * a.y}; }
DEVICE double2 addsub(double2 a) { return double2{a.x + a.y, a.x - a.y}; }
DEVICE double2 foo2(double2 a, double2 b) { a = addsub(a); b = addsub(b); return addsub(double2{a.x * b.x, a.y * b.y}); }
DEVICE double2 foo(double2 a) { return foo2(a, a); }

GLOBAL void square(double2 *io, const double2 *bigTrig) {
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  u32 size = gridDim.x * blockDim.x;
  if (id == 0) {
    io[0] = mul(foo(conjugate(io[0])), 4);
    io[size / 2] = mul(sq(conjugate(io[size / 2])), 8);
    return;
  }

  
}

int main(int argc, char **argv) {
}

/*
pair<int, string> worktodoReadLine() {
  if (auto fi = open("worktodo.txt", "r")) {
    char line[256];
    while (fgets(line, sizeof(line), fi.get())) {
      char aid[33] = {0};
      int exp;
      if ((sscanf(line, "%d", &exp) == 1) ||
          (sscanf(line, "PRP=%32[0-9a-fA-F],%*d,%*d,%d", aid, &exp) == 2)) {
        return {exp, aid};
      }
    }
  }
  return {0, ""};
}

bool worktodoDelete(int E) {
  
}
*/
