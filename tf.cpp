// Copyright (C) 2017-2018 Mihai Preda.

#include "kernel.h"
#include "timeutil.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include <vector>
#include <bitset>

typedef unsigned u32;
typedef unsigned long long u64;

using namespace std;

struct Test {
  u32 exp;
  u64 k;
};

Test tests[] = {
#include "selftest.h"
};
  
// q := 2*exp*c + 1. Is q==1 or q==7 (mod 8)?
bool q1or7mod8(uint exp, uint c) { return !(c & 3) || ((c & 3) + (exp & 3) == 4); }

template<u32 P> bool multiple(u32 exp, u32 c) { return 2 * c * u64(exp) % P == P - 1; }
// { return (((uint) ((exp * (ulong)c) % p)) + 1) % p == 0; }

bool isGoodClass(u32 exp, u32 c) {
  return q1or7mod8(exp, c)
    && !multiple<3>(exp, c)
    && !multiple<5>(exp, c)
    && !multiple<7>(exp, c)
    && !multiple<11>(exp, c);
}

constexpr const u32 NCLASS = (4 * 3 * 5 * 7 * 11);
constexpr const u32 NGOOD  = (2 * 2 * 4 * 6 * 10);

vector<u32> goodClasses(u32 exp) {
  vector<u32> good;
  good.reserve(NGOOD);
  for (u32 c = 0; c < NCLASS; ++c) {
    if (isGoodClass(exp, c)) { good.push_back(c); }
  }
  assert(good.size() == NGOOD);
  return good;
}

// Returns all the primes < 2*N.
template<u32 N> vector<u32> smallPrimes(u32 start) {
  vector<u32> primes;
  if (N < 1) { return primes; }
  if (2 >= start) { primes.push_back(2); }
  u32 limit = sqrt(N);
  bitset<N> notPrime;
  notPrime[0] = true;
  u32 last = 0;
  while (true) {
    u32 p = last + 1;
    while (p < N && notPrime[p]) { ++p; }
    if (p >= N) { return primes; }
    last = p;
    notPrime[p] = true;
    u32 prime = 2 * p + 1;
    if (prime >= start) { primes.push_back(prime); }
    if (p <= limit) { for (u32 i = 2 * p * (p + 1); i < N; i += prime) { notPrime[i] = true; } }
  }
}

// 1/n modulo prime
u32 modInv(u32 n, u32 prime) {
  u32 q = prime / n;
  u32 d = prime - q * n;
  int x = -q;
  int prevX = 1;
  while (d) {
    q = n / d;
    { u32 save = d; d = n - q * d; n = save; }           // n = set(d, n - q * d);
    { int save = x; x = prevX - q * x; prevX = save; }   // prevX = set(x, prevX - q * x);
  }
  u32 ret = (prevX >= 0) ? prevX : (prevX + prime);
  assert(ret < prime);
  return ret;
}

vector<u32> initModInv(u32 exp, const vector<u32> &primes) {
  vector<u32> invs;
  invs.reserve(primes.size());
  for (u32 prime : primes) {
    // assert(prime < u32(-1) / NCLASS); // to prevent 32bit overflow in mul below.
    invs.push_back(modInv(2 * NCLASS * u64(exp) % prime, prime));
  }
  return invs;
}

typedef __uint128_t u128;

vector<u32> initBtc(u32 exp, u64 k, const vector<u32> &primes, const vector<u32> &invs) {
  vector<u32> btcs;
  for (auto primeIt = primes.begin(), invIt = invs.begin(), primeEnd = primes.end(); primeIt != primeEnd; ++primeIt, ++invIt) {
    u32 prime = *primeIt;
    u32 inv   = *invIt;
    u32 qMod = (2 * exp * (k % prime) + 1) % prime;
    u32 btc = (prime - qMod) * u64(inv) % prime;
    assert(btc < prime);
    // assert(2 * exp * u128(k + u64(btc) * NCLASS) % prime == prime - 1);
    btcs.push_back(btc);
  }
  return btcs;
}

u64 startK(u32 exp, float bits) {
  u64 k = exp2(bits - 1) / exp;
  // ((__uint128_t(1) << (bits - 1)) + (exp - 2)) / exp;
  return k - k % NCLASS;
}

float bitLevel(u32 exp, u64 k) { return log2f(float(exp) * float(k) * 2 + 1); }

void print(const vector<u32> &v) {
  for (u32 x : v) { printf("%u, ", x); }
  printf("\n(size %d)\n", int(v.size()));
}

int main(int argc, char **argv) {
  initLog("tf.log");
  
  u32 exp = atoi(argv[1]);
  float startBit = atof(argv[2]);
  
  auto primes = smallPrimes<1024 * 1024 * 8>(13);

  printf("%d primes up to %d\n", int(primes.size()), primes[primes.size() - 1]);

  vector<u32> primeInvs;
  primeInvs.reserve(primes.size());
  for (u32 p : primes) { primeInvs.push_back(u32(-1) / p); }
  
  auto invs = initModInv(exp, primes);
  auto classes = goodClasses(exp);
  assert(classes[0] == 0);
      
  auto devices = getDeviceIDs(true);
  cl_device_id device = devices[0];
  Context context(createContext(device));
  vector<string> defines;
  string clArgs = "";
  // if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/"+tf; }
  clArgs += " -cl-std=CL2.0 -save-temps=t0/tf";
  cl_program program = compile(device, context.get(), "tf", clArgs, defines);
  Queue queue(makeQueue(device, context.get()));

  const int SIEVE_GROUPS = 1024;
  const int BITS_PER_GROUP = 32 * 1024 * 8;
  const int BITS_PER_SIEVE = SIEVE_GROUPS * BITS_PER_GROUP; // 2^28, 256M
  const u64 BITS_PER_CYCLE = BITS_PER_SIEVE * u64(NCLASS); // 2 ^ 40.2
  
  Kernel sieve(program, queue.get(), device, SIEVE_GROUPS, "sieve", false);
  Kernel tf(program, queue.get(), device, 512, "tf", false);
  Kernel initBtc(program, queue.get(), device, 1024, "initBtc", false);

  Buffer bufPrimes(makeBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
                           sizeof(u32) * primes.size(), primes.data()));

  Buffer bufModInvs(makeBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
                           sizeof(u32) * invs.size(), invs.data()));

  Buffer bufInvs(makeBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
                           sizeof(u32) * primeInvs.size(), primeInvs.data()));
  
  Buffer bufBtc(makeBuf(context, CL_MEM_READ_WRITE, sizeof(u32) * primes.size()));


  Buffer bufK(makeBuf(context, CL_MEM_READ_WRITE, BITS_PER_SIEVE * sizeof(u32)));

  Buffer bufN(makeBuf(context, CL_MEM_READ_WRITE, sizeof(u32)));

  /*
  u64 k0 = startK(exp, startBit);
  auto btcs = initBtc(exp, k0, primes, invs);
  queue.write(bufBtc, btcs);
  queue.zero(bufN, sizeof(u32));
  */

  Timer bigTime;
  
  for(u64 k0 = startK(exp, startBit); ; k0 += BITS_PER_CYCLE) {
    log("Starting bitlevel %.4f\n", bitLevel(exp, k0));

    for (int i = 0; i < NGOOD; ++i) {
      int c = classes[i];
      u64 k = k0 + c;
      
      initBtc((u32) primes.size(), exp, k, bufPrimes, bufModInvs, bufBtc);

      queue.zero(bufN, sizeof(u32));
      
      sieve(bufPrimes, bufInvs, bufBtc, bufN, bufK);
      // log("after sieve\n");
      
      tf(exp, k, bufN, bufK);
      // log("after tf\n");
      
      i32 readN;
      read(queue.get(), false, bufN, sizeof(i32), &readN);
      queue.zero(bufN, sizeof(i32));

      /*
      u64 nextK = k0 + ((i == NGOOD - 1) ? BITS_PER_CYCLE : classes[i + 1]);
      // log("before initBtc\n");
      Timer t;
      btcs = initBtc(exp, nextK, primes, invs);
      u64 delta1 = t.deltaMicros();
      // log("after initBtc\n");
      queue.write(bufBtc, btcs);
      u64 delta2 = t.deltaMicros();
      */
      queue.finish();
      int delta1 = 0, delta2 = 0;
      log("bit %.4f, k %llu, class %3d / %3d (%4d), %d, time %d %d %d\n", bitLevel(exp, k0), k, i, NGOOD, c, readN, int(delta1), int(delta2), int(bigTime.deltaMicros()));

      if (readN < 0) {
        log("Found factor K: %llu\n", (k + (-readN) * u64(NCLASS)));
        return 1;
      }
    }
  }

  /*
  
  // StatsInfo stats = sieve.resetStats(); printf("%d %f\n", stats.n, stats.mean);

  
  u64 data[2 * 256] = {0};
  u64 *p = data;
  for (int i = 0; i < 256; ++i) {
    *p++ = 332196173;
    *p++ = 15086682666063ull;
  }
  
  Buffer bufSquare(makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(u32) * 4 * 256, data));

  for (auto t : tests) {    

    for (int i = 0; i < 256; ++i) {
      data[i*2] = t.exp;
      data[i*2 + 1] = t.k;
    }

    write(queue.get(), true, bufSquare, sizeof(u32) * 4 * 256, data);
    
    // square(bufSquare);
    square(bufSquare);

    vector<u32> out = queue.read<u32>(bufSquare, 4 * 256);
    int i;
    for (i = 0; i < 256; ++i) {
      if (!out[i*4]) {
        // printf("%u %d\n", t.exp, i);
        break;
      }
      // printf("%d %x'%08x'%08x\n", i, out[i*4+2], out[i*4+1], out[i*4]);
    }
    if (i == 256) {      
      printf("ok %u\n", t.exp);
    } else {
      printf("fail %u\n", t.exp);
    }
  }

  return 0;
  */


  // for (int i = 0; i < 512; ++i) { printf("%d %u\n", i, primes[i]); }
}
