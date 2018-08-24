// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "TF.h"

#include "kernel.h"
#include "timeutil.h"
#include "checkpoint.h"
#include "args.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <bitset>

cl_device_id getDevice(const Args &args);
void logTimeKernels(std::initializer_list<Kernel *> kerns);

// q := 2*exp*c + 1. Is q==1 or q==7 (mod 8)?
bool q1or7mod8(u32 exp, u32 c) { return !(c & 3) || ((c & 3) + (exp & 3) == 4); }

template<u32 P> bool multiple(u32 exp, u32 c) { return 2 * c * u64(exp) % P == P - 1; }

bool isGoodClass(u32 exp, u32 c) {
  return q1or7mod8(exp, c)
    && !multiple<3>(exp, c)
    && !multiple<5>(exp, c)
    && !multiple<7>(exp, c)
    && !multiple<11>(exp, c)
    && !multiple<13>(exp, c);
}

constexpr const u32 NCLASS = (4 * 3 * 5 * 7 * 11 * 13); // 60060
constexpr const u32 NGOOD  = (2 * 2 * 4 * 6 * 10 * 12); // 11520

vector<u32> goodClasses(u32 exp) {
  vector<u32> good;
  good.reserve(NGOOD);
  for (u32 c = 0; c < NCLASS; ++c) { if (isGoodClass(exp, c)) { good.push_back(c); } }
  assert(good.size() == NGOOD);
  assert(good[0] == 0);
  return good;
}

// Returns all the primes p such that: p >= start and p < 2*N; at most maxSize primes.
template<u32 N> vector<u32> smallPrimes(u32 start, u32 maxSize) {
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
    if (prime >= start) {
      primes.push_back(prime);
      if (primes.size() >= maxSize) { return primes; }
    }
    if (p <= limit) { for (u32 i = 2 * p * (p + 1); i < N; i += prime) { notPrime[i] = true; } }
  }
}

// 1/n modulo prime
u32 modInv(u32 n, u32 prime) {
  const u32 saveN = n;
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
  
  assert(ret < prime && ret * (u64) saveN % prime == 1);
  
  return ret;
}

vector<u32> initModInv(u32 exp, const vector<u32> &primes) {
  vector<u32> invs;
  invs.reserve(primes.size());
  for (u32 prime : primes) { invs.push_back(modInv(2 * NCLASS * u64(exp) % prime, prime)); }
  return invs;
}

u64 startK(u32 exp, int bits) {
  u64 k = exp2(bits - 1) / exp;
  return k - k % NCLASS;
}

double bitLevel(u32 exp, u64 k) { return log2(2 * exp * double(k) + 1); }

constexpr const int SIEVE_GROUPS = 4 * 1024;
constexpr const int LDS_WORDS = 8 * 1024;
constexpr const int BITS_PER_GROUP = 32 * LDS_WORDS;
constexpr const int BITS_PER_SIEVE = SIEVE_GROUPS * BITS_PER_GROUP;
constexpr const u64 BITS_PER_CYCLE = BITS_PER_SIEVE * u64(NCLASS);

constexpr const int SPECIAL_PRIMES = 32;
constexpr const int NPRIMES = 320 * 1024 + SPECIAL_PRIMES;

constexpr const u32 KBUF_BYTES = BITS_PER_SIEVE / 5 * sizeof(u32);

vector<u32> getPrimeInvs(const std::vector<u32> &primes) {
  std::vector<u32> v;
  v.reserve(primes.size());
  for (u32 p : primes) { v.push_back(u32(-1) / p); }
  return v;
}

vector<u32> getSteps(const vector<u32> &primes) {
  std::vector<u32> v;
  v.reserve(primes.size());
  for (u32 p : primes) { v.push_back(BITS_PER_SIEVE % p); }
  return v;
}

// Convert number of K candidates to GHzDays. See primenet_ghzdays() in mfakto output.c
float ghzDays(u64 ks) { return ks * (0.016968 * 1680 / (1ull << 46)); }

// Speed, in GHz == GHzDays / days.
float ghz(u64 ks, float secs) { return 24 * 3600 * ghzDays(ks) / secs; }

class OpenTF : public TF {
  vector<u32> primes;
  Queue queue;

  Kernel sieve, tf, initBtc, stepBtc;
  Buffer bufPrimes, bufInvs, bufSteps, bufModInvs, bufBtc, bufK, bufN, bufFound, bufTotal;
  
public:
  static unique_ptr<TF> make(Args &args) {

    string clArgs = args.clArgs;
    if (!args.dump.empty()) { clArgs += " -save-temps=" + args.dump + "/tf"; }

    bool timeKernels = args.timeKernels;
    
    cl_device_id device = getDevice(args);
    if (!device) { throw "No OpenCL device"; }
    
    log("%s\n", getLongInfo(device).c_str());
    if (args.cpu.empty()) { args.cpu = getShortInfo(device); }

    Context context(createContext(device));
#define DEF(name) {#name, name}
    Holder<cl_program> program(compile(device, context.get(), "tf", clArgs,
                                       {DEF(NCLASS), DEF(SPECIAL_PRIMES), DEF(NPRIMES), DEF(LDS_WORDS)},
                                       args.usePrecompiled));
#undef DEF
    
    if (!program) { throw "OpenCL compilation"; }
    
    return std::make_unique<OpenTF>(program.get(), device, context.get(), timeKernels);
  }

  OpenTF(cl_program program, cl_device_id device, cl_context context, bool timeKernels) :
    primes(smallPrimes<1024 * 1024 * 4>(17, NPRIMES)),
    queue(makeQueue(device, context)),

#define LOAD(name, workGroups) name(program, queue.get(), device, workGroups, #name, timeKernels)
    LOAD(sieve, SIEVE_GROUPS),
    LOAD(tf, 4096),
    LOAD(initBtc, 256),
    LOAD(stepBtc, 256),
#undef LOAD

    bufPrimes(makeBuf( context, BUF_CONST, sizeof(u32) * NPRIMES, primes.data())),    
    bufInvs(makeBuf(   context, BUF_CONST, sizeof(u32) * NPRIMES, getPrimeInvs(primes).data())),
    bufSteps(makeBuf(  context, BUF_CONST, sizeof(u32) * NPRIMES, getSteps(primes).data())),
    bufModInvs(makeBuf(context, CL_MEM_READ_WRITE,    sizeof(u32) * NPRIMES)),
    
    bufBtc(makeBuf(context, BUF_RW, sizeof(u32) * primes.size())),
    bufK(makeBuf(context, BUF_RW, KBUF_BYTES)),
    bufN(makeBuf(context, CL_MEM_READ_WRITE, sizeof(u32))),
    bufFound(makeBuf(context, CL_MEM_READ_WRITE, sizeof(u64))),
    bufTotal(makeBuf(context, CL_MEM_READ_WRITE, sizeof(u64)))
  {
    assert(primes.size() == NPRIMES);
    // log("Sieve: allocating %.1f MB of GPU memory\n", KBUF_BYTES / float(1024 * 1024));
    long double f = 1;
    for (u32 p : primes) { f *= (p - 1) / (double) p; }
    log("Sieve with %d primes (up to %d), expected %.4f%%\n", NPRIMES, primes[NPRIMES - 1], double(f) * 100); 
  }
    
  u64 findFactor(u32 exp, int bitLo, int bitEnd, int nDone, int nTotal, u64 *outBeginK, u64 *outEndK, bool timeKernels) {
    assert(nDone == 0 || nTotal == NGOOD);
    assert(bitLo < bitEnd);
    
    auto classes = goodClasses(exp);
    queue.write(bufModInvs, initModInv(exp, primes));
    
    u64 k0   = startK(exp, bitLo);
    u64 kEnd = startK(exp, bitEnd);

    u32 nSieveGroups = ((kEnd - k0) / NCLASS + (BITS_PER_GROUP - 1)) / BITS_PER_GROUP;
    *outBeginK = k0;
    *outEndK   = k0 + BITS_PER_GROUP * u64(NCLASS) * nSieveGroups;
    
    log("TF %u %d-%d, K %llu - %llu, %dx%d + 1x%d groups, start from class #%d\n",
        exp, bitLo, bitEnd, k0, *outEndK,
        nSieveGroups / SIEVE_GROUPS, SIEVE_GROUPS, nSieveGroups % SIEVE_GROUPS, nDone);
    // exp >> (24 - __builtin_clz(exp)));

    sieve.setFixedArgs(0, bufPrimes, bufInvs, bufBtc, bufN, bufK);
    stepBtc.setFixedArgs(1, bufPrimes, bufSteps, bufBtc, bufN, bufTotal);

    queue.zero(bufFound, sizeof(u64));
    queue.zero(bufN, sizeof(u32));
    queue.zero(bufTotal, sizeof(u64));

    Timer timer;
    for (u32 i = nDone; i < NGOOD; ++i) {
      int c = classes[i];
      u64 k = k0 + c;
      initBtc(NPRIMES, exp, k, bufPrimes, bufModInvs, bufBtc);
      for (int groupsLeft = nSieveGroups; groupsLeft > 0; groupsLeft -= SIEVE_GROUPS, k += BITS_PER_CYCLE) {
        sieve.run(min(groupsLeft, SIEVE_GROUPS));
        tf(bufN, exp, k, bufK, bufFound);
        stepBtc((groupsLeft <= SIEVE_GROUPS) ? 0 : NPRIMES); // , bufPrimes, bufSteps, bufBtc, bufN, bufTotal);
      }
      u64 nFiltered = 0; // queue.read<u64>(bufTotal, 1)[0];
      read(queue.get(), false, bufTotal, sizeof(u64), &nFiltered);
      queue.zero(bufTotal, sizeof(u64));
      u64 foundK = queue.read<u64>(bufFound, 1)[0];
        
      float secs = timer.deltaMicros() / 1000000.0f;
      float speed = ghz(BITS_PER_GROUP * u64(NCLASS) * nSieveGroups / NGOOD, secs);
      int etaMins = int((NGOOD - (i + 1)) * secs / 60 + .5f);
      int days  = etaMins / (24 * 60);
      int hours = etaMins / 60 % 24;
      int mins  = etaMins % 60;
          
      log("TF %u %d-%d %.2f%%, class %4d (%4d), %.3fs (%.0f GHz), ETA %dd %02d:%02d, FCs %llu (%.4f%%)\n",
          exp, bitLo, bitEnd, (i + 1) * 100.0f / NGOOD,
          i, c,
          secs, speed,
          days, hours, mins,
          nFiltered, nFiltered / (float(BITS_PER_GROUP) * nSieveGroups) * 100);

      if (timeKernels) { logTimeKernels({&sieve, &tf, &stepBtc, &initBtc}); }

      if (foundK) { return foundK; }
      
      Checkpoint::saveTF(exp, bitLo, bitEnd, i + 1, NGOOD);
    }
    return 0;
  }
};
