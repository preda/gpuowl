// Copyright Mihai Preda

#include "B1Accumulator.h"

#include "Gpu.h"
#include "Memlock.h"
#include "GmpUtil.h"
#include "Args.h"
#include "checkpoint.h"

#include <tuple>

B1Accumulator::B1Accumulator(Gpu* gpu, Saver* saver, u32 E)
  : E{E}, b1{saver->b1}, nBits{powerSmoothBits(E, b1)}, gpu{gpu}, saver{saver}, N{gpu->getFFTSize()} {
  log("P1(%s) %u bits\n", formatBound(b1).c_str(), nBits);
}

void B1Accumulator::alloc() {
    if (bits.empty()) {
      // Timer timer;
      bits = powerSmoothLSB(E, b1);
      // log("powerSmooth(%u), %u bits, took %.2fs (CPU)\n", b1, u32(bits.size()), timer.elapsedSecs());
    }

    if (bufs.empty()) {      
      assert(!memlock);
      memlock.emplace(gpu->args.masterDir, u32(gpu->args.device));

      u32 nBufs = AllocTrac::availableBytes() / (N * sizeof(i32)) - 5;
      assert(nBufs >= 16);

      log("P1(%s) using %u buffers\n", formatBound(b1).c_str(), nBufs);
      bufs = gpu->makeBufVector(nBufs);
    }

    for (Buffer<int>& buf : bufs) { buf.set(1); }
}

void B1Accumulator::release() {
  if (!bits.empty()) { bits.clear(); }
  
  if (!bufs.empty()) {
    assert(memlock);
    log("P1(%s) releasing %u buffers\n", formatBound(b1).c_str(), u32(bufs.size()));
    bufs.clear();
    memlock.reset();
  }
}

vector<u32> B1Accumulator::fold() {
  if (!b1 || bufs.empty()) { return {}; }
  
  assert(b1);
  assert(!bufs.empty());  
  return gpu->fold(bufs);
}

pair<u32,u32> B1Accumulator::findFirstPop(u32 start) {
  u32 sum = 0;
  for (u32 i = start; i < bits.size(); ++i) {
    if (bits[i]) {
      u32 n = i - start;
      assert(n < 64);
      u64 delta = u64(1) << n;
      if (sum + delta >= bufs.size()) {
        return {i, sum};
      } else {
        sum += delta;
      }
    }
  }
  return {0, sum};
}

vector<u32> B1Accumulator::save(u32 k) {
    if (!bufs.empty()) {
      // assert(k < nBits + 2000); // the Jacobi check may delay the save() a lot such that this assert doesn't hold.
      
      vector<u32> data = fold();

      if (data.empty()) {
        throw Reload{};
      }

      saver->saveP1(k, {nextK, data});
      if (nextK == 0) {
        saver->saveP1Final(data);
        release();
      }
      return data;
    }
    return {};
}

u32 B1Accumulator::findFirstBitSet() const {
  assert(!bits.empty());
  for (u32 i = 0, end = bits.size(); i < end; ++i) { if (bits[i]) { return i; } }
  assert(false);
  return 0;
}

void B1Accumulator::verifyRoundtrip(const Words& expected) {
  auto dataBack = fold();
  if (dataBack.empty()) {
    throw "P1 fold ZERO";
  }
  
  assert(dataBack.size() == expected.size());

  if (dataBack != expected) {
    for (u32 i = 0, nErr = 0; i < expected.size() && nErr < 20; ++i) {
      if (dataBack[i] != expected[i]) {
        ++nErr;
        log("[%u] %08x != %08x\n", i, dataBack[i], expected[i]);
      }
    }
    
    log("fold() does not roundtrip\n");
    throw "fold roundtrip";
  }
}

namespace {

Words randomWords(u32 E, u32 seed) {
  assert(E % 32);
  u32 n = (E - 1) / 32 + 1;
  Words words(n);
  for (u32 i = 0; i < n - 1; ++i) {
    seed *= 4294967291;
    words[i] = seed;
  }
  seed *= 4294967291;
  words[n-1] = seed & ((1 << (E % 32)) - 1);
  return words;
}

}

void B1Accumulator::load(u32 k) {
  if (!b1 || k >= nBits) {
    release();
    nextK = 0;
    return;
  }

  Words data;
  
  if (k == 0) {
    alloc();

    // Timer timer;
    for (u32 i = 0; i < 5; ++i) {
      data = randomWords(E, (i + 1));
      gpu->writeIn(bufs[0], data);
      verifyRoundtrip(data);
    }
    // log("fold() check took %.1fs\n", timer.elapsedSecs());
    
    nextK = findFirstBitSet();
    data = makeWords(E, 1);
  } else {
    std::tie(nextK, data) = saver->loadP1(k);    
  }

  if (!nextK) {
    release();
    return;
  }
  
  alloc();
  gpu->writeIn(bufs[0], data);
  verifyRoundtrip(data);
}

template<typename T>
void B1Accumulator::step(u32 kAt, Buffer<T>& data) {
  assert(nextK && kAt == nextK);
  assert(nextK < bits.size() && bits[nextK]);
  
  auto [nextPop, sum] = findFirstPop(nextK + 1);
  nextK = nextPop;
  assert(nextK == 0 || bits[nextK]);
  
  gpu->mul(bufs[sum], data);
}

template void B1Accumulator::step<int>(u32 kAt, Buffer<int>& data);
template void B1Accumulator::step<double>(u32 kAt, Buffer<double>& data);
