// Copyright Mihai Preda

#include "B1Accumulator.h"

#include "Gpu.h"
#include "Memlock.h"
#include "GmpUtil.h"
#include "Args.h"
#include "checkpoint.h"
#include "util.h"

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
    bufs.clear();
    log("P1(%s) released %u buffers\n", formatBound(b1).c_str(), u32(bufs.size()));
    memlock.reset();
  }
}

vector<u32> B1Accumulator::fold() {
  if (b1 == 0) { return {}; }
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
      assert(k < nBits + 1000);
      
      vector<u32> data = fold();
      // log("B1 at %u res64 %016lx\n", k, residue(data));

      saver->saveP1(k, {nextK, data});
      if (nextK == 0) {
        saver->saveP1Final(data);
        release();
        return data;
      }
    }
    return {};
}

u32 B1Accumulator::findFirstBitSet() const {
  assert(!bits.empty());
  for (u32 i = 0, end = bits.size(); i < end; ++i) { if (bits[i]) { return i; } }
  assert(false);
  return 0;
}

void B1Accumulator::load(u32 k) {
    if (!b1 || k >= nBits) {
      release();
      nextK = 0;
      return;
    }

    if (k == 0) {
      alloc();
      auto data = fold();
      if (data.empty()) { throw "P1 fold ZERO"; }
      // log("%u %u %u %u\n", data[0], data[1], data[2], data[3]); 
      assert(data[0] == 1 && data[1] == 0 && data[2] == 0 && data[3] == 0);
      nextK = findFirstBitSet();
      log("P1(%s) starting\n", formatBound(b1).c_str());
      return;
    }

    auto [loadNextK, data] = saver->loadP1(k);
    nextK = loadNextK;
    if (!nextK) {
      release();
      return;
    }

    alloc();
    gpu->writeIn(bufs[0], data);

    assert(fold() == data);
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
