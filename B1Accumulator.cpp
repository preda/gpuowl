// Copyright Mihai Preda

#include "B1Accumulator.h"

#include "Gpu.h"
#include "Memlock.h"
#include "GmpUtil.h"
#include "Args.h"
#include "checkpoint.h"

#include <numeric>

B1Accumulator::B1Accumulator(Gpu* gpu, Saver* saver, u32 E, u32 maxBufs)
  : E{E}, b1{saver->b1}, nBits{powerSmoothBits(E, b1)}, gpu{gpu}, saver{saver}, maxBufs{maxBufs} { //681
  assert(maxBufs > 0);
  log("B1=%u (%u bits)\n", b1, nBits);
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
      assert(engaged.empty());
      bufs = gpu->makeBufVector(maxBufs);        
      // log("B1: allocating %u buffers\n", maxBufs);
    }

    engaged.clear();
    engaged.resize(maxBufs);
}

vector<u32> B1Accumulator::fold() {
  if (b1 == 0) { return {}; }
    
  u32 n = bufs.size();
  assert(n > 1 && engaged.size() == n);
    
  Timer timer;    
  Words ret;

  u32 nEngaged = std::accumulate(engaged.begin(), engaged.end(), 0);
    
  if (nEngaged == 0) {
    ret.resize((E-1)/32 +1);
    ret[0] = 1;
  } else {
    int last = n - 1;
    while (!engaged[last]) { --last; }
    assert(last >= 0);
      
    if (last >= 1) {
      if (engaged[last - 1]) {
        gpu->mul(bufs[last - 1], bufs[last]);
      } else {
        bufs[last - 1] << bufs[last];
      }

      for (int i = last - 2; i >= 0; --i) {
        gpu->mul(bufs[last], bufs[last - 1]);
        if (engaged[i]) { gpu->mul(bufs[last - 1], bufs[i]); }
      }
      gpu->square(bufs[last]);
      gpu->mul(bufs[0], bufs[last - 1], bufs[last]);
    }
      
    engaged[0] = true;
    for (u32 i = 1; i < n; ++i) { engaged[i] = false; }
    ret = gpu->readAndCompress(bufs[0]);
  }
    
  log("B1 fold(%u) (%u set) took %.2fs\n", n, nEngaged, timer.deltaSecs());
  return ret;
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

void B1Accumulator::load(u32 k) {
    if (!b1 || k >= nBits) {
      release();
      nextK = 0;
      return;
    }

    if (k == 0) {
      alloc();
      nextK = findFirstBitSet();
      log("Starting B1=%u, first bit %u\n", b1, nextK);
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
    engaged[0] = true;
}

template<typename T>
void B1Accumulator::step(u32 kAt, Buffer<T>& data) {
  assert(nextK && kAt == nextK);
  assert(nextK < bits.size() && bits[nextK]);
  assert(engaged.size() == bufs.size());
  
  auto [nextPop, sum] = findFirstPop(nextK + 1);
  nextK = nextPop;
  assert(nextK == 0 || bits[nextK]);
  
  if (!engaged[sum]) {
    bufs[sum].set(1);
    engaged[sum] = true;
  }
    
  gpu->mul(bufs[sum], data);
}

template void B1Accumulator::step<int>(u32 kAt, Buffer<int>& data);
template void B1Accumulator::step<double>(u32 kAt, Buffer<double>& data);
