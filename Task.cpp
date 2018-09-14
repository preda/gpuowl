#include "Task.h"

#include "OpenGpu.h"
#include "OpenTF.h"
#include "Result.h"
#include "checkpoint.h"
#include "args.h"

#include <cstdio>
#include <cmath>

// Ideally how far we want an exponent TF-ed.
static u32 targetBits(u32 exp) { return 81 + 2.5 * (log2(exp) - log2(332000000)); }

static u32 needsMoreTF(u32 exp, Args *args) {
  TFState state = TFState::load(exp);
  u32 bitDone = (state.nDone == state.nTotal) ? state.bitHi : state.bitLo;
  u32 bitTarget = targetBits(exp) + args->tfDelta;
  return (bitDone < bitTarget) ? bitTarget : 0;
}

Task Task::morph(Args *args) {    
  if ((kind == PRP || kind == PRPF) && bitLo && args->enableTF) {
    u32 bitTarget = needsMoreTF(exponent, args);
    if (bitTarget > bitLo) {
      return Task{TF, exponent, "", "", bitLo, bitTarget, 0};
    }
  } else if (kind == PRPF && !PRPFState::canProceed(exponent, B1)) {
    return Task{PM1, exponent, "", "", 0, 0, B1};
  } else if (kind == TF) {
    assert(bitLo < bitHi);
    auto state = TFState::load(exponent);
    u32 bitDone = (state.nDone == state.nTotal) ? state.bitHi : state.bitLo;
    if (bitDone >= bitHi) {
      log("TF(%u) from %u to %u already done to %u\n", exponent, bitLo, bitHi, bitDone);
      // Worktodo::deleteTask(task);
    }
    bitLo = min(bitHi, max(bitLo, bitDone));
  }
  return *this;
}

unique_ptr<TF> makeTF(const Args &args) { return OpenTF::make(args); }
unique_ptr<Gpu> makeGpu(u32 E, const Args &args) { return OpenGpu::make(E, args); }

vector<string> getDevices() {
  vector<string> ret;
  for (auto id : getDeviceIDs(false)) { ret.push_back(getLongInfo(id)); }
  return ret;
}

Result Task::execute(const Args &args) {
  if (kind == TF) {
    assert(bitLo <= bitHi);
    if (bitLo == bitHi) { return Result{Result::NONE}; }
    
    auto state = TFState::load(exponent);
    u32 classDone = (state.bitHi >= bitHi) ? state.nDone : 0;
    u64 beginK = 0, endK = 0;
    string factor = makeTF(args)->findFactor(exponent, bitLo, bitHi, classDone, state.nTotal, &beginK, &endK, args.timeKernels);
    return Result{Result::TF, factor, beginK, endK};
    
  } else if (kind == PM1) {
    string factor = makeGpu(exponent, args)->factorPM1(exponent, B1, args);
    return Result{Result::PM1, factor};
    
  } else if (kind == PRP) {
    u64 res64 = 0;
    u32 nErrors = 0;
    u32 fftSize = 0;
    bool isPrime = makeGpu(exponent, args)->isPrimePRP(exponent, args, &res64, &nErrors, &fftSize);
    return Result{Result::PRP, "", 0, 0, isPrime, res64, nErrors, fftSize};
  }
  assert(false);
  return Result{Result::NONE};
}
