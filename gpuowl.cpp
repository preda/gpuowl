// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "args.h"
#include "Task.h"
#include "Result.h"
#include "worktodo.h"
#include "common.h"

extern string globalCpuName;

int main(int argc, char **argv) {  
  initLog("gpuowl.log");

  Args args;
  if (!args.parse(argc, argv)) { return -1; }
  if (!args.cpu.empty()) { globalCpuName = args.cpu; }

  log("%s %s\n", PROGRAM, VERSION);

  try {
    while (Task task = Worktodo::getTask()) {
      task = task.morph(&args);    
      if (Result result = task.execute(args)) { result.write(args, task); }
      Worktodo::deleteTask(task);
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  }
  
  log("Bye\n");
}

/*
bool writeResultPRP(int E, bool isPrime, u64 res, const string &AID, const string &user, const string &cpu, int nErrors, int fftSize) {
}

bool writeResultTF(int E, u64 factor, int bitLo, int bitHi, u64 beginK, u64 endK,
                   const string &AID, const string &user, const string &cpu) {
}

bool writeResultPM1(int E, const string &factor, u32 B1, const string &AID, const string &user, const string &cpu) {
}
*/

/*
bool loadPRPF(Gpu *gpu, u32 E, u32 desiredB1, u32 desiredBlockSize, int *outK, int *outBlockSize, int *outNErrors) {
  PRPFState loaded = Checkpoint::loadPRPF(E, desiredB1, desiredBlockSize);
  if (!loaded.ok) {
    log("Invalid checkpoint for exponent %d\n", E);
    return false;
  }

  if (loaded.stage == 1) {
    gpu->writeData(loaded.base);        
  } else {  
    gpu->writeState(loaded.check, loaded.base, loaded.blockSize);
    u64 res64 = gpu->dataResidue();
    bool resOK = res64 == loaded.res64;
  
  if (resOK && gpu->checkAndUpdate(loaded.blockSize)) {
    log("OK loaded: %d/%d, blockSize %d, %016llx\n", loaded.k, E, loaded.blockSize, res64);
  } else {
    log("EE loaded: %d/%d, blockSize %d, %016llx, expected %016llx\n", loaded.k, E, loaded.blockSize, res64, loaded.res64);
    return false;
  }
  
  *outK = loaded.k;
  *outBlockSize = loaded.blockSize;
  *outNErrors = loaded.nErrors;
  return true;
}
*/

/*
// Return found factor, or zero.
tuple<u64, u64, u64> doTF(u32 exp, u32 bitLo, u32 bitEnd, Args &args) {
  if (bitLo >= bitEnd) { return false; }
  
  TFState state;
  if (!state.load(exp)) { return false; }

  if (state.bitLo >= bitEnd || (state.nDone == state.nTotal && state.bitHi >= bitEnd)) { return false; }

  if (state.bitLo > bitLo) { bitLo = state.bitLo; }
  
  if (bitLo >= bitEnd) { return false; }

  unique_ptr<TF> tf = makeTF(args);
  assert(tf);

  u64 beginK = 0, endK = 0;
  int nDone = (state.bitHi >= bitEnd) ? state.nDone : 0;
  u64 factor = tf->findFactor(exp, bitLo, bitEnd, nDone, state.nTotal, &beginK, &endK, args.timeKernels);
  return make_tuple(factor, beginK, endK);
}
*/

/*
    case Task::PM1: {
      unique_ptr<Gpu> gpu = makeGpu(task.exponent, args);
      string factor;
      if (!checkPM1(gpu.get(), exp, task.B1, args, factor)
          || !writeResultPM1(exp, factor, task.B1, task.AID, args.user, args.cpu)
          || !Worktodo::deleteTask(task)
          || !factor.empty()) {
        stop = true;
      }
    }


  if (task.kind == Task::TF) {
      assert(task.bitLo < task.bitHi);

      TFState state;
      if (!state.load(exp)) { break; }
      u32 bitDone = (state.nDone == state.nTotal) ? state.bitHi : state.bitLo;
      if (bitDone >= task.bitHi) {
        log("TF(%u) from %u to %u already done to %u; dropping.\n", exp, task.bitLo, task.bitHi, bitDone);
        Worktodo::deleteTask(task);
        continue;
      }
      u32 bitStart = max(task.bitLo, bitDone);
      assert(bitStart < taks.bitHi);
      u32 classDone = (state.bitHi >= task.bitHi) ? state.nDone : 0;
      unique_ptr<TF> tf = makeTF(args);
      u64 beginK = 0, endK = 0;
      u64 factor = tf->findFactor(exp, bitStart, task.bitHi, classDone, state.nTotal, &beginK, &endK, args.timeKernels);
      if (!writeResultTF(exp, factor, bitStart, task.bitHi, beginK, endK, task.AID, args.user, args.cpu)
          || !Worktodo::deleteTask(task)) {
        break;
      }
*/
