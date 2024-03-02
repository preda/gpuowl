// Copyright (C) Mihai Preda

#include "gpuowl.cl"

/*
KERNEL(64) writeGlobals(global double2* trig2ShDP, global double2* trigBhDP, global double2* trigNDP,
                        global double2* trigW,
                        global double2* threadWeights, global double2* carryWeights
                        ) {
  for (u32 k = get_global_id(0); k < 2 * SMALL_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_2SH[k] = trig2ShDP[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT/8 + 1; k += get_global_size(0)) { TRIG_BH[k] = trigBhDP[k]; }

#if TRIG_COMPUTE == 0
  for (u32 k = get_global_id(0); k < ND/8 + 1; k += get_global_size(0)) { TRIG_N[k] = trigNDP[k]; }
#elif TRIG_COMPUTE == 1
  for (u32 k = get_global_id(0); k <= WIDTH/2; k += get_global_size(0)) { TRIG_W[k] = trigW[k]; }
#endif

  // Weights
  for (u32 k = get_global_id(0); k < G_W; k += get_global_size(0)) { THREAD_WEIGHTS[k] = threadWeights[k]; }
  for (u32 k = get_global_id(0); k < BIG_HEIGHT / CARRY_LEN; k += get_global_size(0)) { CARRY_WEIGHTS[k] = carryWeights[k]; }  
}
*/

// Read 64 Word2 starting at position 'startDword'.
KERNEL(64) readResidue(P(Word2) out, CP(Word2) in, u32 startDword) {
  u32 me = get_local_id(0);
  u32 k = (startDword + me) % ND;
  u32 y = k % BIG_HEIGHT;
  u32 x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}

KERNEL(256) sum64(global ulong* out, u32 sizeBytes, global ulong* in) {
  if (get_global_id(0) == 0) { out[0] = 0; }
  
  ulong sum = 0;
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(u64); p += get_global_size(0)) {
    sum += in[p];
  }
  sum = work_group_reduce_add(sum);
  if (get_local_id(0) == 0) {
    u32 low = sum;
    u32 prev = atomic_add((global u32*)out, low);
    u32 high = (sum + prev) >> 32;
    atomic_add(((global u32*)out) + 1, high);
  }
}

// outEqual must be "true" on entry.
KERNEL(256) isEqual(P(bool) outEqual, u32 sizeBytes, global i64 *in1, global i64 *in2) {
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(i64); p += get_global_size(0)) {
    if (in1[p] != in2[p]) {
      *outEqual = false;
      return;
    }
  }
}

// outNotZero must be "false" on entry.
KERNEL(256) isNotZero(P(bool) outNotZero, u32 sizeBytes, global i64 *in) {
  for (i32 p = get_global_id(0); p < sizeBytes / sizeof(i64); p += get_global_size(0)) {
    if (in[p] != 0) {
      *outNotZero = true;
      return;
    }
  }
}

// from transposed to sequential.
KERNEL(64) transposeOut(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(WIDTH, BIG_HEIGHT, lds, in, out);
}

// from sequential to transposed.
KERNEL(64) transposeIn(P(Word2) out, CP(Word2) in) {
  local Word2 lds[4096];
  transposeWords(BIG_HEIGHT, WIDTH, lds, in, out);
}

// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
#ifdef TEST_KERNEL

kernel void testKernel(global double* in, global float* out) {
  uint me = get_local_id(0);

  double x = in[me];
  double d = x + RNDVAL;
  out[me] = fabs((float) (x + (RNDVAL - d)));
}

#endif
