// Copyright (C) Mihai Preda

#include "base.cl"

#if READRESIDUE

// Because the data "in" is stored transposed, and we want to read
// a number of logically successive values, we have a very bad read access pattern
KERNEL(32) readResidue(P(Word2) out, CP(Word2) in) {
  u32 me = get_local_id(0);
  u32 k = (ND - 16 + me) % ND;
  u32 y = k % BIG_HEIGHT;
  u32 x = k / BIG_HEIGHT;
  out[me] = in[WIDTH * y + x];
}
#endif

#if SUM64
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
#endif

#if ISEQUAL
// outEqual must be "true" on entry.
KERNEL(256) isEqual(global i64 *in1, global i64 *in2, P(int) outEqual) {
  for (i32 p = get_global_id(0); p < ND; p += get_global_size(0)) {
    if (in1[p] != in2[p]) {
      *outEqual = 0;
      return;
    }
  }
}
#endif

#if TEST_KERNEL
// Generate a small unused kernel so developers can look at how well individual macros assemble and optimize
kernel void testKernel(global double* in, global float* out) {
  uint me = get_local_id(0);
  double RNDVAL = 1 + 1e-100;

  double x = in[me];
  double d = x + RNDVAL;
  long double a = x + RNDVAL;
  out[me] = fabs((float) (x + (RNDVAL - d))) * sin((double) a);
}

#endif
