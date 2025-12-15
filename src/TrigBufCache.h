// Copyright Mihai Preda

#pragma once

#include "Buffer.h"
#include "FFTConfig.h"

#include <mutex>

using TrigBuf = Buffer<double2>;
using TrigPtr = shared_ptr<TrigBuf>;

class StrongCache {
  vector<TrigPtr> ptrs;
  u32 pos{};

public:
  explicit StrongCache(u32 size) : ptrs(size) {}

  void add(TrigPtr ptr) {
    ptrs.at(pos) = ptr;
    if (++pos >= ptrs.size()) { pos = 0; }
  }
};

class TrigBufCache {  
  const Context* context;
  std::mutex mut;

  std::map<tuple<u32, u32, u32, u32, u32>, TrigPtr::weak_type> small;
  std::map<tuple<u32, u32, u32, u32>, TrigPtr::weak_type> middle;

  // The shared-pointers below keep the most recent set of buffers alive even without any Gpu instance
  // referencing them. This allows a single worker to delete & re-create the Gpu instance and still reuse the buffers.
  StrongCache smallCache{4};
  StrongCache middleCache{4};

public:
  TrigBufCache(const Context* context) :
    context{context}
  {}

  ~TrigBufCache();

  TrigPtr smallTrigCombo(Args *args, FFTConfig fft, u32 width, u32 middle, u32 height, u32 nH, bool tail_single_wide);
  TrigPtr middleTrig(Args *args, FFTConfig fft, u32 SMALL_H, u32 MIDDLE, u32 W);
  TrigPtr smallTrig(Args *args, FFTConfig fft, u32 width, u32 nW, u32 middle, u32 height, u32 nH, bool tail_single_wide);
};


double2 root1Fancy(u32 N, u32 k);               // For small angles, return "fancy" cos - 1 for increased precision
double2 root1(u32 N, u32 k);

float2 root1FancyFP32(u32 N, u32 k);            // For small angles, return "fancy" cos - 1 for increased precision
float2 root1FP32(u32 N, u32 k);

uint2 root1GF31(u32 N, u32 k);
ulong2 root1GF61(u32 N, u32 k);

// Compute the size of the largest possible trig buffer given width, middle, height (in number of float2 values)
#define SMALLTRIG_FP64_SIZE(W,M,H,nH)           (W != H || H == 0 ? W * 5 : SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)) // See genSmallTrigFP64
#define SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)      (H * 5 + (W * M / 2 + 1) * 2 * H / nH)                          // See genSmallTrigComboFP64
#define MIDDLETRIG_FP64_SIZE(W,M,H)             (H + W + H)                                                     // See genMiddleTrigFP64

// Compute the size of the largest possible trig buffer given width, middle, height (in number of float2 values)
#define SMALLTRIG_FP32_SIZE(W,M,H,nH)           (W != H || H == 0 ? W : SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH))     // See genSmallTrigFP32
#define SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH)      (H + (W * M / 2 + 1) * 2 * H / nH)                              // See genSmallTrigComboFP32
#define MIDDLETRIG_FP32_SIZE(W,M,H)             (H + W + H)                                                     // See genMiddleTrigFP32

// Compute the size of the largest possible trig buffer given width, middle, height (in number of uint2 values)
#define SMALLTRIG_GF31_SIZE(W,M,H,nH)           (W != H || H == 0 ? W : SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH))     // See genSmallTrigGF31
#define SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH)      (H + (W * M / 2 + 1) * 2 * H / nH)                              // See genSmallTrigComboGF31
#define MIDDLETRIG_GF31_SIZE(W,M,H)             (H * (M - 1) + W + H)                                           // See genMiddleTrigGF31

// Compute the size of the largest possible trig buffer given width, middle, height (in number of ulong2 values)
#define SMALLTRIG_GF61_SIZE(W,M,H,nH)           (W != H || H == 0 ? W : SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH))     // See genSmallTrigGF61
#define SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH)      (H + (W * M / 2 + 1) * 2 * H / nH)                              // See genSmallTrigComboGF61
#define MIDDLETRIG_GF61_SIZE(W,M,H)             (H * (M - 1) + W + H)                                           // See genMiddleTrigGF61

// Convert above sizes to distances (in units of double2)
#define SMALLTRIG_FP64_DIST(W,M,H,nH)           SMALLTRIG_FP64_SIZE(W,M,H,nH)
#define SMALLTRIGCOMBO_FP64_DIST(W,M,H,nH)      SMALLTRIGCOMBO_FP64_SIZE(W,M,H,nH)
#define MIDDLETRIG_FP64_DIST(W,M,H)             MIDDLETRIG_FP64_SIZE(W,M,H)

#define SMALLTRIG_FP32_DIST(W,M,H,nH)           SMALLTRIG_FP32_SIZE(W,M,H,nH) * sizeof(float) / sizeof(double)
#define SMALLTRIGCOMBO_FP32_DIST(W,M,H,nH)      SMALLTRIGCOMBO_FP32_SIZE(W,M,H,nH) * sizeof(float) / sizeof(double)
#define MIDDLETRIG_FP32_DIST(W,M,H)             MIDDLETRIG_FP32_SIZE(W,M,H) * sizeof(float) / sizeof(double)

#define SMALLTRIG_GF31_DIST(W,M,H,nH)           SMALLTRIG_GF31_SIZE(W,M,H,nH) * sizeof(uint) / sizeof(double)
#define SMALLTRIGCOMBO_GF31_DIST(W,M,H,nH)      SMALLTRIGCOMBO_GF31_SIZE(W,M,H,nH) * sizeof(uint) / sizeof(double)
#define MIDDLETRIG_GF31_DIST(W,M,H)             MIDDLETRIG_GF31_SIZE(W,M,H) * sizeof(uint) / sizeof(double)

#define SMALLTRIG_GF61_DIST(W,M,H,nH)           SMALLTRIG_GF61_SIZE(W,M,H,nH) * sizeof(ulong) / sizeof(double)
#define SMALLTRIGCOMBO_GF61_DIST(W,M,H,nH)      SMALLTRIGCOMBO_GF61_SIZE(W,M,H,nH) * sizeof(ulong) / sizeof(double)
#define MIDDLETRIG_GF61_DIST(W,M,H)             MIDDLETRIG_GF61_SIZE(W,M,H) * sizeof(ulong) / sizeof(double)
