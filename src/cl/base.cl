// Copyright (C) Mihai Preda and George Woltman

#pragma once

/* Tunable paramaters for -ctune :

IN_WG, OUT_WG: 64, 128, 256. Default: 256.
IN_SIZEX, OUT_SIZEX: 4, 8, 16, 32. Default: 32 on AMD, 4 on Nvidia.
UNROLL_W: 0, 1. Default: 0 on AMD, 1 on Nvidia.
UNROLL_H: 0, 1. Default: 1.
*/

/* List of code-specific macros. These are set by the C++ host code or derived
EXP        the exponent
WIDTH
SMALL_HEIGHT
MIDDLE
CARRY_LEN
NW
NH
AMDGPU  : if this is an AMD GPU
HAS_ASM : set if we believe __asm() can be used

-- Derived from above:
BIG_HEIGHT == SMALL_HEIGHT * MIDDLE
ND         number of dwords == WIDTH * MIDDLE * SMALL_HEIGHT
NWORDS     number of words  == ND * 2
G_W        "group width"  == WIDTH / NW
G_H        "group height" == SMALL_HEIGHT / NH
 */

#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#pragma OPENCL FP_CONTRACT ON

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// 64-bit atomics are not used ATM
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#if DEBUG
#define assert(condition) if (!(condition)) { printf("assert(%s) failed at line %d\n", STR(condition), __LINE__ - 1); }
// __builtin_trap();
#else
#define assert(condition)
//__builtin_assume(condition)
#endif // DEBUG

#if AMDGPU
// On AMDGPU the default is HAS_ASM
#if !NO_ASM
#define HAS_ASM 1
#endif
#endif // AMDGPU

// On Nvidia we need the old sync between groups in carryFused
#if !defined(OLD_FENCE) && !AMDGPU
#define OLD_FENCE 1
#endif

#if FFT_VARIANT > 3
#error FFT_VARIANT must be between 0 and 3
#endif

#if defined(TRIG_HI) || defined(CLEAN)
#error Use FFT_VARIANT instead of TRIG_HI or CLEAN
#endif

#define TRIG_HI (FFT_VARIANT & 1)
#define CLEAN (FFT_VARIANT >> 1)

#if !defined(UNROLL_W)
#if AMDGPU
#define UNROLL_W 0
#else
#define UNROLL_W 1
#endif
#endif

#if !defined(UNROLL_H)
#if AMDGPU && (SMALL_HEIGHT >= 1024)
#define UNROLL_H 0
#else
#define UNROLL_H 1
#endif
#endif

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)

#if (NW != 4 && NW != 8) || (NH != 4 && NH != 8)
#error NW and NH must be passed in, expected value 4 or 8.
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

typedef i32 Word;
typedef int2 Word2;

typedef double T;
typedef double2 T2;

#define RE(a) (a.x)
#define IM(a) (a.y)

#define P(x) global x * restrict
#define CP(x) const P(x)

// For reasons unknown, loading trig values into nVidia's constant cache has terrible performance
#if AMDGPU
typedef constant const T2* Trig;
#else
typedef global const T2* Trig;
#endif
// However, caching weights in nVidia's constant cache improves performance.
// Even better is to not pollute the constant cache with weights that are used only once.
// This requires two typedefs depending on how we want to use the BigTab pointer.
// For AMD we can declare BigTab as constant or global - it doesn't really matter.
typedef constant const double2* ConstBigTab;
#if AMDGPU
typedef constant const double2* BigTab;
#else
typedef global const double2* BigTab;
#endif

#define KERNEL(x) kernel __attribute__((reqd_work_group_size(x, 1, 1))) void

// Prototypes
void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base);
void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base);
void bar(void);

void read(u32 WG, u32 N, T2 *u, const global T2 *in, u32 base) {
  for (u32 i = 0; i < N; ++i) { u[i] = in[base + i * WG + (u32) get_local_id(0)]; }
}

void write(u32 WG, u32 N, T2 *u, global T2 *out, u32 base) {
  out += base + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
}

// Parameters we may want to let user tune.  WIDTH other than 512 and 1K is untested.  SMALL_HEIGHT other than 256 and 512 is untested.
#define ROTATION 1                              // Turns on rotating width and small_height rows
#define WIDTH_ROTATE_CHUNK_SIZE 32              // Rotate blocks of 32 T2 values = 512 bytes
#define HEIGHT_ROTATE_CHUNK_SIZE 16             // Rotate blocks of 16 T2 values = 256 bytes
#define VARIABLE_WIDTH_ROTATE 0                 // Each width u[i] gets a different rotation amount
#define MIDDLE_SHUFFLE_WRITE 1                  // Radeon VII likes MiddleShuffleWrite, Titan V apparently not

// nVidia Titan V hates rotating and LDS-less middle writes
#if !AMDGPU
#undef ROTATION
#define ROTATION 0
#undef MIDDLE_SHUFFLE_WRITE
#define MIDDLE_SHUFFLE_WRITE 0
#endif

// Rotate width elements on output from fft_WIDTH and as input to fftMiddleIn.
// Not all lines are rotated the same amount so that fftMiddleIn reads a more varied distribution of addresses.
// This can be faster on AMD GPUs, not certain about nVidia GPUs.
u32 rotate_width_amount(u32 y) {
#if !VARIABLE_WIDTH_ROTATE
  u32 num_sections = WIDTH / WIDTH_ROTATE_CHUNK_SIZE;
  u32 num_rotates = y % num_sections;           // if y increments by SMALL_HEIGHT, final rotate amount won't change after applying "mod WIDTH"
#else
  // Create a formula where each u[i] gets a different rotation amount
  u32 num_sections = WIDTH / WIDTH_ROTATE_CHUNK_SIZE;
  u32 num_rotates = y % num_sections * MIDDLE;  // each increment of y adds MIDDLE
  num_rotates += y / SMALL_HEIGHT;              // each increment of i in u[i] will add 1
  num_rotates &= 255;                           // keep num_rotates small (probably not necessary)
#endif
  return num_rotates * WIDTH_ROTATE_CHUNK_SIZE;
}
u32 rotate_width_x(u32 x, u32 rot_amt) {        // rotate x coordinate using a cached rotate_amount
  return (x + rot_amt) % WIDTH;
}
u32 rotate_width(u32 y, u32 x) {                // rotate x coordinate (no cached rotate amount)
  return rotate_width_x(x, rotate_width_amount(y));
}
void readRotatedWidth(T2 *u, CP(T2) in, u32 y, u32 x) {
#if !ROTATION                                   // No rotation, might be better on nVidia cards
  in += y * WIDTH + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT * WIDTH]; }
#elif !VARIABLE_WIDTH_ROTATE                    // True if adding SMALL_HEIGHT to y results in same rotation amount
  in += y * WIDTH + rotate_width(y, x);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT * WIDTH]; }
#else                                           // Adding SMALL_HEIGHT to y results in different rotation
  in += y * WIDTH;
  u32 rot_amt = rotate_width_amount(y);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[rotate_width_x(x, rot_amt)]; in += SMALL_HEIGHT * WIDTH; rot_amt += SMALL_HEIGHT * WIDTH_ROTATE_CHUNK_SIZE; }
#endif
}
void writeRotatedWidth(u32 WG, u32 N, T2 *u, P(T2) out, u32 line) {
#if !ROTATION                                   // No rotation, might be better on nVidia cards
  out += line * WIDTH + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
#else
  u32 me = (u32) get_local_id(0);
  u32 rot_amt = rotate_width_amount(line);
  out += line * WIDTH;
  for (u32 i = 0; i < N; ++i) { out[rotate_width_x (i * WG + me, rot_amt)] = u[i]; }
#endif
}

// Rotate height elements on output from fft_HEIGHT and as input to fftMiddleOut.
// Not all lines are rotated the same amount so that fftMiddleOut reads a more varied distribution of addresses.
// This can be faster on AMD GPUs, not certain about nVidia GPUs.
u32 rotate_height_amount (u32 y) {
  u32 num_sections = SMALL_HEIGHT / HEIGHT_ROTATE_CHUNK_SIZE;
  return (y % num_sections) * (HEIGHT_ROTATE_CHUNK_SIZE);
}
u32 rotate_height_x(u32 x, u32 rot_amt) {       // rotate x coordinate using a cached rotate_amount
  return (x + rot_amt) % SMALL_HEIGHT;
}
u32 rotate_height(u32 y, u32 x) {               // rotate x coordinate (no cached rotate amount)
  return rotate_height_x(x, rotate_height_amount(y));
}
void readRotatedHeight(T2 *u, CP(T2) in, u32 y, u32 x) {
#if !ROTATION                                   // No rotation, might be better on nVidia cards
  in += y * MIDDLE * SMALL_HEIGHT + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT]; }
#elif 0                                         // Set if adding 1 to y results in same rotation
  y *= MIDDLE;
  in += y * SMALL_HEIGHT + rotate_height(y, x);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT]; }
#else                                           // Adding SMALL_HEIGHT to line results in different rotation
  y *= MIDDLE;
  in += y * SMALL_HEIGHT;
  u32 rot_amt = rotate_height_amount(y);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[rotate_height_x(x, rot_amt)]; in += SMALL_HEIGHT; rot_amt += HEIGHT_ROTATE_CHUNK_SIZE; }
#endif
}
void writeRotatedHeight(u32 WG, u32 N, T2 *u, P(T2) out, u32 line) {
#if !ROTATION                                   // No rotation, might be better on nVidia cards
  out += line * SMALL_HEIGHT + (u32) get_local_id(0);
  for (u32 i = 0; i < N; ++i) { out[i * WG] = u[i]; }
#else
  u32 me = (u32) get_local_id(0);
  u32 rot_amt = rotate_height_amount(line);
  out += line * SMALL_HEIGHT;
  for (u32 i = 0; i < N; ++i) { out[rotate_height_x (i * WG + me, rot_amt)] = u[i]; }
#endif
}


T2 U2(T a, T b) { return (T2) (a, b); }

void bar() {
  // barrier(CLK_LOCAL_MEM_FENCE) is correct, but it turns out that on some GPUs
  // (in particular on Radeon VII and Radeon PRO VII) barrier(0) works as well and is faster.
  // So allow selecting the faster path when it works with -use FAST_BARRIER
#if FAST_BARRIER
  barrier(0);
#else
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

// On "classic" AMD GCN GPUs such as Radeon VII, the wavefront size was always 64. On RDNA GPUs the wavefront can
// be configured to be either 64 or 32. We use the FAST_BARRIER define as an indicator for GCN GPUs.
// On Nvidia GPUs the wavefront size is 32.
#if !WAVEFRONT
#if FAST_BARRIER && AMDGPU
#define WAVEFRONT 64
#else
#define WAVEFRONT 32
#endif
#endif
