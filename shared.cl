#define STR(x) XSTR(x)
#define XSTR(x) #x

#define OVERLOAD __attribute__((overloadable))

#if DEBUG
#define assert(cond) if (!(cond)) { printf("assert(%s) failed at line %d\n", STR(cond), __LINE__); }
#define assert2(cond, mes) if (!(cond)) { printf("assert(%s): \"%s\" at line #%d", STR(cond), mes, __LINE__); }
// __builtin_trap();
#else
#define assert(cond)
#define assert2(cond, mes)
//__builtin_assume(condition)
#endif // DEBUG

#if AMDGPU
// On AMDGPU the default is HAS_ASM
#if !NO_ASM
#define HAS_ASM 1
#endif
#endif // AMDGPU

// The ROCm optimizer does a very, very poor job of keeping register usage to a minimum.  This negatively impacts occupancy
// which can make a big performance difference.  To counteract this, we can prevent some loops from being unrolled.
// For AMD GPUs we do not unroll fft_WIDTH loops. For nVidia GPUs, we unroll everything.
#if !UNROLL_WIDTH && !NO_UNROLL_WIDTH && !AMDGPU
#define UNROLL_WIDTH 1
#endif

// Expected defines: EXP the exponent.
// WIDTH, SMALL_HEIGHT, MIDDLE.

#define BIG_HEIGHT (SMALL_HEIGHT * MIDDLE)
#define ND (WIDTH * BIG_HEIGHT)
#define NWORDS (ND * 2u)

#if WIDTH == 1024 || WIDTH == 256
#define NW 4
#else
#define NW 8
#endif

#if SMALL_HEIGHT == 1024 || SMALL_HEIGHT == 256
#define NH 4
#else
#define NH 8
#endif

#define G_W (WIDTH / NW)
#define G_H (SMALL_HEIGHT / NH)

#if !OUT_WG
#define OUT_WG 256
#endif

#if !OUT_SIZEX
#if AMDGPU
#define OUT_SIZEX 32
#else // AMDGPU
#if G_W >= 64
#define OUT_SIZEX 4
#else
#define OUT_SIZEX 32
#endif
#endif
#endif

#if !OUT_SPACING
#if AMDGPU
#define OUT_SPACING 4
#else
#define OUT_SPACING 1
#endif
#endif

#if !IN_WG
#define IN_WG 256
#endif

#if !IN_SIZEX
#if AMDGPU
#define IN_SIZEX 32
#else // !AMDGPU
#if G_W >= 64
#define IN_SIZEX 4
#else
#define IN_SIZEX 32
#endif
#endif
#endif

#if UNROLL_WIDTH
#define UNROLL_WIDTH_CONTROL
#else
#define UNROLL_WIDTH_CONTROL       __attribute__((opencl_unroll_hint(1)))
#endif

void bar() { barrier(0); }

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

i32 I32(i64 x) { return x; }
u32 U32(u64 x) { return x; }

bool test_(u32 bits, u32 pos) { return (bits >> pos) & 1; }

#define STEP (NWORDS - (EXP % NWORDS))
u32 EXTRA(u64 k) { return STEP * k % NWORDS; }
#define SMALL_BITS (EXP / NWORDS)
bool isBigExtra(u32 extra) { return extra < NWORDS - STEP; }
bool isBigK(u32 k) { return isBigExtra(EXTRA(k)); }
u32 bitlen(bool b) { return SMALL_BITS + b; }
u32 bitlenK(u32 k) { return bitlen(isBigK(k)); }
uint2 bitlen2K(u32 k) { return (uint2) (bitlenK(2*k), bitlenK(2*k + 1)); }
uint2 nBits2(u32 bits, u32 pos) { return (uint2) (bitlen(test_(bits, 2*pos)), bitlen(test_(bits, 2*pos + 1))); }

T2 pair(T a, T b) { return (T2) (a, b); }

T2 OVERLOAD swap(T2 a)      { return pair(a.y, a.x); }
T2 OVERLOAD conjugate(T2 a) { return pair(a.x, -a.y); }
T2 addsub(T2 a) { return pair(a.x + a.y, a.x - a.y); }

#define X2(a, b) { T2 t = a; a = t + b; b = t - b; }
#define SWAP(a, b) { T2 t = a; a = b; b = t; }

Word lowBits(i64 u, u32 bits) {
#if HAS_ASM
  i32 tmp;
#if SMALL_BITS <= 31
  assert(sizeof(Word) == 4);
  assert(bits <= 32);
  __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (I32(u)), "v" (bits));
  return tmp;
#else
  assert(sizeof(Word) == 8);
  assert(bits >= 32);
  __asm("v_bfe_i32 %0, %1, 0, %2" : "=v" (tmp) : "v" (I32(u>>32)), "v" (bits - 32));
  return (I64(tmp) << 32) | U32(u);
#endif
#else
  return ((u << (64 - bits)) >> (64 - bits));
#endif
}
