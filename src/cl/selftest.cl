// Copyright (C) Mihai Preda

#include "base.cl"
#include "trig.cl"
#include "fft3.cl"
#include "fft4.cl"
#include "fft5.cl"
#include "fft6.cl"
#include "fft7.cl"
#include "fft8.cl"
#include "fft9.cl"
#include "fft10.cl"
#include "fft11.cl"
#include "fft12.cl"
#include "fft13.cl"
#include "fft14.cl"
#include "fft15.cl"
#include "fft16.cl"

// Measure instruction latency.
KERNEL(32) testTime(int what, global i64* io) {
#if HAS_ASM
  i64 clock0, clock1;
  
  if (what == 6) { // V_MAD_U64_U32
    u32 a = 2;
    u64 b = 3;
        
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b));
        
    for (int i = 0; i < 48; ++i) {
      __asm("v_mad_u64_u32 %1, vcc, %0, %0, %1" : : "v"(a), "v"(b));
    }
        
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));
  } else if (what == 0) { // V_NOP
    // clock0 = __builtin_readcyclecounter();
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0));
    
    for (int i = 0; i < 48; ++i) {
      __asm("v_nop");
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1) : );
  } else if (what == 1) { // V_ADD_I32
    int a = 2, b = 3;
    
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b));
    
    for (int i = 0; i < 48; ++i) {
      __asm("v_add_i32 %0, %1, %0" : : "v"(a), "v"(b));
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));
  } else if (what == 2) { // V_FMA_F32
    float a = 2, b = 3;
    
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b));
    
    for (int i = 0; i < 48; ++i) {
      __asm("v_fma_f32 %0, %0, %1, %0" : : "v"(a), "v"(b));
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));    
  } else if (what == 3) { // V_ADD_F64
    double a = 2, b = 3;
    
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b));
    
    for (int i = 0; i < 48; ++i) {
      __asm("v_add_f64 %0, %0, %1" : : "v"(a), "v"(b));
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));    
  } else if (what == 4) { // V_FMA_F64
    double a = 2, b = 3, c = 4, d = 5;
    
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b), "v"(c), "v"(d));
    
    for (int i = 0; i < 24; ++i) {
      __asm(
      "v_fma_f64 %0, %0, %1, %0\n\t"
      "v_fma_f64 %2, %2, %3, %2\n\t"
      : : "v"(a), "v"(b), "v"(c), "v"(d));
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));
  } else if (what == 5) { // V_MUL_F64
    double a = 2, b = 3;
    
    __asm (
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock0) : "v"(a), "v"(b));
    
    for (int i = 0; i < 48; ++i) {
      __asm("v_mul_f64 %0, %0, %1" : : "v"(a), "v"(b));
    }
    
    __asm(
    "s_memtime %0\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : "=s"(clock1));
  }
  
  if (get_local_id(0) == 0) {
    io[get_group_id(0)] = clock1 - clock0;
  }
#endif
}


KERNEL(256) testFFT3(global double2* io) {
  T2 u[4];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 3; ++i) { u[i] = io[i]; }
    fft3(u);
    for (int i = 0; i < 3; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT4(global double2* io) {
  T2 u[4];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 4; ++i) { u[i] = io[i]; }
    fft4(u);
    for (int i = 0; i < 4; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT5(global double2* io) {
  T2 u[5];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 5; ++i) { u[i] = io[i]; }
    fft5(u);
    for (int i = 0; i < 5; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT6(global double2* io) {
  T2 u[6];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 6; ++i) { u[i] = io[i]; }
    fft6(u);
    for (int i = 0; i < 6; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT7(global double2* io) {
  T2 u[7];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 7; ++i) { u[i] = io[i]; }
    fft7(u);
    for (int i = 0; i < 7; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT8(global double2* io) {
  T2 u[8];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 8; ++i) { u[i] = io[i]; }
    fft8(u);
    for (int i = 0; i < 8; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT9(global double2* io) {
  T2 u[9];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 9; ++i) { u[i] = io[i]; }
    fft9(u);
    for (int i = 0; i < 9; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT10(global double2* io) {
  T2 u[10];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 10; ++i) { u[i] = io[i]; }
    fft10(u);
    for (int i = 0; i < 10; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT11(global double2* io) {
  T2 u[11];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 11; ++i) { u[i] = io[i]; }
    fft11(u);
    for (int i = 0; i < 11; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT13(global double2* io) {
  T2 u[13];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 13; ++i) { u[i] = io[i]; }
    fft13(u);
    for (int i = 0; i < 13; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testTrig(global double2* out) {
  for (i32 k = get_global_id(0); k < ND / 8; k += get_global_size(0)) {
#if 0
    double angle = M_PI / (ND / 2) * k;
    out[k] = U2(cos(angle), -sin(angle));
#else
    out[k] = slowTrig_N(k, ND/8);
#endif
  }
}

KERNEL(256) testFFT(global double2* io) {
#define SIZE 16
  double2 u[SIZE];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < SIZE; ++i) { u[i] = io[i]; }
    fft16(u);
    for (int i = 0; i < SIZE; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT15(global double2* io) {
  double2 u[15];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 15; ++i) { u[i] = io[i]; }
    fft15(u);
    for (int i = 0; i < 15; ++i) { io[i] = u[i]; }
  }
}

KERNEL(256) testFFT14(global double2* io) {
  double2 u[14];
  if (get_global_id(0) == 0) {
    for (int i = 0; i < 14; ++i) { u[i] = io[i]; }
    fft14(u);
    for (int i = 0; i < 14; ++i) { io[i] = u[i]; }
  }
}
