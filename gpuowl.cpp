// gpuOWL (a GPU OpenCL Lucas-Lehmer primality checker).
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"

#include <cassert>
#include <memory>
#include <cstdio>
#include <cmath>

#define K(program, name) Kernel name(program, #name);

typedef unsigned char byte;

void genBitlen(int E, int N, int W, int H, double *aTab, double *iTab, byte *bitlenTab) {
  double *pa = aTab;
  double *pi = iTab;
  byte   *pb = bitlenTab;

  auto iN = 1 / (long double) N;
  auto en = E / (long double) N;
  
  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      for (int rep = 0; rep < 2; ++rep) {
        long k = (line + col * H) * 2 + rep;
        auto p0 = k * E * iN;
        auto p1 = (k + 1) * E * iN;       
        auto c0 = ceill(p0);
        auto c1 = ceill(p1);
      
        int bits  = ((int) c1) - ((int) c0);
        auto a    = exp2l(c0 - p0);
        *pa++ = a;
        *pi++ = 1 / (8 * W * H * a);
        *pb++ = bits;
      }
    }
  }
}

double *genBigTrig(int W, int H) {
  double *out = new double[2 * W * H];
  double *p = out;
  auto base = - M_PIl / (W * H / 2);
  for (int gy = 0; gy < H / 64; ++gy) {
    for (int gx = 0; gx < W / 64; ++gx) {
      for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
          int k = (gy * 64 + y) * (gx * 64 + x);
          auto angle = k * base;
          *p++ = cosl(angle);
          *p++ = sinl(angle);
        }
      }
    }
  }
  return out;
}

double *genSin(int W, int H) {
  double *data = new double[2 * (W / 2) * H]();
  double *p = data;
  auto base = - M_PIl / (W * H);
  for (int line = 0; line < H; ++line) {
    for (int col = 0; col < (W / 2); ++col) {
      int k = line + (col + ((line == 0) ? 1 : 0)) * H;
      auto angle = k * base;
      *p++ = sinl(angle);
      *p++ = cosl(angle);
    }
  }
  return data;
}

double *smallTrigBlock(int W, int H, double *out) {
  double *p = out;
  for (int line = 1; line < H; ++line) {
    for (int col = 0; col < W; ++col) {
      auto angle = - M_PIl * line * col / (W * H / 2);
      *p++ = cosl(angle);
      *p++ = sinl(angle);
    }
  }
  return p;
}

double *genSmallTrig2K() {
  int size = 2 * 4 * 512;
  double *out = new double[size]();
  double *p = out + 2 * 8;
  p = smallTrigBlock(  8, 8, p);
  p = smallTrigBlock( 64, 8, p);
  p = smallTrigBlock(512, 4, p);
  assert(p - out == size);
  return out;
}

double *genSmallTrig1K() {
  int size = 2 * 4 * 256;
  double *out = new double[size]();
  double *p = out + 2 * 4;
  p = smallTrigBlock(  4, 4, p);
  p = smallTrigBlock( 16, 4, p);
  p = smallTrigBlock( 64, 4, p);
  p = smallTrigBlock(256, 4, p);
  assert(p - out == size);
  return out;
}

int main(int argc, char **argv) {
  printf("owLL 0.1 GPU Lucas-Lehmer\n");
  
  int W = 1024;
  int H = 2048;
  int SIZE  = W * H;
  int N = 2 * SIZE;

  // int E = 39527687;
  // int E = 79517539;
  
  int E = 0;
  int startK = 0;
  if (argc >= 2) { E = atoi(argv[1]); }

  int *data = new int[N]();
  data[0] = 4; // LL root.

  FILE *fi = fopen("owll-checkpoint.bin", "rb");
  if (fi) {
    int saveE, saveK, saveN;
    if (fscanf(fi, "OWLL1 %d %d %d\n", &saveE, &saveK, &saveN) == 3 &&
        (E == 0 || E == saveE) && N == saveN &&
        fread(data, sizeof(int) * N, 1, fi) == 1) {
      E = saveE;
      startK = saveK;
    } else {
      printf("Wrong checkpoint (or move the file \"owll-checkpoint.bin\" out of the way)\n");
    }
    fclose(fi);
  }
  
  if (E == 0 && argc < 2) {
    printf("No exponent argument, and no resume checkpoint found.\nUsage: owLL <exponent>\n");
    exit(0);
  }
  
  printf("LL of %d at iteration %d\n", E, startK);
  printf("FFT %d*%d (%dM digits, %.2f bits per digit)\n", W, H, N / (1024 * 1024), E / (double) N);
  
  Context c;
  Queue q(c);
  q.time("");

  Program program(c, "gpuowl.cl");
  K(program, fftPremul1K); 
  K(program, transposeA);
  K(program, fft2Kt);
  K(program, square2K);
  K(program, fft2K);
  K(program, transposeB);
  K(program, fft1Kt);
  K(program, carryA);
  K(program, carryB);

  q.time("OpenCL compile");
  
  auto *aTab      = new double[N];
  auto *iTab      = new double[N];
  auto *bitlenTab = new byte[N];
  genBitlen(E, N, W, H, aTab, iTab, bitlenTab);
  unsigned firstBitlen[4] = {bitlenTab[0], bitlenTab[1], bitlenTab[2 * W], bitlenTab[2 * W + 1]};
  for (int i = 0, s = 0; i < 4; ++i) { firstBitlen[i] = (s += firstBitlen[i]); }
  // printf("[debug] bitlen %d %d %d %d\n", firstBitlen[0], firstBitlen[1], firstBitlen[2], firstBitlen[3]);
  
  Buf      bufA(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, aTab);
  Buf      bufI(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, iTab);
  Buf bufBitlen(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(byte)   * N, bitlenTab);
  
  delete[] aTab;
  delete[] iTab;
  delete[] bitlenTab;  
  // q.time("gen bitlen");

  double *bigTrig = genBigTrig(W, H);
  Buf bufBigTrig(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, bigTrig);
  delete[] bigTrig;
  // q.time("gen bigTrig");
  
  double *sins = genSin(H, W); // transposed W/H !
  Buf bufSins(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N / 2, sins);
  delete[] sins;

  double *trig1K = genSmallTrig1K();
  double *trig2K = genSmallTrig2K();

  Buf bufTrig1K(c, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2 * 1024, trig1K);
  Buf bufTrig2K(c, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2 * 2048, trig2K);
  
  delete[] trig1K;
  delete[] trig2K;
  // q.time("gen sins trig");
  
  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N);

  Buf bufData (c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)  * N, data);
  Buf bufCarry(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(long) * N / 8);
  Buf bufErr  (c, CL_MEM_READ_WRITE, sizeof(int));
  q.readAndReset(bufErr);

  fftPremul1K.setArgs(bufData, buf1, bufA, bufTrig1K);
  transposeA.setArgs(buf1, bufBigTrig);
  fft2Kt.setArgs(buf1, buf2, bufTrig2K);
  square2K.setArgs(buf2, bufSins);
  fft2K.setArgs(buf2, bufTrig2K);
  transposeB.setArgs(buf2, bufBigTrig);
  fft1Kt.setArgs(buf2, buf1, bufTrig1K);  
  carryA.setArgs(buf1, bufI, bufData, bufCarry, bufBitlen, bufErr);
  carryB.setArgs(bufData, bufCarry, bufBitlen);
  
  q.time("setup");

  float maxErr = 0;

  int logStep   = 10000;
  int saveStep  = 10 * logStep;
  int bigEnd    = E - 2;
  float percent = 100 / (float) bigEnd;
  
  for (int k = startK; k < bigEnd;) {
    for (int checkpointEnd = std::min((k / saveStep + 1) * saveStep, bigEnd); k < checkpointEnd;) {
      for (int logEnd = std::min((k / logStep + 1) * logStep, checkpointEnd); k < logEnd; ++k) {
        q.run(fftPremul1K,  SIZE / 4);
        q.run(transposeA,   SIZE / 16);
        q.run(fft2Kt,       SIZE / 8);
        
        q.run(square2K,     SIZE / 2);
      
        q.run(fft2K,        SIZE / 8);
        q.run(transposeB,   SIZE / 16);
        q.run(fft1Kt,       SIZE / 4);
      
        q.run(carryA,       SIZE / 8);
        q.run(carryB,       SIZE / 8);
      }
    
      float err = q.readAndReset(bufErr) * (1 / (float) (1 << 30));
      maxErr = std::max(err, maxErr);
    
      q.readBlocking(&bufData, 0, sizeof(int) * (2048 + 2), data);
      data[2] = data[2048];
      data[3] = data[2049];
      __int128 residue = data[0] + (((int64_t) data[1]) << firstBitlen[0]) + (((int64_t) data[2]) << firstBitlen[1]) + (((__int128) data[3]) << firstBitlen[2]);

      q.time("%08d (%.2f%%), 0x%016lx error %g (max %g) ", k, k * percent, (unsigned long) residue, err, maxErr);
    }

    q.readBlocking(&bufData, 0, sizeof(int) * N, data);
    FILE *fo = fopen("owll-checkpoint.new", "wb");
    if (fo) {
      fprintf(fo, "OWLL1 %d %d %d\n", E, k, N);
      auto nr = fwrite(data, sizeof(int) * N, 1, fo);
      fclose(fo);
      if (nr == 1) {
        rename("owll-checkpoint.bin", "owll-checkpoint.old");
        rename("owll-checkpoint.new", "owll-checkpoint.bin");
      } else {
        printf("Error writing checkpoint\n");
      }
    } else {
      printf("Can't open checkpoint file \"owll-checkpoint.new\"\n");
    }
    q.time("checkpoint %d", k);
  }
  
  q.readBlocking(&bufData, 0, sizeof(int) * N, data);
  for (int i = 0, cnt = 0; i < N && cnt < 100; ++i) {
    if (data[i]) {
      printf("(%d %d) ", i, data[i]);
      ++cnt;
    }
  }
  printf("\n");

  q.time("start shutdown");
  buf1.release();
  buf2.release();
  bufBigTrig.release();
  bufTrig1K.release();
  bufTrig2K.release();
  bufSins.release();
  bufA.release();
  bufI.release();
  bufBitlen.release();
  bufData.release();
  bufCarry.release();
  bufErr.release();
  q.time("released buffers");
}
