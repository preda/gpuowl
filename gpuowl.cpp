// gpuOWL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"

#include <cassert>
#include <memory>
#include <cstdio>
#include <cmath>
#include <thread>

#define K(program, name) Kernel name(program, #name);

typedef unsigned char byte;
typedef int64_t i64;
typedef uint64_t u64;

void genBitlen(int E, int N, int W, int H, double *aTab, double *iTab, byte *bitlenTab) {
  double *pa = aTab;
  double *pi = iTab;
  byte   *pb = bitlenTab;

  auto iN = 1 / (long double) N;
  
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

bool isAllZero(int *p, int size) {
  for (int *end = p + size; p < end; ++p) { if (*p) { return false; } }
  return true;
}

void getShiftTab(int W, byte *bitlenTab, int tabSize, int *shiftTab) {
  int sh = 0;
  for (int i = 0; i < tabSize; ++i) {
    shiftTab[i] = sh;
    if (sh >= 64) { return; }
    sh += bitlenTab[i / 2 * 2 * W + i % 2];
  }
  assert(false);
}

u64 residue(int N, int W, int *data, const int *shiftTab) {
  i64 r = ((data[N-1] < 0) ? -1 : 0);
  for (int i = 0; shiftTab[i] < 64; ++i) { r += ((i64) data[i / 2 * 2 * W + i % 2]) << shiftTab[i]; }
  printf("%lx\n", (u64) r);
  return r;
}

FILE *logFiles[3] = {0, 0, 0};

int log(const char *fmt, ...) {
  va_list va;
  for (FILE **pf = logFiles; *pf; ++pf) {
    va_start(va, fmt);
    vfprintf(*pf, fmt, va);
    va_end(va);
  }
}

void openLogFile(int E) {
  char logName[128];
  snprintf(logName, sizeof(logName), "log-%d.txt", E);  
  FILE *logf = fopen(logName, "a");
  if (logf) {
    setlinebuf(logf);
    logFiles[1] = logf;
  }
}

class FileSaver {
private:
  char fileNameSave[64], fileNameOld[64], fileNameNew[64];
  int E, N, W, H;
  const char *saveHeader = "LL1 %10d %10d %10d %10d %10d    \n\032";
  
  
public:
  FileSaver(int iniE, int iniN, int iniW, int iniH) : E(iniE), N(iniN), W(iniW), H(iniH) {
    char base[64];
    snprintf(base, sizeof(base), "save-%d", E);
    snprintf(fileNameSave, sizeof(fileNameSave), "%s.bin", base);
    snprintf(fileNameOld,  sizeof(fileNameOld),  "%s.old", base);
    snprintf(fileNameNew,  sizeof(fileNameNew),  "%s.new", base);
  }
  
  bool load(int *data, int *startK) {
    FILE *fi = fopen(fileNameSave, "rb");
    if (!fi) { return true; }
    
    int saveE, saveK, saveW, saveH, unused;
    bool ok = false;
    if (fscanf(fi, saveHeader, &saveE, &saveK, &saveW, &saveH, &unused) == 5 &&
        E == saveE && W == saveW && H == saveH && unused == 0 &&
        fread(data, sizeof(int) * (2 * W * H), 1, fi) == 1) {
      *startK = saveK;
      ok = true;
    }
    fclose(fi);
    if (!ok) { log("Wrong '%s' file, please move it out of the way.\n", fileNameSave); }
    return ok;
  }
  
  void save(int *data, int k) const {
    FILE *fo = fopen(fileNameNew, "wb");
    if (fo) {
      bool ok = fprintf(fo, saveHeader, E, k, W, H, 0) == 64 && fwrite(data, sizeof(int) * N, 1, fo) == 1;
      fclose(fo);
      if (ok) {
        rename(fileNameSave, fileNameOld);
        rename(fileNameNew, fileNameSave);
      } else {
        log("Error saving checkpoint\n");
      }
    } else {
      log("Can't open file '%s'\n", fileNameNew);
    }
    // log("saved %d: %ld ms\n", k, q.time());
  }
};

void doLog(int E, int k, float err, float maxErr, double msPerIter, u64 res) {
  const float percent = 100 / (float) (E - 2);
  if (err > .45f) { log("Error %g is too large!", err); }
  
  int etaMins = (E - 2 - k) * msPerIter * (1 / (double) 60000) + .5;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  log("%08d / %08d [%.2f%%], ms/iter: %.3f, ETA: %dd %02d:%02d  0x%016lx error %g (max %g)\n",
      k, E, k * percent, msPerIter, days, hours, mins, res, err, maxErr);
}

int doit(int E) {
  constexpr int W = 1024;
  constexpr int H = 2048;
  constexpr int SIZE  = W * H;
  constexpr int N = 2 * SIZE;
  
  int startK = 0;
  int *data = new int[N]();
  data[0] = 4; // LL root.

  FileSaver fileSaver(E, N, W, H);
  if (!fileSaver.load(data, &startK)) { return 1; }
    
  log("LL of %d at iteration %d\n", E, startK);
  log("FFT %d*%d (%dM words, %.2f bits per word)\n", W, H, N / (1024 * 1024), E / (double) N);
  
  Context c;
  Queue q(c);
  Timer timer;
  
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

  log("OpenCL compile: %ld ms\n", timer.delta());
  
  auto *aTab      = new double[N];
  auto *iTab      = new double[N];
  auto *bitlenTab = new byte[N];
  genBitlen(E, N, W, H, aTab, iTab, bitlenTab);

  int shiftTab[32];
  getShiftTab(W, bitlenTab, 32, shiftTab);
  
  Buf      bufA(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, aTab);
  Buf      bufI(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, iTab);
  Buf bufBitlen(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(byte)   * N, bitlenTab);
  
  delete[] aTab;
  delete[] iTab;
  delete[] bitlenTab;  

  double *bigTrig = genBigTrig(W, H);
  Buf bufBigTrig(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N, bigTrig);
  delete[] bigTrig;
  
  double *sins = genSin(H, W); // transposed W/H !
  Buf bufSins(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N / 2, sins);
  delete[] sins;

  double *trig1K = genSmallTrig1K();
  double *trig2K = genSmallTrig2K();

  Buf bufTrig1K(c, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2 * 1024, trig1K);
  Buf bufTrig2K(c, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2 * 2048, trig2K);
  
  delete[] trig1K;
  delete[] trig2K;
  
  Buf buf1(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N);
  Buf buf2(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(double) * N);

  Buf bufData (c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)  * N, data);
  Buf bufCarry(c, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(long) * N / 8);
  const uint zero = 0;
  Buf bufErr  (c, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &zero);

  fftPremul1K.setArgs(bufData, buf1, bufA, bufTrig1K);
  transposeA.setArgs(buf1, bufBigTrig);
  fft2Kt.setArgs(buf1, buf2, bufTrig2K);
  square2K.setArgs(buf2, bufSins);
  fft2K.setArgs(buf2, bufTrig2K);
  transposeB.setArgs(buf2, bufBigTrig);
  fft1Kt.setArgs(buf2, buf1, bufTrig1K);  
  carryA.setArgs(buf1, bufI, bufData, bufCarry, bufBitlen, bufErr);
  carryB.setArgs(bufData, bufCarry, bufBitlen);

  log("setup: %ld ms\n", timer.delta());

  constexpr int logStep   = 20000;
  float maxErr = 0;  
  uint rawErr = 0;
  
  for (int k = startK, nextLog = (startK / logStep + 1) * logStep; k < E - 2; ++k) {    
    if (k == nextLog) {
      nextLog += logStep;
      q.read( false, bufErr,  sizeof(uint), &rawErr);
      q.write(false, bufErr,  sizeof(uint), &zero);
      q.read(true, bufData, sizeof(int) * N, data);
      float err = rawErr * (1 / (float) (1 << 30));
      maxErr = std::max(err, maxErr);
      double msPerIter = timer.delta() * (1 / (double) logStep);
      u64 res = residue(N, W, data, shiftTab);      
      doLog(E, k, err, maxErr, msPerIter, res);
      
      if ((data[0] == 0 || data[0] == 2) && isAllZero(data + 1, N)) {
        if (k == E - 2) {
          if (data[0] == 0) { log("*****   M%d is prime!   *****\n", E); }
        } else {
          log("ERROR at iteration %d with %d\n", k, data[0]);
          break;
        }
      }
      fileSaver.save(data, k);
    }

    q.run(fftPremul1K,  SIZE / 4);
    q.run(transposeA,   SIZE / 16);
    q.run(fft2Kt,       SIZE / 8);
        
    q.run(square2K,     SIZE / 2);
      
    q.run(fft2K,        SIZE / 8);
    q.run(transposeB,   SIZE / 16);
    q.run(fft1Kt,       SIZE / 4);
    
    q.run(carryA, N / 16);
    q.run(carryB, N / 16);
  }
}

int main(int argc, char **argv) {
  logFiles[0] = stdout;

  int E = (argc >= 2) ? atoi(argv[1]) : 0;
  
  if (E < 35000000 || E > 78000000) {
    log("Usage: gpuowl <exponent>\n"
	"E.g. gpuowl 77000201\n"
	"Where <exponent> is a Mersenne exponent in the range 35'000'000 to 78'000'000\n"
	);
    return 0;
  }

  openLogFile(E);
  log("gpuOWL v0.1 GPU Lucas-Lehmer primality checker\n");
  
  int ret = doit(E);

  if (logFiles[1]) { fclose(logFiles[1]); }

  return ret;
}
