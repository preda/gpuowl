// gpuOWL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"

#include <memory>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>

#define K(program, name) Kernel name(program, #name);

#ifndef M_PIl
#define M_PIl		3.141592653589793238462643383279502884L
#endif

typedef unsigned char byte;
typedef int64_t i64;
typedef uint64_t u64;

constexpr int MIN_EXP = 35000000;
constexpr int MAX_EXP = 78000000;
const char *AGENT = "gpuowl v0.1";

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
  
  log("%08d / %08d [%.2f%%], ms/iter: %.3f, ETA: %dd %02d:%02d; %016lx error %g (max %g)\n",
      k, E, k * percent, msPerIter, days, hours, mins, res, err, maxErr);
}

int worktodoReadExponent(char *AID) {
  FILE *fi = fopen("worktodo.txt", "r");
  if (!fi) {
    log("No 'worktodo.txt' file found\n");
    return 0;
  }

  char line[256];
  char kind[32];
  int exp;
  int ret = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    if (sscanf(line, "%11[^=]=%32[0-9a-fA-F],%d,%*d,%*d", kind, AID, &exp) == 3 &&
        (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")) &&
        exp >= MIN_EXP && exp <= MAX_EXP) {
      ret = exp;
      break;
    } else {
      log("worktodo.txt line '%s' skipped\n", line);
    }
  }
  fclose(fi);
  return ret;
}

bool worktodoGetLinePos(int E, int *begin, int *end) {
  FILE *fi = fopen("worktodo.txt", "r");
  if (!fi) {
    log("No 'worktodo.txt' file found\n");
    return false;
  }

  char line[256];
  char kind[32];
  int exp;
  bool ret = false;
  long p1 = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    long p2 = ftell(fi);
    if (sscanf(line, "%11[^=]=%*32[0-9a-fA-F],%d,%*d,%*d", kind, &exp) == 2 &&
        (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")) &&
        exp == E) {
      *begin = p1;
      *end = p2;
      ret = true;
      break;
    }
    p1 = p2;
  }
  fclose(fi);
  return ret;
}

bool worktodoDelete(int begin, int end) {
  assert(begin >= 0 && end > begin);
  FILE *fi = fopen("worktodo.txt", "r");
  char buf[64 * 1024];
  int n = fread(buf, 1, sizeof(buf), fi);
  if (n == sizeof(buf) || end > n) {
    fclose(fi);
    return false;
  }
  memmove(buf + begin, buf + end, n - end);
  fclose(fi);
  
  FILE *fo = fopen("worktodo-tmp.tmp", "w");
  int newSize = begin + n - end;
  bool ok = (newSize == 0) || (fwrite(buf, newSize, 1, fo) == 1);
  fclose(fo);
  return ok &&
    (rename("worktodo.txt",     "worktodo.bak") == 0) &&
    (rename("worktodo-tmp.tmp", "worktodo.txt") == 0);
}

bool writeResult(int E, bool isPrime, u64 residue, const char *AID) {
  FILE *fo = fopen("results.txt", "a");
  if (!fo) { return false; }
  fprintf(fo, "M( %d )%c, 0x%lx, offset = 0, n = %dK, %s, AID: %s\n", E, isPrime ? 'P' : 'C', residue, 4096, AGENT, AID);
  fclose(fo);
  return true;
}

bool checkPrime(int E, bool *outIsPrime, u64 *outResidue) {
  constexpr int W = 1024;
  constexpr int H = 2048;
  constexpr int SIZE  = W * H;
  constexpr int N = 2 * SIZE;
  
  int startK = 0;
  int *data = new int[N]();
  data[0] = 4; // LL root.

  FileSaver fileSaver(E, N, W, H);
  if (!fileSaver.load(data, &startK)) { return false; }
    
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
  const unsigned zero = 0;
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

  constexpr int logStep = 20000;
  float maxErr = 0;  
  u64 res;

  int k = startK;
  do {
    for (int nextLog = std::min((k / logStep + 1) * logStep, E - 2); k < nextLog; ++k) {
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

    unsigned rawErr = 0;
    q.read( false, bufErr, sizeof(unsigned), &rawErr);
    q.write(false, bufErr, sizeof(unsigned), &zero);
    q.read( true, bufData, sizeof(int) * N, data);
    float err = rawErr * (1 / (float) (1 << 30));
    maxErr = std::max(err, maxErr);
    double msPerIter = timer.delta() * (1 / (double) logStep);
    res = residue(N, W, data, shiftTab);      
    doLog(E, k, err, maxErr, msPerIter, res);
    fileSaver.save(data, k);
  } while (k < E - 2);

  *outIsPrime = isAllZero(data, N);
  *outResidue = res;
  return true;
}

int main(int argc, char **argv) {
  logFiles[0] = stdout;
  FILE *logf = fopen("gpuowl.log", "a");

#ifdef _DEFAULT_SOURCE
  if (logf) { setlinebuf(logf); }
#endif

  logFiles[1] = logf;
  log("gpuOwL v0.1 GPU Lucas-Lehmer primality checker\n");

  const char *extraOpts = "";
  if (argc > 1) {
    if (!strcmp(argv[1], "-cl") && argc > 2) {
      extraOpts = argv[2];      
    } else if (!strcmp(argv[1], "-h")) {
      log("Command line options:\n-cl -save-temps : to save the compiled ISA\n\n");
    }
  }
  
  cl_device_id device = getDevice();
  char deviceName[256];
  getDeviceName(device, deviceName, sizeof(deviceName));
  log("%s\n", deviceName);

  cl_context context = createContext(device);

  cl_program program = compile(device, context, "gpuowl.cl", extraOpts);
  
  bool someSuccess = false;
  while (true) {
    char AID[64];
    int E = worktodoReadExponent(AID);
    if (E <= 0) {
      if (!someSuccess) {
        log("See http://www.mersenne.org/manual_assignment/\n"
            "Please provide a 'worktodo.txt' file containing GIMPS manual test LL assignements "
            "in the range %d to %d; e.g.\n\n"            
            "Test=3181F68030F6BF3DCD32B77337D5EF6B,71561261,75,1\n"
            "DoubleCheck=3181F68030F6BF3DCD32B77337D5EF6B,71561261,75,1\n",
            MIN_EXP, MAX_EXP);
      }
      break;
    }
    someSuccess = true;

    bool isPrime;
    u64 res;
    if (checkPrime(E, &isPrime, &res)) {
      if (isPrime) { log("*****   M%d is prime!   *****\n", E); }
      
      int lineBegin, lineEnd;
      if (!writeResult(E, isPrime, res, AID) ||
          !worktodoGetLinePos(E, &lineBegin, &lineEnd) ||
          !worktodoDelete(lineBegin, lineEnd)) {
        break;
      }
    } else {
      break;
    }
  }

  release(context);
  log("Bye\n");
  if (logf) { fclose(logf); }
}
