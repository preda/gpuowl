// gpuOWL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#include "clwrap.h"

#include <memory>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <ctime>

#define KERNEL(program, name, ...) cl_kernel name = makeKernel(program, #name); setArgs(name, __VA_ARGS__)

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

const int EXP_MIN_2M = 20000000, EXP_MAX_2M = 40000000, EXP_MIN_4M = 35000000, EXP_MAX_4M = 78000000;

#define VERSION "0.1"

const char *AGENT = "gpuowl v" VERSION;

const unsigned BUF_CONST = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
const unsigned BUF_RW    = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

void genBitlen(int E, int W, int H, double *aTab, double *iTab, byte *bitlenTab) {
  double *pa = aTab;
  double *pi = iTab;
  byte   *pb = bitlenTab;

  auto iN = 1 / (long double) (2 * W * H);
  
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

cl_mem genSmallTrig2K(cl_context context) {
  int size = 2 * 4 * 512;
  double *tab = new double[size]();
  double *p   = tab + 2 * 8;
  p = smallTrigBlock(  8, 8, p);
  p = smallTrigBlock( 64, 8, p);
  p = smallTrigBlock(512, 4, p);
  assert(p - tab == size);
  
  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

cl_mem genSmallTrig1K(cl_context context) {
  int size = 2 * 4 * 256;
  double *tab = new double[size]();
  double *p   = tab + 2 * 4;
  p = smallTrigBlock(  4, 4, p);
  p = smallTrigBlock( 16, 4, p);
  p = smallTrigBlock( 64, 4, p);
  p = smallTrigBlock(256, 4, p);
  assert(p - tab == size);

  cl_mem buf = makeBuf(context, BUF_CONST, sizeof(double) * size, tab);
  delete[] tab;
  return buf;
}

bool isAllZero(int *p, int size) {
  for (int *end = p + size; p < end; ++p) { if (*p) { return false; } }
  return true;
}

void getShiftTab(int W, byte *bitlenTab, int *shiftTab) {
  int sh = 0;
  for (int i = 0; i < 32; ++i) {
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

#ifdef __GNUC__
void log(const char *fmt, ...) __attribute__ ((format(printf, 1, 2)));
#endif

void log(const char *fmt, ...) {
  va_list va;
  for (FILE **pf = logFiles; *pf; ++pf) {
    va_start(va, fmt);
    vfprintf(*pf, fmt, va);
    va_end(va);
  }
}

int checksum(int *data, unsigned words) {
  int sum = 0;
  for (int *p = data, *end = data + words; p < end; ++p) { sum += *p; }
  return sum;
}

class FileSaver {
private:
  char fileNameSave[64];
  int E, W, H;
  // Header: "\nLL2 <exponent> <iteration> <width> <height> <offset> <sum> \n"
  const char *headerFormat = "\nLL2 %d %d %d %d %d %d\n";
  
public:
  FileSaver(int iniE, int iniW, int iniH) : E(iniE), W(iniW), H(iniH) {
    snprintf(fileNameSave, sizeof(fileNameSave), "c%d.ll", E);
  }

  bool loadLL1(int *data, int *startK) {
    char fileName[64];
    snprintf(fileName, sizeof(fileName), "save-%d.bin", E);
    FILE *fi = fopen(fileName, "rb");
    if (!fi) { return true; }

    int saveE, saveK, saveW, saveH, unused;
    bool ok = false;
    if (fscanf(fi, "LL1 %10d %10d %10d %10d %10d    \n\032", &saveE, &saveK, &saveW, &saveH, &unused) == 5 &&
        E == saveE && W == saveW && H == saveH && unused == 0 &&
        fread(data, sizeof(int) * (2 * W * H), 1, fi) == 1) {
      *startK = saveK;
      ok = true;
    }
    fclose(fi);
    if (!ok) { log("Wrong '%s' file, please move it out of the way.\n", fileNameSave); }
    return ok;
  }
  
  bool load(int *data, int *startK) {
    FILE *fi = fopen(fileNameSave, "rb");    
    if (!fi) { 
      log("Checkpoint file '%s' not found.\n", fileNameSave);
      return loadLL1(data, startK);
    }
    
    int N = 2 * W * H;
    int wordsSize = sizeof(int) * N;
    int n = fread(data, 1, wordsSize + 128, fi);
    fclose(fi);

    if (n < wordsSize || n >= wordsSize + 128) {
      log("File '%s' has invalid size (%d)\n", fileNameSave, n);
      return false;
    }
        
    int fileE, fileK, fileW, fileH, fileOffset, fileSum;
    char *header = (char *) (data + N);
    header[n - wordsSize] = 0;
    
    if (sscanf(header, headerFormat, &fileE, &fileK, &fileW, &fileH, &fileOffset, &fileSum) != 6 ||
        !(E == fileE && W == fileW && H == fileH && 0 == fileOffset)) {
      log("File '%s' has wrong tailer '%s'\n", fileNameSave, header);
      return false;
    }
    
    if (fileSum != checksum(data, N)) {
      log("File '%s' has wrong checksum (expected %d got %d)\n", fileNameSave, fileSum, checksum(data, N));
      return false;
    }
    
    *startK = fileK;
    return true;
  }

  bool write(int size, const void *data, const char *name) {
    FILE *fo = fopen(name, "wb");
    if (fo) {
      int nWritten = fwrite(data, size, 1, fo);
      fclose(fo);      
      if (nWritten == 1) {
        return true;
      } else {
        log("Error writing file '%s'\n", name);
      }
    } else {
      log("Can't write file '%s'\n", name);
    }
    return false;
  }

  int prepareHeader(int *data, int k) {
    int N = 2 * W * H;
    int n = snprintf((char *) (data + N), 128, headerFormat, E, k, W, H, 0, checksum(data, N));
    assert(n < 128);
    return n;
  }
  
  void save(int *data, int k) {
    char fileTemp[64], filePrev[64];
    snprintf(fileTemp, sizeof(fileTemp), "b%d.ll", E);
    snprintf(filePrev, sizeof(filePrev), "t%d.ll", E);

    int headerSize = prepareHeader(data, k);    
    if (write(sizeof(int) * 2 * W * H + headerSize, data, fileTemp)) {
      remove(filePrev);
      rename(fileNameSave, filePrev);
      rename(fileTemp, fileNameSave);      
    }
  }

  void savePersist(int *data, int k, u64 residue) {
    int headerSize = prepareHeader(data, k);    
    char name[64];
    snprintf(name, sizeof(name), "s%d.%d.%016llx.ll", E, k, (unsigned long long) residue);
    write(sizeof(int) * 2 * W * H + headerSize, data, name);
  }
};

void doLog(int E, int k, float err, float maxErr, double msPerIter, u64 res) {
  const float percent = 100 / (float) (E - 2);
  int etaMins = (E - 2 - k) * msPerIter * (1 / (double) 60000) + .5;
  int days  = etaMins / (24 * 60);
  int hours = etaMins / 60 % 24;
  int mins  = etaMins % 60;
  
  log("%08d / %08d [%.2f%%], ms/iter: %.3f, ETA: %dd %02d:%02d; %016llx error %g (max %g)\n",
      k, E, k * percent, msPerIter, days, hours, mins, (unsigned long long) res, err, maxErr);
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
  *AID = 0;
  while (true) {
    if (fscanf(fi, "%255s\n", line) < 1) { break; }
    if (((sscanf(line, "%11[^=]=%32[0-9a-fA-F],%d,%*d,%*d", kind, AID, &exp) == 3
          && (!strcmp(kind, "Test") || !strcmp(kind, "DoubleCheck")))
         || sscanf(line, "Test=%d", &exp) == 1)
        && exp >= EXP_MIN_2M && exp <= EXP_MAX_4M) {
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
  fprintf(fo, "M( %d )%c, 0x%016llx, offset = 0, n = %dK, %s, AID: %s\n",
          E, isPrime ? 'P' : 'C', (unsigned long long) residue, 4096, AGENT, AID);
  fclose(fo);
  return true;
}

void setupExponentBufs(cl_context context, int E, int W, int H, cl_mem *pBufA, cl_mem *pBufI, cl_mem *pBufBitlen, int *shiftTab) {
  int N = 2 * W * H;
  double *aTab    = new double[N];
  double *iTab    = new double[N];
  byte *bitlenTab = new byte[N];
  
  genBitlen(E, W, H, aTab, iTab, bitlenTab);
  getShiftTab(W, bitlenTab, shiftTab);
  
  *pBufA      = makeBuf(context, BUF_CONST, sizeof(double) * N, aTab);
  *pBufI      = makeBuf(context, BUF_CONST, sizeof(double) * N, iTab);
  *pBufBitlen = makeBuf(context, BUF_CONST, sizeof(byte)   * N, bitlenTab);

  delete[] aTab;
  delete[] iTab;
  delete[] bitlenTab;
}

bool checkPrime(int H, cl_context context, cl_program program, cl_queue q, cl_mem bufTrig1K, cl_mem bufTrig2K,
                int E, int logStep, int saveStep, bool doTimeKernels, bool *outIsPrime, u64 *outResidue) {
  const int W = 1024;
  const int N = 2 * W * H;
  assert(H == 2048);

  int startK = 0;
  int *data = new int[N + 32]();
  int *saveData = new int[N];
  data[0]     = 4; // LL root.
  
  FileSaver fileSaver(E, W, H);
  if (!fileSaver.load(data, &startK)) { return false; }
  
  log("LL FFT %dK (%d*%d*2) of %d (%.2f bits/word) at iteration %d\n",
      N / 1024, W, H, E, E / (double) N, startK);
    
  Timer timer;
  
  cl_mem bufA, bufI, bufBitlen;
  int shiftTab[32];
  setupExponentBufs(context, E, W, H, &bufA, &bufI, &bufBitlen, shiftTab);

  double *bigTrig = genBigTrig(W, H);
  cl_mem bufBigTrig = makeBuf(context, BUF_CONST, sizeof(double) * N, bigTrig);
  delete[] bigTrig;
  
  double *sins = genSin(H, W); // transposed W/H !
  cl_mem bufSins = makeBuf(context, BUF_CONST, sizeof(double) * N / 2, sins);
  delete[] sins;

  cl_mem buf1     = makeBuf(context, BUF_RW, sizeof(double) * N);
  cl_mem buf2     = makeBuf(context, BUF_RW, sizeof(double) * N);
  cl_mem bufCarry = makeBuf(context, BUF_RW, sizeof(long)   * N / 8);

  const unsigned zero = 0;
  cl_mem bufErr   = makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &zero);
  cl_mem bufData  = makeBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, data);

  KERNEL(program, fftPremul1K, bufData, buf1, bufA, bufTrig1K);
  KERNEL(program, transpose1K, buf1, bufBigTrig);
  KERNEL(program, fft2K_1K, buf1, buf2, bufTrig2K);
  
  KERNEL(program, csquare2K, buf2, bufSins);

  KERNEL(program, fft2K, buf2, bufTrig2K);
  KERNEL(program, mtranspose2K, buf2, bufBigTrig);
  KERNEL(program, fft1K_2K, buf2, buf1, bufTrig1K);
  
  KERNEL(program, carryA, buf1, bufI, bufData, bufCarry, bufBitlen, bufErr);
  KERNEL(program, carryB_2K,          bufData, bufCarry, bufBitlen, bufErr);

  log("OpenCL setup: %ld ms\n", (long) timer.delta());

  float maxErr  = 0;  
  u64 res;
  int k = startK;
  int saveK = 0;
  bool isCheck = false;

  const int nKernels = 9;
  TimeCounter *counters[nKernels] = {nullptr};
  MicroTimer microTimer;
  if (doTimeKernels) {
    for (int i = 0; i < nKernels; ++i) { counters[i] = new TimeCounter(&microTimer); }
  }
  
  do {
    microTimer.delta();
    for (int nextLog = std::min((k / logStep + 1) * logStep, E - 2); k < nextLog; ++k) {
      run(q, fftPremul1K,  N / 8,  counters[0]);
      run(q, transpose1K,  N / 32, counters[1]);
      run(q, fft2K_1K,     N / 16, counters[2]);
      
      run(q, csquare2K,    N / 4,  counters[3]);
      
      run(q, fft2K,        N / 16, counters[4]);
      run(q, mtranspose2K, N / 32, counters[5]);
      run(q, fft1K_2K,     N / 8,  counters[6]);
        
      run(q, carryA,       N / 16, counters[7]);
      run(q, carryB_2K,    N / 16, counters[8]);
    }
    
    unsigned rawErr = 0;
    read(q,  false, bufErr,  sizeof(unsigned), &rawErr);
    write(q, false, bufErr,  sizeof(unsigned), &zero);
    read(q,  true,  bufData, sizeof(int) * N, data);
    float err = rawErr * (1 / (float) (1 << 30));
    float prevMaxErr = maxErr;
    maxErr = std::max(err, maxErr);
    double msPerIter = timer.delta() * (1 / (double) logStep);
    res = residue(N, W, data, shiftTab);      
    doLog(E, k, err, maxErr, msPerIter, res);
    fileSaver.save(data, k);
    if (k / saveStep != (k - logStep) / saveStep) {
      fileSaver.savePersist(data, k, res);
    }

    if (doTimeKernels) {
      const char *name[nKernels] = {
        "fftPremul1K", "transpose1K", "fft2K_1K", "csquare2K", "fft2K", "mtranspose2K", "fft1K_2K", "carryA", "carryB_2K",
      };
      u64 total = 0;
      for (int i = 0; i < nKernels; ++i) { total += counters[i]->get(); }
      const float iLogStep = 1 / (float) logStep;
      const float iTotal   = 1 / (float) total;
      for (int i = 0; i < nKernels; ++i) {
        u64 c = counters[i]->get();
        counters[i]->reset();
        log("  %-12s %.1fus, %2.1f%%\n", name[i], c * iLogStep, c * 100 * iTotal);
      }
      log("  %-12s %.1fus\n", "Total", total * iLogStep);
    }
    
    if (err >= .5f) {
      log("Error %g is way too large, stopping.\n", err);
      return false;
    }

    if (isCheck) {
      isCheck = false;
      if (!memcmp(data, saveData, sizeof(int) * N)) {
        log("Consistency checked OK, continuing.\n");
      } else {
        log("Consistency check FAILED, something is wrong, stopping.\n");
        return false;
      }
    } else if (prevMaxErr > 0 && err > 1.166f * prevMaxErr) {
      log("Error jump by %.2f%%, doing a consistency check.\n", (err / prevMaxErr - 1) * 100);      
      isCheck = true;
      write(q, true, bufData, sizeof(int) * N, saveData);
      k = saveK;
    }
    
    memcpy(saveData, data, sizeof(int) * N);
    saveK = k;
  } while (k < E - 2);

  *outIsPrime = isAllZero(data, N);
  *outResidue = res;
  return true;
}

// return false to stop.
bool parseArgs(int argc, char **argv, const char **extraOpts, int *logStep, int *saveStep, int *forceDevice, bool *timeKernels) {
  const int DEFAULT_LOGSTEP = 20000;
  *logStep  = DEFAULT_LOGSTEP;
  *saveStep = 0;
  *timeKernels = false;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      log("Command line options:\n"
          "-cl <CL compiler options>\n"
          "    e.g. -cl -save-temps or -cl -save-temps=prefix or -cl -save-temps=folder/ to save ISA code\n"
          "-logstep  <N> : to log every <N> iterations (default %d)\n"
          "-savestep <N> : to persist checkpoint every <N> iterations (default 500*logstep == %d)\n"
          "-time kernels : to benchmark kernels\n"
          "-device   <N> : select specific device among:\n", *logStep, 500 * *logStep);

      cl_device_id devices[16];
      int ndev = getDeviceIDs(false, 16, devices);
      for (int i = 0; i < ndev; ++i) {
        char info[256];
        getDeviceInfo(devices[i], sizeof(info), info);
        log("    %d : %s\n", i, info);
      }

      log("\nFiles used by gpuOwL:\n"
          "    - worktodo.txt : contains exponents to test \"Test=N\", one per line\n"
          "    - results.txt : contains LL results\n"
          "    - cN.ll : the most recent checkpoint for exponent <N>; will resume from here\n"
          "    - tN.ll : the previous checkpoint, to be used if cN.ll is lost or corrupted\n"
          "    - bN.ll : a temporary checkpoint that is renamed to cN.ll once successfully written\n"
          "    - sN.iteration.residue.ll : a persistent checkpoint at the given iteration\n");

      log("\nThe lines in worktodo.txt must be of one of these forms:\n"
          "Test=70100200\n"
          "Test=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1\n"
          "DoubleCheck=3181F68030F6BF3DCD32B77337D5EF6B,70100200,75,1\n"
          "Test=0,70100200,0,0\n");
      
      return false;
    } else if (!strcmp(arg, "-cl")) {
      if (i < argc - 1) {
        *extraOpts = argv[++i];
      } else {
        log("-cl expects options string to pass to CL compiler\n");
        return false;
      }
    } else if (!strcmp(arg, "-logstep")) {
      if (i < argc - 1) {
        *logStep = atoi(argv[++i]);
        if (*logStep <= 0) {
          log("invalid -logstep '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-logstep expects <N> argument\n");
        return false;
      }
    } else if (!strcmp(arg, "-savestep")) {
      if (i < argc - 1) {
        *saveStep = atoi(argv[++i]);
        if (*saveStep <= 0) {
          log("invalid -savestep '%s'\n", argv[i]);
          return false;
        }
      } else {
        log("-savestep expects <N> argument\n");
        return false;
      }
    } else if(!strcmp(arg, "-time")) {
      if (i < argc - 1 && !strcmp(argv[++i], "kernels")) {
        *timeKernels = true;
      } else {
        log("-time expects 'kernels'\n");
        return false;
      }
    } else if (!strcmp(arg, "-device")) {
      if (i < argc - 1) {
        *forceDevice = atoi(argv[++i]);
        int nDevices = getNumberOfDevices();
        if (*forceDevice < 0 || *forceDevice >= nDevices) {
          log("invalid -device %d (must be between [0, %d]\n", *forceDevice, nDevices - 1);
          return false;
        }        
      } else {
        log("-device expects <N> argument\n");
        return false;
      }
    } else {
      log("Argument '%s' not understood\n", arg);
      return false;
    }
  }

  if (!*saveStep) { *saveStep = 500 * *logStep; }
  if (*saveStep < *logStep || *saveStep % *logStep) {
    log("It is recommended to use a persist step that is a multiple of the log step (you provided %d, %d)\n", *saveStep, *logStep);
  }
  
  return true;
}

int main(int argc, char **argv) {
  logFiles[0] = stdout;
  FILE *logf = fopen("gpuowl.log", "a");

#ifdef _DEFAULT_SOURCE
  if (logf) { setlinebuf(logf); }
#endif

  logFiles[1] = logf;
  time_t t = time(NULL);
  log("gpuOwL v" VERSION " GPU Lucas-Lehmer primality checker; %s", ctime(&t));


  const char *extraOpts = "";
  int forceDevice = -1;
  int logStep = 0;
  int saveStep = 0;
  bool doTimeKernels = false;
  if (!parseArgs(argc, argv, &extraOpts, &logStep, &saveStep, &forceDevice, &doTimeKernels)) { return 0; }
  
  cl_device_id device;
  if (forceDevice >= 0) {
    cl_device_id devices[16];
    int n = getDeviceIDs(false, 16, devices);
    assert(n > forceDevice);
    device = devices[forceDevice];
  } else {
    int n = getDeviceIDs(true, 1, &device);
    if (n <= 0) {
      log("No GPU device found. See --help for how to select a specific device.\n");
      return 8;
    }
  }
  
  char info[256];
  getDeviceInfo(device, sizeof(info), info);
  log("%s\n", info);
  
  log("Will log every %d iterations, and persist checkpoint every %d iterations.\n", logStep, saveStep);
  
  cl_context context = createContext(device);

  cl_queue queue = makeQueue(device, context);

  cl_program program = compile(device, context, "gpuowl.cl", extraOpts);
  if (!program) { exit(1); }
  
  cl_mem bufTrig1K = genSmallTrig1K(context);
  cl_mem bufTrig2K = genSmallTrig2K(context);
  
  bool someSuccess = false;
  while (true) {
    char AID[64];
    int E = worktodoReadExponent(AID);
    if (E <= 0) {
      if (!someSuccess) {
        log("Please provide a 'worktodo.txt' file containing LL exponents to test in the range %d to %d.\n"
            "See http://www.mersenne.org/manual_assignment/\n",
            EXP_MIN_4M, EXP_MAX_4M);
      }
      break;
    }
    someSuccess = true;

    bool isPrime;
    u64 res;
    if (checkPrime(2048, context, program, queue, bufTrig1K, bufTrig2K, E, logStep, saveStep, doTimeKernels, &isPrime, &res)) {
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
  
  release(program);
  release(context);
  
  log("\nBye\n");
  if (logf) { fclose(logf); }
}
