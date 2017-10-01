// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>
#include <memory>

namespace std {
template<> struct default_delete<FILE> {
  void operator()(FILE *f) {
    fprintf(stderr, "file closed\n");
    fclose(f);
  }
 };
}

FILE *open(const char *name, const char *mode) {
  FILE *f = fopen(name, mode);
  if (!f) { log("Can't open '%s' (mode '%s')\n", name, mode); }
  return f;
}

struct Header {
  const char *headerFormat = "OWL 1 %d %d %d %d %d %d\n";
  
  int E;
  int W;
  int H;
  int k;
  int sum;
  int nErrors;

  bool read(FILE *fi, int expectedE) {
    char buf[256];
    bool ok = fgets(buf, sizeof(buf), fi) &&
      sscanf(buf, headerFormat, &E, &k, &W, &H, &sum, &nErrors) == 6;
    assert(E == expectedE);
    return ok;
  }
};

class Checkpoint {
private:
  char fileNameSave[64], fileNamePrev[64], fileNameTemp[64];
  int E;

  // Header: "OWL <format-version> <exponent> <iteration> <width> <height> <sum> <nErrors>\n"
  const char *headerFormat = "OWL 1 %d %d %d %d %d %d\n";

  static int auxSum(int sum, const int *data, size_t size) {
    for (const int *p = data, *end = data + size; p < end; ++p) { sum += *p; }
    return sum;
  }

  static int checksum(size_t N, const int *data, const int *checkBits) {
    int sum = 0;
    sum = auxSum(sum, data, N);
    sum = auxSum(sum, checkBits, N);
    return sum;
  }
  
  bool write(const char *name, int W, int H, int k, const int *data, const int *checkBits, int nErrors) {
    int N = 2 * W * H;
    int sum = checksum(N, data, checkBits);
    char header[256];
    int n = snprintf(header, sizeof(header), headerFormat, E, k, W, H, sum, nErrors);
    assert(n >= 0 && u32(n) < sizeof(header));
    int dataSize = sizeof(int) * N;

    if (FILE *fo = open(name, "wb")) {
      bool ok = (fputs(header, fo) >= 0)
        && fwrite(data, dataSize, 1, fo)
        && fwrite(checkBits, dataSize, 1, fo);
      fclose(fo);
      if (ok) { return true; }
      log("File '%s': error writing\n", name);
    }
    return false;
  }
  
public:
  static bool readSize(int E, int *W, int *H) {
    char fileNameSave[64];
    snprintf(fileNameSave, sizeof(fileNameSave), "%d.owl", E);
    std::unique_ptr<FILE> fi{fopen(fileNameSave, "rb")};
    if (!fi) { return false; }
    Header header;
    if (!header.read(fi.get(), E)) { return false; }    
    *W = header.W;
    *H = header.H;
    return true;
  }

 Checkpoint(int iniE) : E(iniE) {
    snprintf(fileNameSave, sizeof(fileNameSave), "%d.owl", E);
    snprintf(fileNamePrev, sizeof(fileNamePrev), "%d-prev.owl", E);
    snprintf(fileNameTemp, sizeof(fileNameTemp), "%d-temp.owl", E);
  }
  
  bool load(int W, int H, int *startK, int *data, int *checkBits, int *nErrors) {
    *startK = 0;
    std::unique_ptr<FILE> fi{fopen(fileNameSave, "rb")};
    if (!fi) { return true; }

    Header header;
    if (!header.read(fi.get(), E)) { return false; }
    if (header.W != W || header.H != H) { return false; }

    int N = 2 * W * H;
    int dataSize = sizeof(int) * N;
    if (!fread(data, dataSize, 1, fi.get()) || !fread(checkBits, dataSize, 1, fi.get())) {
      log("File '%s': wrong size\n", fileNameSave);
      return false;
    }

    if (checksum(N, data, checkBits) != header.sum) {
      log("File '%s': wrong checksum\n", fileNameSave);
      return false;
    }

    *startK = header.k;
    return true;
  }
  
  void save(int W, int H, int k, int *data, int *checkBits, bool savePersist, int nErrors) {
    if (write(fileNameTemp, W, H, k, data, checkBits, nErrors)) {
      remove(fileNamePrev);
      rename(fileNameSave, fileNamePrev);
      rename(fileNameTemp, fileNameSave);      
    }
    if (savePersist) {
      char name[64];
      snprintf(name, sizeof(name), "%d.%d.owl", E, k);
      write(name, W, H, k, data, checkBits, nErrors);
    }
  }
};
