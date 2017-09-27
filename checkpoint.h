// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "common.h"

#include <cstdio>

FILE *open(const char *name, const char *mode) {
  FILE *f = fopen(name, mode);
  if (!f) { log("Can't open '%s' (mode '%s')\n", name, mode); }
  return f;
}

class Checkpoint {
private:
  char fileNameSave[64], fileNamePrev[64], fileNameTemp[64];
  int E, W, H;
  // Header: "\n<signature> <exponent> <iteration> <width> <height> <sum> <nErrors>\n"
  const char *headerFormat = "\nLL4 %d %d %d %d %d %d\n";

  bool write(const char *name, int k, const int *data, const int *checkBits, int nErrors) {
    if (FILE *fo = open(name, "wb")) {
      int N = 2 * W * H;
      int dataSize = sizeof(int) * N;
      int sum = checksum(checksum(0, data, N), checkBits, N);
      bool ok = fwrite(data, dataSize, 1, fo)
        && fwrite(checkBits, dataSize, 1, fo)
        && (fprintf(fo, headerFormat, E, k, W, H, sum, nErrors) > 0);
      fclose(fo);
      if (ok) { return true; }
      log("File '%s': error writing\n", name);
    }
    return false;
  }

  static int checksum(int sum, const int *data, size_t size) {
    for (const int *p = data, *end = data + size; p < end; ++p) { sum += *p; }
    return sum;
  }
  
public:
  Checkpoint(int iniE, int iniW, int iniH) : E(iniE), W(iniW), H(iniH) {
    snprintf(fileNameSave, sizeof(fileNameSave), "%d.owl", E);
    snprintf(fileNamePrev, sizeof(fileNamePrev), "%d-prev.owl", E);
    snprintf(fileNameTemp, sizeof(fileNameTemp), "%d-temp.owl", E);
  }
  
  bool load(int *startK, int *data, int *checkBits, int *nErrors) {
    *startK = 0;
    if (FILE *fi = fopen(fileNameSave, "rb")) {
      int N = 2 * W * H;
      int dataSize = sizeof(int) * N;
    
      if (!fread(data, dataSize, 1, fi) || !fread(checkBits, dataSize, 1, fi)) {
        fclose(fi);
        log("File '%s': wrong size\n", fileNameSave);
        return false;
      }

      int expectedSum = checksum(checksum(0, data, N), checkBits, N);
      int fileE, fileK, fileW, fileH, fileSum;
      int nRead = fscanf(fi, headerFormat, &fileE, &fileK, &fileW, &fileH, &fileSum, nErrors);
      fclose(fi);
      if (nRead != 6 || E != fileE || W != fileW || H != fileH || fileSum != expectedSum) {
        log("File '%s': invalid\n", fileNameSave);
        return false;
      }
      *startK = fileK;
    }
    return true;
  }
  
  void save(int k, int *data, int *checkBits, bool savePersist, int nErrors) {
    if (write(fileNameTemp, k, data, checkBits, nErrors)) {
      remove(fileNamePrev);
      rename(fileNameSave, fileNamePrev);
      rename(fileNameTemp, fileNameSave);      
    }
    if (savePersist) {
      char name[64];
      snprintf(name, sizeof(name), "%d.%d.owl", E, k);
      write(name, k, data, checkBits, nErrors);
    }
  }
};
