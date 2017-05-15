// Copyright (C) 2017 Mihai Preda.

#include "common.h"

#include <cstdio>

FILE *open(const char *name, const char *mode) {
  FILE *f = fopen(name, mode);
  if (!f) { log("Can't open '%s' (mode '%s')\n", name, mode); }
  return f;
}

class Checkpoint {
private:
  char fileNameSave[64];
  int E, W, H;
  // Header: "\nLL2 <exponent> <iteration> <width> <height> <offset> <sum> \n"
  const char *headerFormat = "\nLL2 %d %d %d %d %d %d\n";

  bool write(int size, const void *data, const char *name) {
    if (FILE *fo = open(name, "wb")) {
      int nWritten = fwrite(data, size, 1, fo);
      fclose(fo);      
      if (nWritten == 1) {
        return true;
      } else {
        log("Error writing file '%s'\n", name);
      }
    }
    return false;
  }

  int prepareHeader(int *data, int k) {
    int N = 2 * W * H;
    int n = snprintf((char *) (data + N), 128, headerFormat, E, k, W, H, 0, checksum(data, N));
    assert(n < 128);
    return n;
  }

  static int checksum(int *data, unsigned words) {
    int sum = 0;
    for (int *p = data, *end = data + words; p < end; ++p) { sum += *p; }
    return sum;
  }

public:
  Checkpoint(int iniE, int iniW, int iniH) : E(iniE), W(iniW), H(iniH) {
    snprintf(fileNameSave, sizeof(fileNameSave), "c%d.ll", E);
  }
  
  bool load(int *data, int *startK) {
    FILE *fi = open(fileNameSave, "rb");    
    if (!fi) { return true; }
    
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
  
  void save(int *data, int k, bool savePersist, u64 residue) {
    char fileTemp[64], filePrev[64];
    snprintf(fileTemp, sizeof(fileTemp), "b%d.ll", E);
    snprintf(filePrev, sizeof(filePrev), "t%d.ll", E);

    int headerSize = prepareHeader(data, k);
    const int totalSize = sizeof(int) * 2 * W * H + headerSize;
    if (write(totalSize, data, fileTemp)) {
      remove(filePrev);
      rename(fileNameSave, filePrev);
      rename(fileTemp, fileNameSave);      
    }
    if (savePersist) {
      char name[64];
      snprintf(name, sizeof(name), "s%d.%d.%016llx.ll", E, k, (unsigned long long) residue);
      write(totalSize, data, name);
    }
  }
};
