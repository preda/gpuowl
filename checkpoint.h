// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017 Mihai Preda.

#pragma once

#include "state.h"
#include "common.h"

#include <string>

using std::string;

class Checkpoint {
private:
  struct HeaderV3 {
    // <exponent> <iteration> <nErrors> <check-step>
    static constexpr const char *HEADER = "OWL 3 %d %d %d %d\n";

    int E, k, nErrors, checkStep;

    bool read(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors, &checkStep) == 4; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors, checkStep) > 0); }
  };

  struct HeaderV2 {
    // <exponent> <iteration> <nErrors>
    static constexpr const char *HEADER = "OWL 2 %d %d %d\n";

    int E, k, nErrors;

    bool read(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors) == 3; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors) > 0); }
  };

  struct HeaderV1 {
    // <exponent> <iteration> <width> <height> <sum> <nErrors>
    static constexpr const char *HEADER = "OWL 1 %d %d %d %d %d %d\n";
    
    int E, k, W, H, sum, nErrors;

    bool read(const char *line) { return sscanf(line, HEADER, &E, &k, &W, &H, &sum, &nErrors) == 6; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, W, H, sum, nErrors) > 0); }
  };

  static bool write(FILE *fo, const auto &vect) { return fwrite(&vect[0], vect.size() * sizeof(vect[0]), 1, fo); }
  static bool  read(FILE *fi, int n, auto &vect) {
    vect.resize(n);
    return fread(&vect[0], n * sizeof(vect[0]), 1, fi);
  }
  
  static bool write(const string &name, const CompactState &compact, int k, int nErrors, int checkStep) {
    int E = compact.E;
    int nWords = (E - 1) / 32 + 1;
    assert(int(compact.data.size()) == nWords && int(compact.check.size()) == nWords);
    HeaderV3 header{E, k, nErrors, checkStep};
    auto fo(open(name, "wb"));
    return fo
      && header.write(fo.get())
      && write(fo.get(), compact.data)
      && write(fo.get(), compact.check);
  }

  static std::string fileName(int E) { return std::to_string(E) + ".owl"; }
  
public:
  
  static bool load(int E, int W, int H, State *state, int *k, int *nErrors, int *checkStep) {
    *k = 0;
    *nErrors = 0;
    *checkStep = 500;
    
    auto fi{open(fileName(E), "rb", false)};
    if (!fi) {
      state->reset();
      return true;
    }

    char line[256];
    if (!fgets(line, sizeof(line), fi.get())) { return false; }

    int N = 2 * W * H;
    assert(state->N == N);

    {
      HeaderV3 header;
      if (header.read(line)) {
        if (header.E != E) { return false; }
        {
          int nWords = (E - 1) / 32 + 1;
          std::vector<u32> data, check;
          if (!read(fi.get(), nWords, data) || !read(fi.get(), nWords, check)) { return false; }
          CompactState(E, std::move(data), std::move(check)).expandTo(state, true, W, H, E);
        }
        *k = header.k;
        *nErrors = header.nErrors;
        *checkStep = header.checkStep;
        return true;
      }
    }
    
    {
      HeaderV2 header;
      if (header.read(line)) {
        if (header.E != E) { return false; }
        {
          int nWords = (E - 1) / 32 + 1;
          std::vector<u32> data, check;
          if (!read(fi.get(), nWords, data) || !read(fi.get(), nWords, check)) { return false; }
          CompactState(E, std::move(data), std::move(check)).expandTo(state, true, W, H, E);
        }
        *k = header.k;
        *nErrors = header.nErrors;
        *checkStep = 1000;
        return true;
      }
    }
    
    {
      HeaderV1 header;
      if (header.read(line)) {
        if (header.E != E || header.W != W || header.H != H) { return false; }
        *k = header.k;
        *nErrors = header.nErrors;
        *checkStep = 1000;
        int dataSize = sizeof(int) * N;
        return fread(state->data.get(), dataSize, 1, fi.get()) && fread(state->check.get(), dataSize, 1, fi.get());
      }
    }
    
    return false;
  }
  
  static void save(const CompactState &compact, int k, int nErrors, int checkStep) {
    int E = compact.E;
    string saveFile = fileName(E);
    string strE = std::to_string(E);
    string tempFile = strE + "-temp.owl";
    string prevFile = strE + "-prev.owl";
    
    if (write(tempFile, compact, k, nErrors, checkStep)) {
      remove(prevFile.c_str());
      rename(saveFile.c_str(), prevFile.c_str());
      rename(tempFile.c_str(), saveFile.c_str());
    }
    const int saveStep = 10'000'000;
    if (k && (k % saveStep == 0)) {
      string persistFile = strE + "." + std::to_string(k) + ".owl";
      write(persistFile, compact, k, nErrors, checkStep);
    }
  }
};
