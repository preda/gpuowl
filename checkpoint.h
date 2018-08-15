#pragma once

#include "common.h"

#include <vector>
#include <cassert>

struct LoadResult {
  bool ok;
  int k;
  int blockSize;
  int nErrors;
  vector<u32> bits;
  u64 res64;
};

class Checkpoint {
private:
  static constexpr const int BLOCK_SIZE = 200;

  struct HeaderTF1 {
    // Exponent, bitHi, classDone, classTotal.
    static constexpr const char *HEADER = "OWL TF 1 %d %d %d %d\n";
    int E;
    int bitHi;
    int classDone, classTotal;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, bitHi, classDone, classTotal) > 0; }
    bool parse(FILE *fi) { return fscanf(fi, HEADER, &E, &bitHi, &classDone, &classTotal) == 4; }    
  };
  
  struct HeaderV5 {
    static constexpr const char *HEADER_R = R"(OWL 5
Comment: %255[^
]
Type: PRP
Exponent: %d
Iteration: %d
PRP-block-size: %d
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

    static constexpr const char *HEADER_W = R"(OWL 5
Comment: %s
Type: PRP
Exponent: %d
Iteration: %d
PRP-block-size: %d
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

    int E, k, blockSize;
    int nErrors;
    u64 res64;
    string comment;
    
    bool parse(FILE *fi) {
      char buf[256];
      bool ok = fscanf(fi, HEADER_R, buf, &E, &k, &blockSize, &res64, &nErrors) == 6;
      if (ok) { comment = buf; }
      return ok;
    }
    
    bool write(FILE *fo) { return (fprintf(fo, HEADER_W, comment.c_str(), E, k, blockSize, res64, nErrors) > 0); }
  };
  
  struct HeaderV4 {
    // <exponent> <iteration> <nErrors> <check-step> <checksum>
    static constexpr const char *HEADER = "OWL 4 %d %d %d %d %016llx\n";

    int E, k, nErrors, checkStep;
    u64 checksum;

    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors, &checkStep, &checksum) == 5; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors, checkStep, checksum) > 0); }
  };
  
  struct HeaderV3 {
    // <exponent> <iteration> <nErrors> <check-step>
    static constexpr const char *HEADER = "OWL 3 %d %d %d %d\n";

    int E, k, nErrors, checkStep;

    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors, &checkStep) == 4; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors, checkStep) > 0); }
  };
  
  static bool write(FILE *fo, const vector<u32> &vect) { return fwrite(vect.data(), vect.size() * sizeof(vect[0]), 1, fo); }
  static bool read(FILE *fi, int n, vector<u32> &out) {
    out.resize(n);
    return fread(out.data(), n * sizeof(u32), 1, fi);
  }
  
  static bool write(int E, const string &name, const std::vector<u32> &check, int k, int nErrors, int blockSize, u64 res64) {
    const int nWords = (E - 1) / 32 + 1;
    assert(int(check.size()) == nWords);

    // u64 sum = checksum(check);
    HeaderV5 header{E, k, blockSize, nErrors, res64, string("gpuOwL v") + VERSION + "; " + timeStr()};
    auto fo(open(name, "wb"));
    return fo
      && header.write(fo.get())
      && fwrite(check.data(), (E - 1)/8 + 1, 1, fo.get());
  }

  static std::string fileName(int E, const string &suffix = "") { return std::to_string(E) + suffix + ".owl"; }

  static u64 checksum(const std::vector<u32> &data) {
    u32 a = 1;
    u32 b = 0;
    for (u32 x : data) {
      a += x;
      b += a;
    }
    return (u64(a) << 32) | b;
  }
  
public:

  static bool loadTF(int E, int *outBitHi, int *outClassDone, int *outClassTotal) {
    if (auto fi{open(fileName(E, ".tf"), "rb", false)}) {
      HeaderTF1 h;
      bool ok = h.parse(fi.get());
      assert(ok && h.E == E);
      if (ok) {
        *outBitHi = h.bitHi;
        *outClassDone = h.classDone;
        *outClassTotal = h.classTotal;
        return true;
      }
    }
    return false;
  }

  static bool saveTF(int E, int bitHi, int classDone, int classTotal) {
    string saveFile = fileName(E, ".tf");
    string tempFile = fileName(E, "-temp.tf");
    string prevFile = fileName(E, "-prev.tf");

    HeaderTF1 header{E, bitHi, classDone, classTotal};
    {
      auto fo(open(tempFile, "wb"));
      if (!fo || !header.write(fo.get())) { return false; }
    }

    remove(prevFile.c_str());
    rename(saveFile.c_str(), prevFile.c_str());
    rename(tempFile.c_str(), saveFile.c_str());
    return true;
  }
  
  static LoadResult load(int E, int preferredBlockSize) {
    const int nWords = (E - 1) / 32 + 1;
    
    {
      auto fi{open(fileName(E), "rb", false)};    
      if (!fi) {
        std::vector<u32> check(nWords);
        check[0] = 1;
        return {true, 0, preferredBlockSize, 0, check, 0x3};
      }

      HeaderV5 header;
      if (header.parse(fi.get())) {
        assert(header.E == E);
        std::vector<u32> check(nWords);
        if (!fread(check.data(), (E - 1) / 8 + 1, 1, fi.get())) { return {false}; }
        return {true, header.k, header.blockSize, header.nErrors, check, header.res64};
      }
    }

    auto fi{open(fileName(E), "rb", false)};    
    if (!fi) {
      std::vector<u32> check(nWords);
      check[0] = 1;
      return {true, 0, BLOCK_SIZE, 0, check};
    }
    
    char line[256];
    if (!fgets(line, sizeof(line), fi.get())) { return {false}; }

    {
      HeaderV4 header;
      if (header.parse(line)) {
        assert(header.E == E);
        std::vector<u32> check;
        if (!read(fi.get(), nWords, check) || header.checksum != checksum(check)) { return {false}; }
        return {true, header.k, header.checkStep, header.nErrors, check};
      }
    }

    {
      HeaderV3 header;
      if (header.parse(line)) {
        assert(header.E == E);
        std::vector<u32> data, check;
        if (!read(fi.get(), nWords, data) ||
            !read(fi.get(), nWords, check)) {
          return {false};
        }
        return {true, header.k, header.checkStep, header.nErrors, check};        
      }
    }

    return {false};
  }
  
  static void save(int E, const vector<u32> &check, int k, int nErrors, int checkStep, u64 res64) {
    string saveFile = fileName(E);
    string strE = std::to_string(E);
    string tempFile = strE + "-temp.owl";
    string prevFile = strE + "-prev.owl";
    
    if (write(E, tempFile, check, k, nErrors, checkStep, res64)) {
      remove(prevFile.c_str());
      rename(saveFile.c_str(), prevFile.c_str());
      rename(tempFile.c_str(), saveFile.c_str());
    }
    const int persistStep = 20'000'000;
    if (k && (k % persistStep == 0)) {
      string persistFile = strE + "." + std::to_string(k) + ".owl";
      write(E, persistFile, check, k, nErrors, checkStep, res64);
    }
  }
};
