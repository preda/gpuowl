#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <cassert>

struct LoadResult {
  bool ok;
  u32 k;
  u32 blockSize;
  int nErrors;
  vector<u32> bits;
  u64 res64;
  u32 B1;
};

struct TFState {
  int bitLo;
  int bitHi;
  int nDone;
  int nTotal;
};

class Checkpoint {
private:
  struct HeaderTF2 {
    // Exponent, bitLo, bitHi, classDone, classTotal.
    static constexpr const char *HEADER = "OWL TF 2 %d %d %d %d %d\n";
    int E;
    int bitLo, bitHi;
    int nDone, nTotal;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, bitLo, bitHi, nDone, nTotal) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &E, &bitLo, &bitHi, &nDone, &nTotal) == 5; }
  };

  struct HeaderTF1 {
    // Exponent, bitLo, classDone, classTotal.
    static constexpr const char *HEADER = "OWL TF 1 %d %d %d %d\n";
    int E;
    int bitLo;
    int nDone, nTotal;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, bitLo, nDone, nTotal) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &E, &bitLo, &nDone, &nTotal) == 4; }    
  };

  struct HeaderP1 {
    // Exponent, iteration, B1.
    static constexpr const char *HEADER = "OWL P-1 1 %u %u %u\n";

    u32 E, k, B1;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, k, B1) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &B1) == 3; }
  };
  
  struct HeaderV5 {
    static constexpr const char *HEADER_R = R"(OWL 5
Comment: %255[^
]
Type: PRP
Exponent: %u
Iteration: %u
PRP-block-size: %u
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

    static constexpr const char *HEADER_W = R"(OWL 5
Comment: %s
Type: PRP
Exponent: %u
Iteration: %u
PRP-block-size: %u
Residue-64: 0x%016llx
Errors: %d
End-of-header:
\0)";

    u32 E, k, blockSize;
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
    static constexpr const char *HEADER = "OWL 4 %u %u %d %u %016llx\n";

    u32 E, k;
    int nErrors;
    u32 checkStep;
    u64 checksum;

    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors, &checkStep, &checksum) == 5; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors, checkStep, checksum) > 0); }
  };
  
  struct HeaderV3 {
    // <exponent> <iteration> <nErrors> <check-step>
    static constexpr const char *HEADER = "OWL 3 %u %u %d %u\n";

    u32 E, k;
    int nErrors;
    u32 checkStep;

    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &nErrors, &checkStep) == 4; }
    bool write(FILE *fo) { return (fprintf(fo, HEADER, E, k, nErrors, checkStep) > 0); }
  };
  
  static bool write(FILE *fo, const vector<u32> &vect) { return fwrite(vect.data(), vect.size() * sizeof(vect[0]), 1, fo); }
  static bool read(FILE *fi, int n, vector<u32> &out) {
    out.resize(n);
    return fread(out.data(), n * sizeof(u32), 1, fi);
  }
  
  static bool write(u32 E, const string &name, const std::vector<u32> &check, u32 k, int nErrors, u32 blockSize, u64 res64) {
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

  static TFState loadTF(int E) {
    if (auto fi{open(fileName(E, ".tf"), "rb", false)}) {
      char line[256];
      if (!fgets(line, sizeof(line), fi.get())) { return {0}; }

      {
        HeaderTF2 h;
        if (h.parse(line)) {
          assert(h.E == E && h.bitLo < h.bitHi);
          return {h.bitLo, h.bitHi, h.nDone, h.nTotal};
        }
      }
      
      {
        HeaderTF1 h;
        if (h.parse(line)) {
          assert(h.E == E);
          return {h.bitLo, h.bitLo + 1, h.nDone, h.nTotal};
        }
      }

      assert(false);
    }
    return {0};
  }

  static bool saveTF(int E, int bitLo, int bitEnd, int nDone, int nTotal) {
    string saveFile = fileName(E, ".tf");
    string tempFile = fileName(E, "-temp.tf");
    string prevFile = fileName(E, "-prev.tf");

    HeaderTF2 header{E, bitLo, bitEnd, nDone, nTotal};
    {
      auto fo(open(tempFile, "wb"));
      if (!fo || !header.write(fo.get())) { return false; }
    }

    remove(prevFile.c_str());
    rename(saveFile.c_str(), prevFile.c_str());
    rename(tempFile.c_str(), saveFile.c_str());
    return true;
  }

  static LoadResult loadPM1(u32 E, u32 B1) {
    const int nWords = (E - 1) / 32 + 1;
    auto fi{open(fileName(E, ".pm1"), "rb", false)};
    if (!fi) {
      std::vector<u32> bits(nWords);
      bits[0] = 1;
      return {true, 0, 0, 0, bits, 0, B1};
    }

    char line[256];
    if (fgets(line, sizeof(line), fi.get())) {
      HeaderP1 h;
      if (h.parse(line)) {
        assert(h.E == E);
        std::vector<u32> bits(nWords);
        if (fread(bits.data(), (E - 1) / 8 + 1, 1, fi.get())) {
          return {true, h.k, 0, 0, bits, 0, h.B1};
        }
      }
    }
    return {false};
  }

  static bool savePM1(u32 E, const vector<u32> &bits, u32 k, u32 B1) {
    string saveFile = fileName(E, ".pm1");
    string tempFile = fileName(E, "-temp.pm1");
    string prevFile = fileName(E, "-prev.pm1");

    HeaderP1 header{E, k, B1};
    {
      auto fo(open(tempFile, "wb"));
      if (!fo || !header.write(fo.get()) || !fwrite(bits.data(), (E - 1)/8 + 1, 1, fo.get())) { return false; }
    }
      
    remove(prevFile.c_str());
    rename(saveFile.c_str(), prevFile.c_str());
    rename(tempFile.c_str(), saveFile.c_str());
    return true;    
  }
  
  static LoadResult load(u32 E, u32 preferredBlockSize) {
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
    assert(fi);
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
  
  static void save(u32 E, const vector<u32> &check, int k, int nErrors, int checkStep, u64 res64) {
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
