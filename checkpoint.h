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

struct PRPFState {
  bool ok;
  int stage;
  u32 k;
  u32 B1;
  u32 blockSize;
  u64 res64;
  vector<u32> base;
  vector<u32> check;
};

struct TFState {
  int bitLo;
  int bitHi;
  int nDone;
  int nTotal;
};

class Checkpoint {
private:
  struct HeaderPRPF1 {
    static constexpr const char *HEADER = "OWL PRPF 1 %d %u %u %u %u %016llx\n";
    // PRPF proceeds by first doing a first-stage P-1(B1), followed by PRP with the Base generated in the first stage.
    // This is indicated by "stage" being 1 or 2.

    int stage;
    u32 E;
    u32 k;
    u32 B1;
    u32 blockSize;
    u64 res64;

    bool write(FILE *fo) { return fprintf(fo, HEADER, stage, E, k, B1, blockSize, res64) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &stage, &E, &k, &B1, &blockSize, &res64) == 6; }
  };
  
  struct HeaderTF2 {
    // Exponent, bitLo, bitHi, classDone, classTotal.
    static constexpr const char *HEADER = "OWL TF 2 %u %d %d %d %d\n";
    u32 E;
    int bitLo, bitHi;
    int nDone, nTotal;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, bitLo, bitHi, nDone, nTotal) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &E, &bitLo, &bitHi, &nDone, &nTotal) == 5; }
  };

  struct HeaderP1 {
    // Exponent, iteration, B1.
    static constexpr const char *HEADER = "OWL P-1 1 %u %u %u\n";

    u32 E, k, B1;

    bool write(FILE *fo) { return fprintf(fo, HEADER, E, k, B1) > 0; }
    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &B1) == 3; }
  };

  struct HeaderPRP6 {
    // Exponent, iteration, block-size, res64, nErrors.
    static constexpr const char *HEADER = "OWL PRP 6 %u %u %u %016llx %d\n";

    u32 E, k, blockSize;
    int nErrors;
    u64 res64;

    bool parse(const char *line) { return sscanf(line, HEADER, &E, &k, &blockSize, &res64, &nErrors) == 4; }
    bool write(FILE *fo) {        return fprintf(fo,  HEADER, E,   k,  blockSize,  res64,  nErrors) > 0; }
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
    
  static u64 checksum(const std::vector<u32> &data) {
    u32 a = 1;
    u32 b = 0;
    for (u32 x : data) {
      a += x;
      b += a;
    }
    return (u64(a) << 32) | b;
  }

  static std::string fileName(int E, const string &suffix = "") { return std::to_string(E) + suffix + ".owl"; }
  
  template<typename Header>
  static bool write(const string &fileName, Header &header, const vector<const vector<u32> *> &datas) {
    auto fo(open(fileName, "wb"));
    if (!fo || !header.write(fo.get())) { return false; }
    for (auto *pv : datas) {
      assert(pv->size() == (header.E - 1) / 32 + 1);
      if (!fwrite(pv->data(), pv->size() * 4, 1, fo.get())) { return false; }
    }
    return true;
  }

  template<typename Header>
  static bool save(Header &header, const string &suffix, const vector<const vector<u32> *> &datas = {}, const string &persist = "") {
    u32 E = header.E;
    string tempFile = fileName(E, "-temp" + suffix);
    if (!write(tempFile, header, datas)) { return false; }

    string prevFile = fileName(E, "-prev" + suffix);
    remove(prevFile.c_str());

    string saveFile = fileName(E, suffix);
    rename(saveFile.c_str(), prevFile.c_str());
    rename(tempFile.c_str(), saveFile.c_str());

    if (!persist.empty()) {
      string persistFile = fileName(E, persist + suffix);
      return write(persistFile, header, datas);
    }
    return true;
  }

public:

  static TFState loadTF(u32 E) {
    if (auto fi{open(fileName(E, ".tf"), "rb", false)}) {
      char line[256];
      if (!fgets(line, sizeof(line), fi.get())) { return {0}; }
      HeaderTF2 h;
      if (h.parse(line)) {
        assert(h.E == E && h.bitLo < h.bitHi);
        return {h.bitLo, h.bitHi, h.nDone, h.nTotal};
      }
      assert(false);
    }
    return {0};
  }

  static bool saveTF(u32 E, int bitLo, int bitEnd, int nDone, int nTotal) {
    HeaderTF2 header{E, bitLo, bitEnd, nDone, nTotal};
    return save(header, ".tf");
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
    HeaderP1 header{E, k, B1};
    return save(header, ".pm1", {&bits});
  }

  static PRPFState loadPRPF(u32 E, u32 prefB1, u32 prefBlockSize) {
    const int nWords = (E - 1) / 32 + 1;
    const int nBytes = (E - 1) / 8 + 1;

    vector<u32> base(nWords);
    vector<u32> check(nWords);
    
    auto fi{open(fileName(E, ".prpf"), "rb", false)};
    if (!fi) {
      base[0] = 1;
      return {true, 1, 0, prefB1, prefBlockSize, 0x1, base, check};
    }

    char line[256];
    if (!fgets(line, sizeof(line), fi.get())) { return {false}; }

    HeaderPRPF1 header;
    if (header.parse(line)) {
      assert(header.E == E);
      if (!fread(base.data(), nBytes, 1, fi.get())) { return {false}; }
      if (header.stage > 1) {
        // In stage 2 there is also "check" in addition to "base".
        if (!fread(check.data(), nBytes, 1, fi.get())) { return {false}; }
      }
      return {true, header.stage, header.k, header.B1, header.blockSize, header.res64, base, check};
    }
    return {false};
  }

  template<typename Header> bool write(const string &fileName, const Header &header, const vector<u32> &check) {
    assert(check.size() == (header.E - 1) / 32 + 1);
    auto fo(open(fileName, "wb"));
    return fo
      && header.write(fo.get())
      && fwrite(check.data(), check.size() * 4, 1, fo.get());
  }
  
  static bool savePRPF(u32 E, int stage, u32 k, u32 B1, u32 blockSize, u64 res64, const vector<u32> base, const vector<u32> check) {
    assert(stage == 1 || stage == 2);
    HeaderPRPF1 header{stage, E, k, B1, blockSize, res64};
    bool doPersist = (stage == 1) ? k && k % 1'000'000 == 0 : (k % 10'000'000 == 0);    
    return save(header, ".prpf", {&base, &check}, doPersist ? "."s + to_string(k) : ""s);
  }
  
  static LoadResult loadPRP(u32 E, u32 preferredBlockSize) {
    const int nWords = (E - 1) / 32 + 1;

    {
      auto fi{open(fileName(E), "rb", false)};    
      if (!fi) {
        std::vector<u32> check(nWords);
        check[0] = 1;
        return {true, 0, preferredBlockSize, 0, check, 0x3};
      }

      char line[256];
      if (!fgets(line, sizeof(line), fi.get())) { return {false}; }
      
      HeaderPRP6 header;
      if (header.parse(line)) {
        assert(header.E == E);
        vector<u32> check(nWords);
        if (!fread(check.data(), nWords * 4, 1, fi.get())) { return {false}; }
        return {true, header.k, header.blockSize, header.nErrors, check, header.res64};
      }
    }
        
    {
      auto fi{open(fileName(E), "rb", false)};    
      assert(fi);
      
      HeaderV5 header;
      if (header.parse(fi.get())) {
        assert(header.E == E);
        std::vector<u32> check(nWords);
        if (!fread(check.data(), (E - 1) / 8 + 1, 1, fi.get())) { return {false}; }
        return {true, header.k, header.blockSize, header.nErrors, check, header.res64};
      }
    }
    
    return {false};
  }
  
  static bool savePRP(u32 E, const vector<u32> &check, u32 k, int nErrors, u32 checkStep, u64 res64) {
    HeaderPRP6 header{E, k, checkStep, nErrors, res64};
    const int persistStep = 20'000'000;    
    bool doPersist = k && (k % persistStep == 0);
    return save(header, "", {&check}, doPersist ? "."s + to_string(k) : ""s);
  }
};
