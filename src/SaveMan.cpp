// Copyright (C) Mihai Preda

#include "SaveMan.h"
#include "File.h"

#include <vector>
#include <queue>
#include <filesystem>
#include <cinttypes>

// E, k, block-size, res64, nErrors, CRC
static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

// E, k, CRC
static constexpr const char *LL_v1 = "OWL LL 1 E=%u k=%u CRC=%u\n";

const constexpr u32 nKeep = 20;

template<> PRPState load<PRPState>(File&& fi, u32 exponent, u32 k) {
  if (!k) {
    assert(!fi);
    return {exponent, k, 400, 3, makeWords(exponent, 1), 0};
  }

  string header = fi.readLine();
  u32 fileE, fileK, blockSize, nErrors, crc;
  u64 res64;
  vector<u32> check;

  if (sscanf(header.c_str(), PRP_v12, &fileE, &fileK, &blockSize, &res64, &nErrors, &crc) == 6) {
    assert(exponent == fileE && k == fileK);
    check = fi.readWithCRC<u32>(nWords(exponent), crc);
  } else {
    log("Loading PRP @ %d: bad header '%s'\n", k, header.c_str());
    throw "bad savefile";
  }
  return {exponent, k, blockSize, res64, check, nErrors};
}


template<> LLState load<LLState>(File&& fi, u32 exponent, u32 k) {
  if (!k) {
    assert(!fi);
    return {exponent, k, makeWords(exponent, 4)};
  }

  string header = fi.readLine();
  u32 fileE, fileK, crc;
  vector<u32> data;

  if (sscanf(header.c_str(), LL_v1, &fileE, &fileK, &crc) == 3) {
    assert(exponent == fileE && k == fileK);
    data = fi.readWithCRC<u32>(nWords(exponent), crc);
  } else {
    log("Loading LL @ %d: bad header '%s'\n", k, header.c_str());
    throw "bad savefile";
  }
  return {exponent, k, data};
}

void save(File&& fo, const PRPState& state) {
  assert(state.check.size() == nWords(state.exponent));
  if (fo.printf(PRP_v12, state.exponent, state.k, state.blockSize, state.res64, state.nErrors, crc32(state.check)) <= 0) {
      throw(ios_base::failure("can't write header"));
  }
  fo.write(state.check);
}

void save(File&& fo, const LLState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(LL_v1, state.exponent, state.k, crc32(state.data)) <= 0) {
      throw(ios_base::failure("can't write header"));
  }
  fo.write(state.data);
}

FileMan::FileMan(std::string_view kind, u32 exponent) :
  kind{kind},
  exponent{exponent}
{
  base = fs::current_path() / (this->kind + '-' + to_string(exponent));
  if (!fs::exists(base)) { fs::create_directory(base); }

  vector<u32> ks = listIterations();
  for (u32 k : ks) {
    minVal.push({value(k), k});
    lastK = max(lastK, k);
  }
}

vector<u32> FileMan::listIterations() {
  vector<u32> ret;
  const string prefix = to_string(exponent) + '-';
  for (const auto& entry : fs::directory_iterator(base)) {
    if (entry.is_regular_file()) {
      string name = entry.path().filename().string();
      auto dot = name.find('.');
      if (name.size() >= prefix.size() && name.substr(0, prefix.size()) == prefix
          && dot != string::npos && name.substr(dot + 1) == kind) {
        assert(dot > prefix.size());
        size_t end = 0;
        u32 k = std::stoul(name.substr(prefix.size(), dot), &end);
        if (end != dot - prefix.size()) {
          log("Savefile ignored: '%s'\n", name.c_str());
        } else {
          ret.push_back(k);
        }
      }
    }
  }
  return ret;
}

void FileMan::removeAll() {
  error_code dummy;
  fs::remove_all(base, dummy);
}

float FileMan::value(u32 k) {
  assert(k > 0);
  u32 dist = (k < exponent) ? (exponent - k) : 1;
  u32 nice = 1;

  while (k % 2 == 0) {
    k /= 2;
    nice *= 2;
  }

  while (k % 5 == 0) {
    k /= 5;
    nice *= 5;
  }

  return nice / float(dist);
}

static string str9(u32 k) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%09u", k);
  return buf;
}

fs::path FileMan::path(u32 k) {
  return base / (to_string(exponent) + '-' + str9(k) + '.' + kind);
}

void FileMan::del(u32 k) {
  error_code dummy;
  fs::remove(path(k), dummy);
}

File FileMan::write(u32 k)
{
  assert(k > lastK);
  File f = File::openWrite(path(k));

  if (minVal.size() > nKeep) {
    u32 kDel = minVal.top().second;
    minVal.pop();
    del(kDel);
  }

  minVal.push({value(k), k});
  lastK = k;
  return f;
}

File FileMan::readLast()
{
  return lastK ? File::openRead(path(lastK)) : File{};
}
