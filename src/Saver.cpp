// Copyright (C) Mihai Preda

#include "Saver.h"
#include "CycleFile.h"
#include "File.h"

#include <charconv>
#include <cmath>
#include <type_traits>
#include <cinttypes>
#include <filesystem>
#include <algorithm>

namespace {

// E, k, block-size, res64, nErrors, CRC
static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

// E, k, CRC
static constexpr const char *LL_v1 = "OWL LL 1 E=%u k=%u CRC=%u\n";

bool startsWith(const string& s, const string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

vector<u32> savefiles(fs::path dir, const string& prefix, const string& kind) {
  vector<u32> v;
  for (const auto& entry: fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      string filename = entry.path().filename().string();
      auto dot = filename.find('.');
      if (dot != string::npos && startsWith(filename, prefix) && filename.substr(dot + 1) == kind) {
        assert(dot > prefix.size());
        string id = filename.substr(prefix.size(), dot - prefix.size());
        if (id == "unverified") { continue; }
        u32 k = 0;
        const char* first = id.data();
        const char* end   = first + id.size();
        auto res = from_chars(first, end, k);
        if (res.ptr != end) {
          log("Savefile ignored: '%s' '%s' %p %p\n", filename.c_str(), id.c_str(), end, res.ptr);
        } else {
          v.push_back(k);
        }
      }
    }
  }
  std::sort(v.begin(), v.end());
  return v;
}

string str9(u32 k) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%09u", k);
  return buf;
}

fs::path pathFor(fs::path base, const string& prefix, const string& kind, u32 k) {
  return base / (prefix + str9(k) + '.' + kind);
}

fs::path pathUnverified(fs::path base, const string& prefix) {
  return base / (prefix + "unverified.prp");
}

// find the "most advanced" file in dir with a name of the form
// <prefix><id>.<kind>
// e.g.: 125784077-010000000.prp
fs::path findLast(fs::path dir, const string& prefix, const string& kind) {
  vector<u32> v = savefiles(dir, prefix, kind);
  if (v.empty()) { return {}; }
  u32 lastK = v.back();
  fs::path path = pathFor(dir, prefix, kind, lastK);
  assert(is_regular_file(path));
  return path;
}

optional<PRPState> readState(const PRPState& dummy, File fi, u32 exponent) {
  u32 fileE{}, fileK{}, fileBlockSize{}, nErrors{}, crc{};
  u64 res64{};
  vector<u32> check;

  string header = fi.readLine();
  if (sscanf(header.c_str(), PRP_v12, &fileE, &fileK, &fileBlockSize, &res64, &nErrors, &crc) != 6) {
    log("Loading PRP from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    return {};
  }

  assert(exponent == fileE);
  try {
    check = fi.readWithCRC<u32>(nWords(exponent), crc);
  } catch (const char* e) {
    log("Bad CRC in '%s'\n", fi.name.c_str());
    return {};
  }

  return {{exponent, fileK, fileBlockSize, res64, check, nErrors}};
}

optional<LLState> readState(const LLState& dummy, File fi, u32 exponent) {
  u32 fileE{}, fileK{}, crc{};
  vector<u32> data;

  string header = fi.readLine();
  if (sscanf(header.c_str(), LL_v1, &fileE, &fileK, &crc) != 3) {
    log("Loading LL from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
    return {};
  }

  assert(exponent == fileE);
  try {
    data = fi.readWithCRC<u32>(nWords(exponent), crc);
  } catch (const char*) {
    log("Bad CRC in '%s'\n", fi.name.c_str());
    return {};
  }
  return {{exponent, fileK, data}};
}

void save(const File& fo, const PRPState& state) {
  assert(state.check.size() == nWords(state.exponent));
  if (fo.printf(PRP_v12, state.exponent, state.k, state.blockSize, state.res64, state.nErrors, crc32(state.check)) <= 0) {
      throw(ios_base::failure("can't write header"));
  }
  fo.write(state.check);
}

void save(const File& fo, const LLState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(LL_v1, state.exponent, state.k, crc32(state.data)) <= 0) {
      throw(ios_base::failure("can't write header"));
  }
  fo.write(state.data);
}

} // namespace

template<> PRPState Saver<PRPState>::initState() {
  return {exponent, 0, blockSize, 3, makeWords(exponent, 1), 0};
}

template<> LLState Saver<LLState>::initState() {
  return {exponent, 0, makeWords(exponent, 4)};
}


// ---- Saver ----

template<typename State>
Saver<State>::Saver(u32 exponent, u32 blockSize, u32 nSavefiles) :
  exponent{exponent},
  blockSize{blockSize},
  prefix{to_string(exponent) + '-'},
  nSavefiles{nSavefiles}
{
  assert(blockSize && blockSize % 100 == 0 && 10'000 % blockSize == 0);

  base = std::is_same_v<State, PRPState> ?
        fs::current_path() / to_string(exponent)
      : fs::current_path() / (string(State::KIND) + '-' + to_string(exponent));

  if (!fs::exists(base)) { fs::create_directories(base); }

  trash = base / "bad";
  if (!fs::exists(trash)) { fs::create_directories(trash); }
}

template<typename State>
Saver<State>::~Saver() = default;

template<typename State>
void Saver<State>::moveToTrash(fs::path src) {
  log("Removing bad savefile '%s'\n", src.string().c_str());
  fs::path dest = trash / src.filename();
  fs::remove(dest);
  try {
    fs::rename(src, dest);
  } catch (const fs::filesystem_error& e) {
    log("Can't rename '%s' to '%s'\n", src.string().c_str(), dest.string().c_str());
    fs::remove(src);
  }
}

template<typename State>
fs::path Saver<State>::mostRecentSavefile() {
  fs::path path = pathUnverified(base, prefix);
  error_code dummy;
  if (!fs::is_regular_file(path, dummy)) {
    path = findLast(base, prefix, State::KIND);
  }
  return path;
}

template<typename State>
State Saver<State>::load() {
  for (int i = 0; i < 2; ++i) {
    fs::path path = mostRecentSavefile();

    if (path.empty()) {
      // no savefiles at all
      return initState();
    }

    File fi = File::openRead(path);
    optional<State> maybeState;

    if (fi && (maybeState = readState(State{}, std::move(fi), exponent))) { return *maybeState; }

    // if for any reason opening the savefile failed, move it out of the way and retry
    log("Bad savefile '%s'\n", path.string().c_str());
    moveToTrash(path);
  }
  throw "bad savefiles";
}

template<typename State>
void Saver<State>::trimFiles() {
  static const constexpr u32 N = 4;

  vector<u32> v = savefiles(base, prefix, State::KIND);

  while (v.size() > N) {
    int leastIdx = -1;
    double leastValue = 1e100;
    u32 prevK = 0, prevPrevK = 0;
    assert(!v.empty());
    for (u32 i = 0; i < v.size() - 1; ++i) {
      u32 k = v[i];
      double value = ldexp(k - prevPrevK, -(int(v.size()) - i - 1));
      if (value < leastValue) {
        leastValue = value;
        leastIdx = i;
      }
      prevPrevK = prevK;
      prevK = k;
    }
    assert(leastIdx >= 0);
    u32 k = v[leastIdx];
    assert(leastIdx < int(v.size()) - 1);
    // log("Deleting savefile %u\n", k);
    fs::path path = pathFor(base, prefix, State::KIND, k);
    fs::remove(path);
    v.erase(v.begin() + leastIdx);
  }
}

template<typename State>
void Saver<State>::save(const State& state) {
  fs::path path = pathFor(base, to_string(exponent) + '-', State::KIND, state.k);
  ::save(*CycleFile{path}, state);
  trimFiles();
  // log("rm '%s'\n", pathUnverified(base, prefix).string().c_str());
  fs::remove(pathUnverified(base, prefix));
}

template<>
void Saver<PRPState>::saveUnverified(const PRPState& state) const {
  ::save(*CycleFile{pathUnverified(base, prefix)}, state);
}

template<typename State>
void Saver<State>::dropMostRecent() {
  fs::path path = mostRecentSavefile();
  assert(!path.empty());
  if (!path.empty()) { moveToTrash(path); }
}

template<typename State>
void Saver<State>::clear() {
  error_code dummy;
  fs::remove_all(base, dummy);
}

template class Saver<PRPState>;
template class Saver<LLState>;
