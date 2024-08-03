// Copyright (C) Mihai Preda

#include "Saver.h"
#include "CycleFile.h"
#include "File.h"
#include "fs.h"

#include <charconv>
#include <cmath>
#include <type_traits>
#include <cinttypes>
#include <filesystem>
#include <algorithm>

namespace {

// E, k, block-size, res64, nErrors, CRC
static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

// Anticipated next version of the header.
// Has general number form N=k*b^E+c, and labels for values.
static constexpr const char *PRP_v13 = "OWL PRP 13 N=1*2^%u-1 k=%u block=%u res64=%016" SCNx64 " err=%u time=%lf\n";
// static constexpr const char *PRP_v13_PRI = "OWL PRP 13 N=1*2^%u-1 k=%u block=%u res64=%016" PRIx64 " err=%u time=%.0lf\n";

// E, k, CRC
static constexpr const char *LL_v1 = "OWL LL 1 E=%u k=%u CRC=%u\n";

// Anticipated next version.
// Push version number to sync it with PRP.
static constexpr const char *LL_v13 = "OWL LL 13 N=1*2^%u-1 k=%u time=%lf\n";

struct BadHeaderError { string name; };

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

PRPState readState(const PRPState& dummy, File fi) {
  u32 exponent{}, k{}, blockSize{}, nErrors{};
  u64 res64{};
  double elapsed{};

  string header = fi.readLine();

  if (sscanf(header.c_str(), PRP_v13, &exponent, &k, &blockSize, &res64, &nErrors, &elapsed) == 6) {
    return {exponent, k, blockSize, res64, fi.readChecked<u32>(nWords(exponent)), nErrors, elapsed};
  }

  u32 crc{};
  if (sscanf(header.c_str(), PRP_v12, &exponent, &k, &blockSize, &res64, &nErrors, &crc) == 6) {
    return {exponent, k, blockSize, res64, fi.readWithCRC<u32>(nWords(exponent), crc), nErrors, 0};
  }

  log("Loading PRP from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

LLState readState(const LLState& dummy, File fi) {
  u32 exponent{}, k{};
  double elapsed{};

  string header = fi.readLine();

  if (sscanf(header.c_str(), LL_v13, &exponent, &k, &elapsed) == 3) {
    return {exponent, k, fi.readChecked<u32>(nWords(exponent)), elapsed};
  }

  u32 crc{};
  if (sscanf(header.c_str(), LL_v1, &exponent, &k, &crc) == 3) {
    return {exponent, k, fi.readWithCRC<u32>(nWords(exponent), crc), 0};
  }

  log("Loading LL from '%s': bad header '%s'\n", fi.name.c_str(), header.c_str());
  throw BadHeaderError{fi.name};
}

void writeState(const File& fo, const PRPState& state) {
  assert(state.check.size() == nWords(state.exponent));
  if (fo.printf(PRP_v13, state.exponent, state.k, state.blockSize, state.res64, state.nErrors, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.check);
}

void writeState(const File& fo, const LLState& state) {
  assert(state.data.size() == nWords(state.exponent));
  if (fo.printf(LL_v13, state.exponent, state.k, state.elapsed) <= 0) {
    throw WriteError{fo.name};
  }
  fo.writeChecked(state.data);
}

double roundNumberScore(u32 x) {
  if (x == 0) { return 1; }

  double score = 0;
  while (x % 10 == 0) {
    x /= 10;
    ++score;
  }
  if (x % 5 == 0) { score += .7; }
  if (x % 2 == 0) { score += .3; }
  return score;
}

} // namespace

template<> PRPState Saver<PRPState>::initState() {
  return {exponent, 0, blockSize, 3, makeWords(exponent, 1), 0, 0};
}

template<> LLState Saver<LLState>::initState() {
  return {exponent, 0, makeWords(exponent, 4), 0};
}


// ---- Saver ----

template<typename State>
Saver<State>::Saver(u32 exponent, u32 blockSize, u32 nSavefiles) :
  exponent{exponent},
  blockSize{blockSize},
  prefix{to_string(exponent) + '-'},
  nSavefiles{nSavefiles}
{
  assert(blockSize && (blockSize % 100 == 0) && (10'000 % blockSize == 0));

  base = std::is_same_v<State, PRPState> ?
        fs::current_path() / to_string(exponent)
      : fs::current_path() / (string(State::KIND) + '-' + to_string(exponent));

  if (!fs::exists(base)) { fs::create_directories(base); }
}

template<typename State>
Saver<State>::~Saver() = default;

template<typename State>
void Saver<State>::clear(u32 exponent) {
  error_code dummy;
  fs::path base = std::is_same_v<State, PRPState> ?
        fs::current_path() / to_string(exponent)
      : fs::current_path() / (string(State::KIND) + '-' + to_string(exponent));
  fs::remove_all(base, dummy);
}

template<typename State>
void Saver<State>::moveToTrash(fs::path src) {
  log("Removing bad savefile '%s'\n", src.string().c_str());
  fancyRename(src, src + ".bad"s);
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

    if (File fi{File::openRead(path)}; fi) {
      try {
        State state = readState(State{}, std::move(fi));
        assert(state.exponent == exponent);
        if (state.exponent == exponent) {
          return state;
        }
      } catch (const CRCError& e) {
      } catch (const BadHeaderError& e) {
      } catch (const ReadError& e) {
      }
    }

    // if for any reason reading the savefile failed, move it out of the way and retry
    log("Bad savefile '%s'\n", path.string().c_str());
    moveToTrash(path);
  }
  throw "bad savefile retry";
}

template<typename State>
void Saver<State>::trimFiles() {
  vector<u32> v = savefiles(base, prefix, State::KIND);

  assert(nSavefiles > 0);
  while (v.size() > nSavefiles) {
    int bestIdx = -1;
    double bestSpan = 1e20;
    u32 prevK = 0;

    for (u32 i = 0; i < v.size() - 1; ++i) {
      u32 k = v[i];
      double niceBias = std::min(1.0, roundNumberScore(k) - 4);
      double span = (v[i + 1] - prevK) * niceBias;
      prevK = k;
      if (span < bestSpan) {
        bestSpan = span;
        bestIdx = i;
      }
    }
    assert(bestIdx >= 0);
    u32 k = v[bestIdx];
    // log("Deleting savefile %u\n", k);
    fs::path path = pathFor(base, prefix, State::KIND, k);
    fs::remove(path);
    v.erase(v.begin() + bestIdx);
  }
}

template<typename State>
void Saver<State>::save(const State& state) {
  fs::path path = pathFor(base, to_string(exponent) + '-', State::KIND, state.k);
  ::writeState(*CycleFile{path}, state);
  trimFiles();
  // log("rm '%s'\n", pathUnverified(base, prefix).string().c_str());
  fs::remove(pathUnverified(base, prefix));
}

template<>
void Saver<PRPState>::saveUnverified(const PRPState& state) const {
  ::writeState(*CycleFile{pathUnverified(base, prefix)}, state);
}

template<typename State>
void Saver<State>::dropMostRecent() {
  fs::path path = mostRecentSavefile();
  assert(!path.empty());
  if (!path.empty()) { moveToTrash(path); }
}


template class Saver<PRPState>;
template class Saver<LLState>;
