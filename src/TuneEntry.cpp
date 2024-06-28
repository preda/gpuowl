#include "TuneEntry.h"
#include "Args.h"
#include "CycleFile.h"

#include <cassert>

// Returns whether *results* was updated.
bool TuneEntry::update(vector<TuneEntry>& results) const {
  u32 maxExp = fft.maxExp();
  [[maybe_unused]] bool didErase = false;

  int i{};
  for (i = results.size() - 1; i >= 0 && results[i].cost > cost; --i) {
    if (results[i].fft.maxExp() <= maxExp) {
      results.erase(std::next(results.begin(), i));
      didErase = true;
    }
  }

  if (i >= 0 && results[i].fft.maxExp() >= maxExp) {
    assert(!didErase);
    return false;
  }

  results.insert(std::next(results.begin(), i + 1), *this);
  return true;
}

// Returns whether entry *e* represents an improvement over *results* (i.e. would update the results).
bool TuneEntry::willUpdate(const vector<TuneEntry>& results) const {
  u32 maxExp = fft.maxExp();
  for (const auto& r : results) {
    if (r.cost > cost) {
      break;
    } else if (r.fft.maxExp() >= maxExp) {
      return false;
    }
  }
  return true;
}

vector<TuneEntry> TuneEntry::readTuneFile(const Args& args) {
  fs::path tuneFile = "tune.txt";
  if (!fs::exists(tuneFile)) {
    tuneFile = args.masterDir / "tune.txt";
  }

  // if (!fs::exists(tuneFile)) { log("Tune file %s not found\n", tuneFile.string().c_str()); }

  vector<TuneEntry> results;
  File fi = File::openRead(tuneFile);
  if (!fi) { return {}; }

  [[maybe_unused]] u32 prevMaxExp{};
  [[maybe_unused]] double prevCost{};

  for (const string& line : fi) {
    char specBuf[32];
    double cost{};
    if (sscanf(line.c_str(), "%lf %31s", &cost, specBuf) < 2) {
      log("tune.txt line '%s' ignored\n", line.c_str());
    }
    FFTConfig fft{specBuf};
    assert(cost >= prevCost && fft.maxExp() > prevMaxExp);
    prevCost = cost;
    prevMaxExp = fft.maxExp();
    results.push_back({cost, fft});
  }
  if (args.verbose && !results.empty()) { log("Read %u entries from %s\n", u32(results.size()), tuneFile.string().c_str()); }
  return results;
}

void TuneEntry::writeTuneFile(const vector<TuneEntry>& results) {
  [[maybe_unused]] u32 prevMaxExp{};
  [[maybe_unused]] double prevCost{};
  CycleFile tune{"tune.txt"};
  for (const TuneEntry& r : results) {
    u32 maxExp = r.fft.maxExp();
    assert(r.cost >= prevCost && maxExp > prevMaxExp);
    prevCost = r.cost;
    prevMaxExp = maxExp;
    tune->printf("%6.1f %14s # %u\n", r.cost, r.fft.spec().c_str(), maxExp);
  }
}
