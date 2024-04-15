// Copyright (C) Mihai Preda

#include "SaveMan.h"
#include "File.h"

#include <cmath>
#include <vector>
#include <filesystem>
#include <algorithm>

const constexpr u32 nKeep = 20;

SaveMan::SaveMan(std::string_view kind, u32 exponent) :
  kind{kind},
  exponent{exponent}
{
  base = (kind == "prp") ? fs::current_path() / to_string(exponent)
                         : fs::current_path() / (this->kind + '-' + to_string(exponent));

  if (!fs::exists(base)) { fs::create_directories(base); }
  points = listIterations();
  std::sort(points.begin(), points.end());
}

vector<u32> SaveMan::listIterations() {
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

void SaveMan::removeAll() {
  error_code dummy;
  fs::remove_all(base, dummy);
}

static string str9(u32 k) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%09u", k);
  return buf;
}

fs::path SaveMan::path(u32 k) {
  return base / (to_string(exponent) + '-' + str9(k) + '.' + kind);
}

void SaveMan::del(u32 k) {
  error_code dummy;
  fs::remove(path(k), dummy);
}

static u32 tweak(u32 p) {
  assert(p);
  u32 m = 1;
  while (p % 10 == 0) { p /= 10; m *= 10; }
  return (p + 1) * m;
}

File SaveMan::write(u32 k) {
  assert(k > 1 && k > getLastK());
  File f = File::openWrite(path(k));

  for (u32 i = 0; i < points.size(); ++i) {
    u32 p = tweak(points[i]);
    assert(p < k);
    float r = float(p) / k;
    float cumVal = r * r * nKeep;
    if (cumVal < i) {
      del(points[i]);
      points.erase(points.begin() + i);
      break;
    }
  }
  points.push_back(k);
  return f;
}

File SaveMan::readLast() {
  u32 k = getLastK();
  return k ? File::openReadThrow(path(k)) : File{};
}
