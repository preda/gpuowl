// Copyright (C) Mihai Preda

#include "SaveMan.h"
#include "File.h"

#include <vector>
#include <queue>
#include <filesystem>


const constexpr u32 nKeep = 20;

SaveMan::SaveMan(std::string_view kind, u32 exponent) :
  kind{kind},
  exponent{exponent}
{

  base = (kind == "prp") ? fs::current_path() / to_string(exponent)
                         : fs::current_path() / (this->kind + '-' + to_string(exponent));

  if (!fs::exists(base)) { fs::create_directories(base); }

  vector<u32> ks = listIterations();
  for (u32 k : ks) {
    minVal.push({value(k), k});
    lastK = max(lastK, k);
  }
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

float SaveMan::value(u32 k) {
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

fs::path SaveMan::path(u32 k) {
  return base / (to_string(exponent) + '-' + str9(k) + '.' + kind);
}

void SaveMan::del(u32 k) {
  error_code dummy;
  fs::remove(path(k), dummy);
}

File SaveMan::write(u32 k)
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

File SaveMan::readLast() {
  return lastK ? File::openReadThrow(path(lastK)) : File{};
}
