// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"

#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

class ProofCache {
  const u32 E;
  std::unordered_map<u32, Words> pending;
  fs::path proofPath;
  
  bool write(u32 k, const Words& words);

  Words read(u32 k) const;

  void flush();
  
public:
  ProofCache(u32 E, const fs::path& proofPath) : E{E}, proofPath{proofPath} {}
  
  ~ProofCache() { flush(); }
  
  void save(u32 k, const Words& words) {
    if (pending.empty() && write(k, words)) { return; }    
    pending[k] = words;
    flush();
  }

  Words load(u32 k) const {
    auto it = pending.find(k);
    return (it == pending.end()) ? read(k) : it->second;
  }

  void clear() { pending.clear(); }

  bool checkExists(u32 k) const;
};
