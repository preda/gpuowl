// Copyright (C) Mihai Preda.

#pragma once

#include "common.h"

#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

class ProofCache {
  std::unordered_map<u32, Words> pending;
  fs::path proofPath;
  
  bool write(u32 k, const Words& words);

  Words read(u32 E, u32 k) const;

  void flush();
  
public:
  ProofCache(const fs::path& proofPath) : proofPath{proofPath} {}
  
  ~ProofCache() { flush(); }
  
  void save(u32 k, const Words& words) {
    if (pending.empty() && write(k, words)) { return; }    
    pending[k] = words;
    flush();
  }

  Words load(u32 E, u32 k) const {
    auto it = pending.find(k);
    return (it == pending.end()) ? read(E, k) : it->second;
  }

  void clear() { pending.clear(); }
};
