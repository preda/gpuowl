// Copyright (C) Mihai Preda.

#include "ProofCache.h"
#include "File.h"

bool ProofCache::write(u32 k, const Words& words) {
  try {
    File f = File::openWriteNowaitsync(proofPath / to_string(k));
    f.write(words);
    f.write<u32>({crc32(words)});
  } catch (fs::filesystem_error& e) {
    return false;
  }

  assert(words == read(k));
  return true;
}

Words ProofCache::read(u32 k) const {
  File f = File::openReadThrow(proofPath / to_string(k));
  vector<u32> words = f.read<u32>(E / 32 + 2);
  u32 checksum = words.back();
  words.pop_back();
  if (checksum != crc32(words)) {
    log("checksum %x (expected %x) in '%s'\n", crc32(words), checksum, f.name.c_str());
    throw fs::filesystem_error{"checksum mismatch", {}};
  }
  return words;
}


void ProofCache::flush() {
  for (auto it = pending.cbegin(), end = pending.cend(); it != end && write(it->first, it->second); it = pending.erase(it));
  if (!pending.empty()) {
    log("Could not write %u residues under '%s' -- hurry make space!\n", u32(pending.size()), proofPath.string().c_str());
  }
}
