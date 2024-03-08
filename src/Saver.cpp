// Copyright (C) Mihai Preda

#include "Saver.h"
#include "SaveMan.h"
#include "File.h"

#include <cinttypes>

// E, k, block-size, res64, nErrors, CRC
static constexpr const char *PRP_v12 = "OWL PRP 12 %u %u %u %016" SCNx64 " %u %u\n";

// E, k, CRC
static constexpr const char *LL_v1 = "OWL LL 1 E=%u k=%u CRC=%u\n";

template<typename State>
State load(File&& f, u32 E, u32 k);

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

template<typename State>
Saver<State>::Saver(u32 exponent) :
  man{make_unique<SaveMan>(State::KIND, exponent)}
{}

template<typename State>
Saver<State>::~Saver() = default;

template<typename State>
State Saver<State>::load() {
  return ::load<State>(man->readLast(), man->exponent, man->getLastK());
}

template<typename State>
void Saver<State>::save(const State& s) {
  ::save(man->write(s.k), s);
}

template<typename State>
void Saver<State>::clear() { man->removeAll(); }

template class Saver<PRPState>;
template class Saver<LLState>;
