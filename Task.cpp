// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "GmpUtil.h"
#include "Worktodo.h"
#include "checkpoint.h"
#include "version.h"
#include "Proof.h"
#include "log.h"

#include <cstdio>
#include <cmath>
#include <thread>
#include <cassert>

namespace {

string json(const vector<string>& v) {
  bool isFirst = true;
  string s = "{";
  for (const std::string& e : v) {
    if (e.empty()) { continue; }
    if (!isFirst) { s += ", "; } else { isFirst = false; }
    s += e;
  }
  return {isFirst ? ""s : (s + '}')};
}

struct Hex {
  explicit Hex(u64 value) : value{value} {}
  u64 value;
};

string json(Hex x) { return '"' + hex(x.value) + '"'; }
string json(const string& s) { return '"' + s + '"'; }
string json(u32 x) { return json(to_string(x)); }

template<typename T> string json(const string& key, const T& value) { return json(key) + ':' + json(value); }

string maybe(const string& key, const string& value) { return value.empty() ? ""s : json(key, value); }

template<typename T> void operator+=(vector<T>& a, const vector<T>& b) { a.insert(a.end(), b.begin(), b.end()); }

vector<string> commonFields(u32 E, const char *worktype, const string &status) {
  return {json("status", status),
          json("exponent", E),
          json("worktype", worktype)
  };
}

vector<string> tailFields(const std::string &AID, const Args &args) {
  return {json("program", vector<string>{json("name", "gpuowl"), json("version", VERSION)}),
          maybe("user", args.user),
          maybe("computer", args.cpu),
          maybe("aid", AID),
          maybe("uid", args.uid),
          json("timestamp", timeStr())
  };
}

void writeResult(u32 E, const char *workType, const string &status, const std::string &AID, const Args &args,
                 const vector<string>& extras) {
  vector<string> fields = commonFields(E, workType, status);
  fields += extras;
  fields += tailFields(AID, args);
  string s = json(std::move(fields));
  log("%s\n", s.c_str());
  File::append(args.resultsFile, s + '\n');
}

}

void Task::writeResultPRP(const Args &args, bool isPrime, u64 res64, u32 fftSize, u32 nErrors, const fs::path& proofPath) const {
  vector<string> fields{json("res64", Hex{res64}),
                        json("residue-type", 1),
                        json("errors", vector<string>{json("gerbicz", nErrors)}),
                        json("fft-length", fftSize)
  };

  // "proof":{"version":1, "power":6, "hashsize":64, "md5":"0123456789ABCDEF"}, 
  if (!proofPath.empty()) {
    ProofInfo info = proof::getInfo(proofPath);
    fields.push_back(json("proof", vector<string>{
            json("version", 1),
            json("power", info.power),
            json("hashsize", 64),
            json("md5", info.md5)
            }));
  }
  
  writeResult(exponent, "PRP-3", isPrime ? "P" : "C", AID, args, fields);
}

void Task::writeResultPM1(const Args& args, const string& factor, u32 fftSize) const {
  assert(B1);
  bool hasFactor = !factor.empty();

  // An approximation: if no factor is found, we know stage2 was done. Otherwise be 'conservative' and assume not done.
  bool didStage2 = !hasFactor;

  string bounds = "\"B1\":"s + to_string(B1) + (didStage2 ? ", \"B2\":"s + to_string(B2) : "");
  writeResult(exponent, "PM1", hasFactor ? "F" : "NF", AID, args,
              {json("B1", B1),
               didStage2 ? json("B2", B2) : "",
               json("fft-length", fftSize),
               factor.empty() ? "" : (json("factors") + ':' + "[\""s + factor + "\"]")
              });
}

void Task::adjustBounds(Args& args) {
  if (kind == PRP && (wantsPm1 || args.B1 || args.B2)) {
    if (B1 == 0 && args.B1) { B1 = args.B1; }
    if (B2 == 0 && args.B2) { B2 = args.B2; }

    if (B1 == 0) { B1 = u32(float(exponent) / 10'000'000 + .5f) * 500'000; }
    if (B2 == 0) { B2 = B1 * args.B2_B1_ratio; }

    if (B1 < 10000) {
      log("B1=%u too small, adjusted to %u\n", B1, 10000);
      B1 = 10000;
    }
      
    if (B2 < 2 * B1) {
      log("B2=%u too small, adjusted to %u\n", B2, 2 * B1);
      B2 = 2 * B1;
    }
  }
}

void Task::execute(const Args& args) {
  LogContext pushContext(std::to_string(exponent));
  
  if (kind == VERIFY) {
    Proof proof = Proof::load(verifyPath);
    auto gpu = Gpu::make(proof.E, args);
    bool ok = proof.verify(gpu.get());
    log("proof '%s' %s\n", verifyPath.c_str(), ok ? "verified" : "failed");
    return;
  }

  assert(kind == PRP);
  auto gpu = Gpu::make(exponent, args);
  auto fftSize = gpu->getFFTSize();

  if (kind == PRP) {
    auto [factor, isPrime, res64, nErrors, proofPath] = gpu->isPrimePRP(args, *this);
    if (factor.empty()) {
      writeResultPRP(args, isPrime, res64, fftSize, nErrors, proofPath);
    }
    
    Worktodo::deleteTask(*this);
    if (!isPrime) { Saver::cleanup(exponent, args); }
  }
}
