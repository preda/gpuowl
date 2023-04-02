// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "GmpUtil.h"
#include "Worktodo.h"
#include "Saver.h"
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

constexpr int platform() {
  /*
  0  => 'Windows16', // 16-bit
  1  => 'Windows',   // 32-bit
  2  => 'Linux',     // 32-bit
  3  => 'Solaris',   // never happened
  4  => 'Windows64',
  5  => 'WindowsService',
  6  => 'FreeBSD',
  7  => 'OS/2',
  8  => 'Linux64',
  9  => 'Mac OS X',
  10 => 'Mac OS X 64-bit',
  11 => 'Haiku',
  12 => 'FreeBSD64',
  */

  enum {WIN_16=0, WIN_32, LINUX_32, SOLARIS, WIN_64, WIN_SERV, FREEBSD_32, OS2, LINUX_64, MACOSX_32, MACOSX_64, HAIKU, FREEBSD_64, OS_COUNT};
  assert(OS_COUNT == 13);

  const constexpr bool IS_32BIT = (sizeof(void*) == 4);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  return IS_32BIT ? WIN_32 : WIN_64;

#elif __APPLE__
  return IS_32BIT ? MACOSX_32 : MACOSX_64;

#elif __linux__
  return IS_32BIT ? LINUX_32 : LINUX_64;

#else
#error "Unknown OS platform"
#endif

}

vector<string> commonFields(u32 E, const char *worktype, const string &status) {
  return {
    json("status", status),
    json("exponent", E),
    json("worktype", worktype),
    json("port", platform()),
  };
}

vector<string> tailFields(const std::string &AID, const Args &args) {
  assert(VERSION[0] == 'v');
  return {json("program", vector<string>{json("name", "gpuowl"), json("version", VERSION + 1)}), // skip leading "v" from version
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
/*
string Task::kindStr() const {
  assert(kind == PRP || kind == PM1);
  return kind == PRP ? "PRP" : "PM1";
}
*/

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

  u32 reportB2 = B2;

  writeResult(exponent, "PM1", hasFactor ? "F" : "NF", AID, args,
              {json("B1", B1),
               (reportB2 > B1) ? json("B2", reportB2) : "",
               json("fft-length", fftSize),
               factor.empty() ? "" : (json("factors") + ':' + "[\""s + factor + "\"]")
              });
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

  assert(kind == PRP || kind == PM1);

  auto gpu = Gpu::make(exponent, args);
  auto fftSize = gpu->getFFTSize();

  if (kind == PRP) {
    auto [factor, isPrime, res64, nErrors, proofPath] = gpu->isPrimePRP(args, *this);
    if (factor.empty()) {
      writeResultPRP(args, isPrime, res64, fftSize, nErrors, proofPath);
    }

    Worktodo::deleteTask(*this);
    if (!isPrime) { Saver::cleanup(exponent, args); }
  } else { // P-1
    LogContext p1{"P1"};
    assert(!line.empty());  // We want to pass the same line to mprime following first-stage
    gpu->doPm1(args, *this);
    File::openAppend(args.mprimeDir/"worktodo.add").write(line);
    Worktodo::deleteTask(*this);
    /*
    {
      char buf[256];
      snprintf(buf, sizeof(buf), "Pminus1=%s,1,2,%u,-1,%u,%u,%u\n",
               AID.empty() ? "N/A" : AID.c_str(), exponent, B1, B1, howFarFactored);
      File fo = File::openAppend(args.mprimeDir/"worktodo.add");
      fo.write(""s + buf);
    }
    */
  }
}
