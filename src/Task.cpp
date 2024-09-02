// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "Worktodo.h"
#include "Saver.h"
#include "version.h"
#include "Proof.h"
#include "log.h"
#include "timeutil.h"

#include <cmath>
#include <cassert>

namespace {

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
static_assert(OS_COUNT == 13);

constexpr int platform() {

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

struct OsInfo {
  string os;
  string release;
  string arch;
};

[[maybe_unused]] OsInfo getOsInfoMinimum() {
  int plat = platform();
  string os = plat == LINUX_64 || plat == LINUX_32 ? "Linux" : plat == 4 ? "Windows" : plat == MACOSX_64 ? "MacOS" : "";
  return {os, "", ""};
}

#if __has_include(<sys/utsname.h>)

#include <sys/utsname.h>

OsInfo getOsInfo() {
  utsname buf{};
  uname(&buf);
  return OsInfo{buf.sysname, buf.release, buf.machine};
}

#else

OsInfo getOsInfo() { return getOsInfoMinimum(); }

#endif

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

string json(const string& s) { return '"' + s + '"'; }
string json(u32 x) { return to_string(x); }

template<typename T> string json(const string& key, const T& value) { return json(key) + ':' + json(value); }

string maybe(const string& key, const string& value) { return value.empty() ? ""s : json(key, value); }

template<typename T> void operator+=(vector<T>& a, const vector<T>& b) { a.insert(a.end(), b.begin(), b.end()); }


vector<string> commonFields(u32 E, const char *worktype, const string &status) {
  return {
    json("status", status),
    json("exponent", E),
    json("worktype", worktype),
  };
}

vector<string> tailFields(const std::string &AID, const Args &args) {
  assert(*VERSION); // version string isn't empty
  OsInfo os = getOsInfo();
  return {json("program", vector<string>{
                 json("name", "prpll"),
                 json("version", (VERSION[0] == 'v') ? VERSION + 1 : VERSION), // skip leading "v" from version
                 json("port", platform()),
                 json("os", vector<string>{
                   json("os", os.os),
                   maybe("version", os.release),
                   maybe("architecture", os.arch)
                 })
               }),
          maybe("user", args.user),
          maybe("aid", AID),
          maybe("uid", args.uid.empty() ? args.tailDir() : args.uid),
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

void Task::writeResultPRP(const Args &args, bool isPrime, u64 res64, const string& res2048, u32 fftSize, u32 nErrors, const fs::path& proofPath) const {
  vector<string> fields{json("res64", hex(res64)),
                        json("res2048", res2048),
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

void Task::writeResultLL(const Args &args, bool isPrime, u64 res64, u32 fftSize) const {
  vector<string> fields{json("res64", hex(res64)),
                        json("fft-length", fftSize),
                        json("shift-count", 0),
                        json("error-code", "00000000"), // I don't know the meaning of this
  };

  writeResult(exponent, "LL", isPrime ? "P" : "C", AID, args, fields);
}

void Task::execute(GpuCommon shared, Queue *q, u32 instance) {
  if (kind == VERIFY) { exponent = proof::getInfo(verifyPath).exp; }

  assert(exponent);

  LogContext pushContext(std::to_string(exponent));

  FFTConfig fft = FFTConfig::bestFit(*shared.args, exponent, shared.args->fftSpec);

  auto gpu = Gpu::make(q, exponent, shared, fft);

  if (kind == VERIFY) {
    Proof proof{Proof::load(verifyPath)};
    assert(proof.E == exponent);
    bool ok = proof.verify(gpu.get());
    log("proof '%s' %s\n", verifyPath.c_str(), ok ? "verified" : "failed");

  } else if (kind == PRP || kind == LL) {
    bool isPrime;
    if (kind == PRP) {
      auto [tmpIsPrime, res64, nErrors, proofPath, res2048] = gpu->isPrimePRP(*this);
      isPrime = tmpIsPrime;
      writeResultPRP(*shared.args, isPrime, res64, res2048, fft.size(), nErrors, proofPath);
    } else { // LL
      auto [tmpIsPrime, res64] = gpu->isPrimeLL(*this);
      isPrime = tmpIsPrime;
      writeResultLL(*shared.args, isPrime, res64, fft.size());
    }

    Worktodo::deleteTask(*this, instance);

    if (isPrime) {
      log("%u is PRIME!\n", exponent);
    } else {
      gpu->clear(kind == PRP);
    }
  } else {
    throw "Unexpected task kind " + to_string(kind);
  }
}
