// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "GmpUtil.h"
#include "Background.h"
#include "checkpoint.h"
#include "version.h"

#include <cstdio>
#include <cmath>
#include <thread>
#include <cassert>

namespace {

std::string to_string(const std::vector<std::string>& v) {
  bool isFirst = true;
  string s = "{";
  for (const std::string& e : v) {
    if (e.empty()) { continue; }
    if (!isFirst) { s += ", "; } else { isFirst = false; }
    s += e;
  }
  return {isFirst ? ""s : (s + '}')};
}

std::string json(const std::string& s) { return '"' + s + '"'; }

template<typename T>
std::string json(const std::optional<T>& opt) { return opt ? json(*opt) : ""s; }

std::string json(const std::string& key, const std::string& value) { return json(key) + ':' + json(value); }
std::string json(const std::string& key, const char* value) { return json(key) + ':' + json(value); }
std::string json(const std::string& key, const std::optional<std::string>& v) { return v ? json(key, *v) : ""s; }
std::string json(const std::string& key, const std::vector<std::string>& v) { return json(key) + ':' + to_string(v); }


std::optional<std::string> optStr(const std::string& s) { return s.empty() ? std::optional<std::string>{} : s; }

void writeResult(const string &part, u32 E, const char *workType, const string &status,
                        const std::string &AID, const Args &args) {
  std::string s = to_string({
                             json("exponent", std::to_string(E)),
                             json("worktype", workType),
                             json("status", status),
                             json("program", {json("name", "gpuowl"), json("version", VERSION)}),
                             json("timestamp", timeStr()),
                             json("user", optStr(args.user)),
                             json("computer", optStr(args.cpu)),
                             json("uid", optStr(args.uid)),
                             json("aid", optStr(AID)),
                             part
    });
               
  log("%s\n", s.c_str());
  File::append(args.resultsFile, s + '\n');
}

string factorStr(const string &factor) { return factor.empty() ? "" : (", \"factors\":[\"" + factor + "\"]"); }

string fftStr(u32 fftSize) { return string("\"fft-length\":") + std::to_string(fftSize); }

string resStr(u64 res64) {
  char buf[64];
  snprintf(buf, sizeof(buf), ", \"res64\":\"%s\"", hex(res64).c_str());
  return buf;
}

}

void Task::adjustBounds(Args& args) {
  if (kind == PM1) {
    if (B1 == 0) { B1 = args.B1; }
    if (B2 == 0) { B2 = args.B2 ? args.B2 : (B1 * args.B2_B1_ratio); }
    if (B1 < 15015) {
      log("B1=%u too small, adjusted to 15015\n", B1);
      B1 = 15015;
    }
    if (B2 <= B1) {
      log("B2=%u too small, adjusted to %u\n", B2, B1 * 2);
      B2 = B1 * 2;
    }
  }
}

void Task::writeResultPRP(const Args &args, bool isPrime, u64 res64, u32 fftSize, u32 nErrors) const {
  assert(B1 == 0 && B2 == 0);

  string status = isPrime ? "P" : "C";
  writeResult(fftStr(fftSize) + resStr(res64) + ", \"residue-type\":1, \"errors\":{\"gerbicz\":" + to_string(nErrors) + "}",
              exponent, "PRP-3", status, AID, args);
}

void Task::writeResultPM1(const Args& args, const string& factor, u32 fftSize, bool didStage2) const {
  string status = factor.empty() ? "NF" : "F";
  string bounds = ", \"B1\":"s + to_string(B1) + (didStage2 ? ", \"B2\":"s + to_string(B2) : "");

  writeResult(fftStr(fftSize) + bounds + factorStr(factor), exponent, "PM1", status, AID, args);
}

bool Task::execute(const Args& args, Background& background) {
  assert(kind == PRP || kind == PM1);
  auto gpu = Gpu::make(exponent, args);
  auto fftSize = gpu->getFFTSize();
  
  if (kind == PRP) {
    auto [isPrime, res64, nErrors] = gpu->isPrimePRP(exponent, args);
    writeResultPRP(args, isPrime, res64, fftSize, nErrors);
    if (args.proofPow) {
      gpu->buildProof(exponent, args);
    }
    if (args.cleanup && !isPrime) deleteSaveFiles(exponent);
    return true;
  } else if (kind == PM1) {
    auto result = gpu->factorPM1(exponent, args, B1, B2);
    if (holds_alternative<string>(result)) {
      string factor = get<string>(result);
      writeResultPM1(args, factor, fftSize, false);
      return true;
    } else {
      vector<u32> &data = get<vector<u32>>(result);
      if (data.empty()) {
        writeResultPM1(args, "", fftSize, false);
        return true;
      } else {
        background.run([args, fftSize, data{std::move(data)}, task{*this}](){
                         string factor = GCD(task.exponent, data, 0);
                         log("%u P2 GCD: %s\n", task.exponent, factor.empty() ? "no factor" : factor.c_str());
                         task.writeResultPM1(args, factor, fftSize, true);
                       });
        return true;
      }
    }
  }
  assert(false);
  return false;
}
