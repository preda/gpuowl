// Copyright (C) Mihai Preda.

#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "File.h"
#include "GmpUtil.h"
#include "Background.h"
#include "version.h"

#include <cstdio>
#include <cmath>
#include <thread>
#include <cassert>

static void writeResult(const string &part, u32 E, const char *workType, const string &status,
                        const std::string &AID, const Args &args) { // const std::string &user, const std::string &cpu) {
  std::string uid;
  if (!args.user.empty()) { uid += ", \"user\":\"" + args.user + '"'; }
  if (!args.cpu.empty())  { uid += ", \"computer\":\"" + args.cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  
  char buf[512];
  snprintf(buf, sizeof(buf), "{\"exponent\":\"%u\", \"worktype\":\"%s\", \"status\":\"%s\", "
           "\"program\":{\"name\":\"gpuowl\", \"version\":\"%s\"}, \"timestamp\":\"%s\"%s%s%s}",
           E, workType, status.c_str(), VERSION, timeStr().c_str(), uid.c_str(), aidJson.c_str(), part.c_str());
  
  log("%s\n", buf);
  File::append(args.resultsFile, std::string(buf) + '\n');
}

static string factorStr(const string &factor) { return factor.empty() ? "" : (", \"factors\":[\"" + factor + "\"]"); }

static string fftStr(u32 fftSize) { return string(", \"fft-length\":") + to_string(fftSize); }

static string resStr(u64 res64) {
  char buf[64];
  snprintf(buf, sizeof(buf), ", \"res64\":\"%s\"", hex(res64).c_str());
  return buf;
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
