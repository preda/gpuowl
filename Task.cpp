#include "Task.h"

#include "Gpu.h"
#include "Args.h"
#include "file.h"
#include "GmpUtil.h"

#include <cstdio>
#include <cmath>
#include <thread>

static bool writeResult(const string &part, u32 E, const char *workType, const string &status,
                        const std::string &AID, const std::string &user, const std::string &cpu) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  
  char buf[512];
  snprintf(buf, sizeof(buf), "{\"exponent\":\"%u\", \"worktype\":\"%s\", \"status\":\"%s\", "
           "\"program\":{\"name\":\"%s\", \"version\":\"%s\"}, \"timestamp\":\"%s\"%s%s%s}",
           E, workType, status.c_str(), PROGRAM, VERSION, timeStr().c_str(), uid.c_str(), aidJson.c_str(), part.c_str());
  
  log("%s\n", buf);
  auto fo = openAppend("results.txt");
  if (!fo) { return false; }

  fprintf(fo.get(), "%s\n", buf);
  return true;
}

static string factorStr(const string &factor) { return factor.empty() ? "" : (", \"factors\":[\"" + factor + "\"]"); }

static string fftStr(u32 fftSize) { return string(", \"fft-length\":") + to_string(fftSize); }

static string resStr(u64 res64) {
  char buf[64];
  snprintf(buf, sizeof(buf), ", \"res64\":\"%016llx\"", res64);
  return buf;
}

bool Task::writeResultPRP(const Args &args, bool isPrime, u64 res64, u32 fftSize) const {
  assert(B1 == 0 && B2 == 0);

  string status = isPrime ? "P" : "C";
  return writeResult(fftStr(fftSize) + resStr(res64) + ", \"residue-type\":4",
                     exponent, "PRP-3", status, AID, args.user, args.cpu);
}

bool Task::writeResultPM1(const Args& args, const string& factor, u32 fftSize) const {
  string status = factor.empty() ? "NF" : "F";
  string bounds = ", \"B1\":"s + to_string(B1) + ", \"B2\":"s + to_string(B2);

  return writeResult(fftStr(fftSize) + bounds + factorStr(factor),
                     exponent, "PM1", status, AID, args.user, args.cpu);
}

bool Task::execute(const Args &args) {
  assert(kind == PRP || kind == PM1);
  auto gpu = Gpu::make(exponent, args);
  auto fftSize = gpu->getFFTSize();
  
  if (kind == PRP) {
    auto [isPrime, res64] = gpu->isPrimePRP(exponent, args);
    return writeResultPRP(args, isPrime, res64, fftSize);
  } else if (kind == PM1) {
    auto result = gpu->factorPM1(exponent, args, B1, B2);
    if (holds_alternative<string>(result)) {
      string factor = get<string>(result);
      return writeResultPM1(args, factor, fftSize);
    } else {
      std::thread([&args, fftSize, task=*this, data=get<vector<u32>>(std::move(result))](){
                    string gcd = GCD(task.exponent, data, 0);
                    log("%u P-1 final GCD: %s\n", task.exponent, gcd.empty() ? "no factor" : gcd.c_str());
                    task.writeResultPM1(args, gcd, fftSize);
                  }
        ).detach();
      return true;
    }
  }
  assert(false);
  return false;
}
