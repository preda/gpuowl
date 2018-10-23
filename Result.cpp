// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Result.h"

#include "Task.h"
#include "args.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cassert>
#include <string>

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

bool TFResult::write(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"bitlo\":%u, \"bithi\":%u, \"begink\":\"%llu\", \"endk\":\"%llu\", \"rangecomplete\":%s%s",
           task.bitLo, task.bitHi, beginK, endK, factor.empty() ? "true" : "false", factorStr(factor).c_str());           
  return writeResult(buf, task.exponent, "TF", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
}

static string resStr(u64 res64) {
  char buf[64];
  snprintf(buf, sizeof(buf), ", \"res64\":\"%016llx\"", res64);
  return buf;
}

bool PRPResult::write(const Args &args, const Task &task, u32 fftSize) {
  u32 B1 = task.B1;
  
  bool hasFactor = !factor.empty();
  if (hasFactor) {
    assert(!isPrime);
    assert(res64 == 0);
    assert(B1 != 0);
  }

  string status = isPrime ? "P" : (hasFactor ? "F" : "C");
  string fftLength = string(", \"fft-length\":") + to_string(fftSize);
  
  char buf[256];
  if (B1 == 0) {
    assert(baseRes64 == 3);
    string str = resStr(res64) + ", \"residue-type\":4";
    return writeResult(fftLength + str, task.exponent, "PRP-3", status, task.AID, args.user, args.cpu);
  }
  
  string r1 = hasFactor ? "" : resStr(res64);
  string r2 = resStr(baseRes64);
    
  snprintf(buf, sizeof(buf), "%s%s, \"base\":{\"b1\":\"%u\", \"bias\":{\"2\":19}%s}",
           factorStr(factor).c_str(), r1.c_str(), B1, r2.c_str());
  return writeResult(fftLength + buf, task.exponent, "PRP,P-1", status, task.AID, args.user, args.cpu);
}
