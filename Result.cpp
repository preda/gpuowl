// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Result.h"

#include "Task.h"
#include "args.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cassert>
#include <string>

Result::~Result() {}

static bool writeResult(const char *part, u32 E, const char *workType, const char *status,
                        const std::string &AID, const std::string &user, const std::string &cpu) {
  std::string uid;
  if (!user.empty()) { uid += ", \"user\":\"" + user + '"'; }
  if (!cpu.empty())  { uid += ", \"computer\":\"" + cpu + '"'; }
  std::string aidJson = AID.empty() ? "" : ", \"aid\":\"" + AID + '"';
  
  char buf[512];
  snprintf(buf, sizeof(buf), "{\"exponent\":\"%u\", \"worktype\":\"%s\", \"status\":\"%s\", "
           "\"program\":{\"name\":\"%s\", \"version\":\"%s\"}, \"timestamp\":\"%s\"%s%s, %s}",
           E, workType, status, PROGRAM, VERSION, timeStr().c_str(), uid.c_str(), aidJson.c_str(), part);
  
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

/*
bool PFResult::write(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"B1\":\"%u\"%s", args.B1, factorStr(factor).c_str());
  return writeResult(buf, task.exponent, "PM1", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
}
*/

bool PRPResult::write(const Args &args, const Task &task) {
  char buf[256];
  if (B1 == 0) {
    assert(baseRes64 == 3);
    assert(factor.empty());
    snprintf(buf, sizeof(buf), "\"res64\":\"%016llx\", \"residue-type\":4", res64);
    return writeResult(buf, task.exponent, "PRP-3", isPrime ? "P" : "C", task.AID, args.user, args.cpu);
  }
  
  snprintf(buf, sizeof(buf), "\"B1\":\"%u\", \"res64\":\"%016llx\", \"res64base\":\"%016llx\"%s",
           B1, res64, baseRes64, factorStr(factor).c_str());
  return writeResult(buf, task.exponent, "PRPF", isPrime ? "P" : "C", task.AID, args.user, args.cpu);
}
