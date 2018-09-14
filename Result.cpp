// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#include "Result.h"

#include "Task.h"
#include "args.h"
#include "timeutil.h"
#include "file.h"

#include <cstdio>
#include <cassert>
#include <string>

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

bool Result::writeTF(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"bitlo\":%u, \"bithi\":%u, \"begink\":\"%llu\", \"endk\":\"%llu\", \"rangecomplete\":%s%s",
           task.bitLo, task.bitHi, beginK, endK, factor.empty() ? "true" : "false", factorStr(factor).c_str());           
  return writeResult(buf, task.exponent, "TF", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
}

bool Result::writePM1(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"B1\":\"%u\"%s", task.B1, factorStr(factor).c_str());
  return writeResult(buf, task.exponent, "PM1", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
}

bool Result::writePRP(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"residue-type\":1, \"fft-length\":\"%uK\", \"res64\":\"%016llx\", \"errors\":{\"gerbicz\":%u}",
           fftSize / 1024, res, nErrors);
  
  return writeResult(buf, task.exponent, "PRP-3", isPrime ? "P" : "C", task.AID, args.user, args.cpu);
}

bool Result::writePRPF(const Args &args, const Task &task) {
  char buf[256];
  snprintf(buf, sizeof(buf), "\"residue-type\":4, \"res64\":\"%016llx\"%s", res, factorStr(factor).c_str());
  
  return writeResult(buf, task.exponent, "PRPF", isPrime ? "P" : "C", task.AID, args.user, args.cpu);
}

bool Result::write(const Args &args, const Task &task) {
  switch (kind) {
  case TF: return writeTF(args, task);
  case PM1: return writePM1(args, task);
  case PRP: return writePRP(args, task);
  case PRPF: return writePRPF(args, task);
  case NONE: assert(false);
  }
  assert(false);
  return false;
}
