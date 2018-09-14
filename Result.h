// gpuOwL, a GPU OpenCL Lucas-Lehmer primality checker.
// Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "Task.h"
#include "args.h"
#include "common.h"

#include <cstdio>
// #include <cstring>
#include <cassert>
#include <string>

struct Result {
private:
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
    auto fo = open("results.txt", "a");
    if (!fo) { return false; }

    fprintf(fo.get(), "%s\n", buf);
    return true;
  }

  string factorStr() { return factor.empty() ? "" : (", \"factors\":[\"" + factor + "\"]"); }
  
  bool writeTF(const Args &args, const Task &task) {
    char buf[256];
    snprintf(buf, sizeof(buf), "\"bitlo\":%u, \"bithi\":%u, \"begink\":\"%llu\", \"endk\":\"%llu\", \"rangecomplete\":%s%s",
             task.bitLo, task.bitHi, beginK, endK, factor.empty() ? "true" : "false", factorStr().c_str());           
    return writeResult(buf, task.exponent, "TF", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
  }

  bool writePM1(const Args &args, const Task &task) {
    char buf[256];
    snprintf(buf, sizeof(buf), "\"B1\":\"%u\"%s", task.B1, factorStr().c_str());
    return writeResult(buf, task.exponent, "PM1", factor.empty() ? "NF" : "F", task.AID, args.user, args.cpu);
  }

  bool writePRP(const Args &args, const Task &task) {
    char buf[256];
    snprintf(buf, sizeof(buf), "\"residue-type\":1, \"fft-length\":\"%uK\", \"res64\":\"%016llx\", \"errors\":{\"gerbicz\":%u}",
             fftSize / 1024, res, nErrors);

    return writeResult(buf, task.exponent, "PRP-3", isPrime ? "P" : "C", task.AID, args.user, args.cpu);
  }
  
public:
  enum Kind {NONE = 0, TF, PM1, PRP, PRPF};

  Kind kind;

  // TF & PM1
  string factor; //empty for no factor. Decimal otherwise.

  // TF
  u64 beginK;
  u64 endK;

  // PRP
  bool isPrime;
  u64 res;
  u32 nErrors;
  u32 fftSize;
  
  bool write(const Args &args, const Task &task) {
    switch (kind) {
    case TF: return writeTF(args, task);
    case PM1: return writePM1(args, task);
    case PRP: return writePRP(args, task);
    case PRPF: assert(false);
    case NONE: assert(false);
    }
    assert(false);
    return false;
  }

  operator bool() { return kind != NONE; }
};
