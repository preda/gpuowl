// GpuOwl Mersenne primality tester; Copyright (C) 2017-2018 Mihai Preda.

#pragma once

#include "common.h"

#include <string>

class Task;
class Args;

struct Result {
private:
  /*
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
  */
  
  bool writeTF(const Args &args, const Task &task);
  bool writePM1(const Args &args, const Task &task);
  bool writePRP(const Args &args, const Task &task);
  bool writePRPF(const Args &args, const Task &task);
  
public:
  enum Kind {NONE = 0, TF, PM1, PRP, PRPF};

  Kind kind;

  // TF, PM1, PRPF
  string factor; //empty for no factor. Decimal otherwise.

  // TF
  u64 beginK;
  u64 endK;

  // PRP & PRPF
  bool isPrime;
  u64 res;

  // PRP only
  u32 nErrors;
  u32 fftSize;
  
  bool write(const Args &args, const Task &task);
  
  operator bool() { return kind != NONE; }
};
