// Copyright Mihai Preda.

#include "Args.h"
#include "file.h"
#include "clwrap.h"
#include "FFTConfig.h"

#include <vector>
#include <string>
#include <regex>
#include <cstring>
#include <cassert>

string mergeArgs(int argc, char **argv) {
  string ret;
  for (int i = 1; i < argc; ++i) {
    ret += argv[i];
    ret += " ";
  }
  return ret;
}

vector<pair<string, string>> splitArgLine(const string& line) {
  vector<pair<string, string>> ret;
  std::regex rx("\\s*(-+\\w+)\\s+([^-]\\S*)?\\s*([^-]*)");
  for (std::sregex_iterator it(line.begin(), line.end(), rx); it != std::sregex_iterator(); ++it) {
    smatch m = *it;
    // printf("'%s' '%s' '%s'\n", m.str(1).c_str(), m.str(2).c_str(), m.str(3).c_str());
    string prefix = m.prefix().str();
    string suffix = m.str(3);
    if (!prefix.empty()) { log("Args: unexpected '%s' before '%s'\n", prefix.c_str(), m.str(0).c_str()); }
    if (!suffix.empty()) { log("Args: unexpected '%s' in '%s'\n", suffix.c_str(), m.str(0).c_str()); }
    if (!prefix.empty() || !suffix.empty()) { throw "Argument syntax"; }
    ret.push_back(pair(m.str(1), m.str(2)));
  }
  return ret;
}

void printHelp() {
  printf(R"(
Command line options:

-dir <folder>      : specify work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-user <name>       : specify the user name.
-cpu  <name>       : specify the hardware name.
-time              : display kernel profiling information.
-fft <size>        : specify FFT size, such as: 5000K, 4M, +2, -1.
-block <value>     : PRP GEC block size. Default 400. Smaller block is slower but detects errors sooner.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-B1                : P-1 B1, default 500000
-rB2               : ratio of B2 to B1, default 30
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-pm1 <exponent>    : run a single P-1 test and exit, ignoring worktodo.txt
-device <N>        : select a specific device:
)");

  vector<cl_device_id> deviceIds = getDeviceIDs();
  for (unsigned i = 0; i < deviceIds.size(); ++i) { printf("%2u : %s\n", i, getLongInfo(deviceIds[i]).c_str()); }
  printf("\nFFT Configurations:\n");
  
  vector<FFTConfig> configs = FFTConfig::genConfigs();
  configs.push_back(FFTConfig{}); // dummy guard for the loop below.
  string variants;
  u32 activeSize = 0;
  for (auto c : configs) {
    if (c.fftSize != activeSize) {
      if (!variants.empty()) {
        printf("FFT %5s [%6.2fM - %7.2fM] %s\n",
               numberK(activeSize).c_str(),
               activeSize * 1.5 / 1'000'000, FFTConfig::getMaxExp(activeSize) / 1'000'000.0,
               variants.c_str());
        variants.clear();
      }
    }
    activeSize = c.fftSize;
    variants += " "s + FFTConfig::configName(c.width, c.height, c.middle);
  }
}

void Args::parse(string line) {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
  log("config: %s\n", line.c_str());
  
  auto args = splitArgLine(line);
  for (const auto& [key, s] : args) {
    // log("'%s' : '%s'\n", k.c_str(), v.c_str());

    if (key == "-h" || key == "--help") { printHelp(); throw "help"; }
    else if (key == "-prp") { prpExp = stol(s); }
    else if (key == "-pm1") { pm1Exp = stol(s); }
    else if (key == "-B1") { B1 = stoi(s); }
    else if (key == "-rB2") { B2_B1_ratio = stoi(s); }
    else if (key == "-fft") { fftSize = stoi(s) * ((s.back() == 'K') ? 1024 : ((s.back() == 'M') ? 1024 * 1024 : 1)); }
    else if (key == "-dump") { dump = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-cpu") { cpu = s; }
    else if (key == "-time") { timeKernels = true; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-dir") { dir = s; }
    else if (key == "-carry") {
      if (s == "short" || s == "long") {
        carry = s == "short" ? CARRY_SHORT : CARRY_LONG;
      } else {
        log("-carry expects short|long\n");
        throw "-carry expects short|long";
      }
    } else if (key == "-block") {
      blockSize = stoi(s);
      if (blockSize <= 0 || 10000 % blockSize) {
        log("Invalid blockSize %u, must divide 10000\n", blockSize);
        throw "-block size";
      }
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }
}

void Args::parse(int argc, char **argv) { parse(mergeArgs(argc, argv)); }
