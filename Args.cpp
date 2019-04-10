// GpuOwL, a Mersenne primality tester. Copyright (C) Mihai Preda.

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
  std::regex rx("(-+\\w+)\\s+(\\w*)\\s*([^-]*)");
  for (std::sregex_iterator it(line.begin(), line.end(), rx); it != std::sregex_iterator(); ++it) {
    smatch m = *it;
    if (!m[3].str().empty()) { log("Args: ignored '%s' in '%s'\n", m[3].str().c_str(), m[0].str().c_str()); }
    ret.push_back(pair(m[1], m[2]));
  }
  return ret;
}

void printHelp() {
  printf(R"(
Command line options:

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

bool Args::parse(int argc, char **argv) {
  string line = mergeArgs(argc, argv);
  log("Args: %s\n", line.c_str());
  
  auto args = splitArgLine(line);
  for (const auto& [key, s] : args) {
    // log("'%s' : '%s'\n", k.c_str(), v.c_str());

    if (key == "-h" || key == "--help") { printHelp(); return false; }
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
    else if (key == "-carry") {
      if (s == "short" || s == "long") {
        carry = s == "short" ? CARRY_SHORT : CARRY_LONG;
      } else {
        log("-carry expects short|long\n");
        return false;
      }
    } else if (key == "-block") {
      blockSize = stoi(s);
      if (blockSize <= 0 || 10000 % blockSize) {
        log("Invalid blockSize %u, must divide 10000\n", blockSize);
        return false;
      }
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      return false;
    }
  }
  return true;
}
