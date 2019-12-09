// Copyright Mihai Preda.

#include "Args.h"
#include "File.h"
#include "FFTConfig.h"
#include "clwrap.h"

#include <vector>
#include <string>
#include <regex>
#include <cstring>
#include <cassert>
#include <iterator>
#include <sstream>
#include <algorithm>

string Args::mergeArgs(int argc, char **argv) {
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

void Args::printHelp() {
  printf(R"(
Command line options:

-dir <folder>      : specify local work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and results.txt
                     Multiple GpuOwl instances, each in its own directory, can share a pool of assignments and report
                     the results back to the common pool.
-user <name>       : specify the user name.
-cpu  <name>       : specify the hardware name.
-time              : display kernel profiling information.
-fft <size>        : specify FFT size, such as: 5000K, 4M, +2, -1.
-block <value>     : PRP GEC block size. Default %u. Smaller block is slower but detects errors sooner.
-log <step>        : log every <step> iterations, default %u. Multiple of 10000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-B1                : P-1 B1 bound, default %u
-B2                : P-1 B2 bound, default B1 * 30
-rB2               : ratio of B2 to B1. Default %u, used only if B2 is not explicitly set
-cleanup           : delete save files at end of run
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-pm1 <exponent>    : run a single P-1 test and exit, ignoring worktodo.txt
-results <file>    : name of results file, default 'results.txt'
-iters <N>         : run next PRP test for <N> iterations and exit. Multiple of 10000.
-maxAlloc          : limit GPU memory usage to this value in MB (needed on non-AMD GPUs)
-yield             : enable work-around for CUDA busy wait taking up one CPU core
-nospin            : disable progress spinner
-use NEW_FFT8,OLD_FFT5,NEW_FFT10: comma separated list of defines, see the #if tests in gpuowl.cl (used for perf tuning)
-device <N>        : select a specific device:
)", blockSize, logStep, B1, B2_B1_ratio);
  // -proof [<power>]   : enable experimental PRP proof generation. Default <power> is 7.
  vector<cl_device_id> deviceIds = getAllDeviceIDs();
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
  auto args = splitArgLine(line);
  for (const auto& [key, s] : args) {
    if (key == "-h" || key == "--help") { printHelp(); throw "help"; }
    else if (key == "-proof") {
      proofPow = s.empty() ? 7 : stoi(s);
      if (proofPow < 7 || proofPow > 9) {
        log("-proofPow <power>: power must be between 7 and 9 (got %d); using 7\n", proofPow);
        proofPow = 7;
      }
    }
    else if (key == "-pool") { masterDir = s; }
    else if (key == "-results") { resultsFile = s; }
    else if (key == "-maxBufs") { maxBuffers = stoi(s); }
    else if (key == "-maxAlloc") { maxAlloc = size_t(stoi(s)) << 20; }
    else if (key == "-log") { logStep = stoi(s); assert(logStep && (logStep % 10000 == 0)); }
    else if (key == "-iters") { iters = stoi(s); assert(iters && (iters % 10000 == 0)); }
    else if (key == "-prp") { prpExp = stoll(s); }
    else if (key == "-pm1") { pm1Exp = stoll(s); }
    else if (key == "-B1") { B1 = stoi(s); }
    else if (key == "-B2") { B2 = stoi(s); }
    else if (key == "-rB2") { B2_B1_ratio = stoi(s); }
    else if (key == "-fft") { fftSize = stoi(s) * ((s.back() == 'K') ? 1024 : ((s.back() == 'M') ? 1024 * 1024 : 1)); }
    else if (key == "-dump") { dump = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-cpu") { cpu = s; }
    else if (key == "-time") { timeKernels = true; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-dir") { dir = s; }
    else if (key == "-yield") { cudaYield = true; }
    else if (key == "-nospin") { noSpin = true; }
    else if (key == "-cleanup") { cleanup = true; }
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
    } else if (key == "-use") {
      string ss = s;
      std::replace(ss.begin(), ss.end(), ',', ' ');
      std::istringstream iss{ss};
      flags.insert(flags.end(), std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{});
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }

  if (cpu.empty()) {
    cpu = getShortInfo(getDevice(device)) + "-" + std::to_string(device);
  }

  if (logStep % blockSize) {
    log("blockSize (%u) must divide logStep (%u)\n", blockSize, logStep);
    throw "args: blockSize, logStep";
  }
  if (!masterDir.empty()) {
    if (resultsFile.find_first_of('/') == std::string::npos) {
      resultsFile = masterDir + '/' + resultsFile;
    }
  }
  File::openAppend(resultsFile);  // verify that it's possible to write results
}
