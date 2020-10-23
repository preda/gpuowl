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

vector<pair<string, string>> splitArgLine(const string& inputLine) {
  vector<pair<string, string>> ret;

  // The line must be ended with at least one space for the regex to function correctly.
  string line = inputLine + ' ';
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
-dir <folder>      : specify local work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and results.txt
                     Multiple GpuOwl instances, each in its own directory, can share a pool of assignments and report
                     the results back to the common pool.
-uid <unique_id>   : specifies to use the GPU with the given unique_id (only on ROCm/Linux)
-user <name>       : specify the user name.
-cpu  <name>       : specify the hardware name.
-time              : display kernel profiling information.
-fft <spec>        : specify FFT e.g.: 1152K, 5M, 5.5M, 256:10:1K
-block <value>     : PRP error-check block size. Must divide 10'000.
-log <step>        : log every <step> iterations. Multiple of 10'000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-B1                : P-1 B1 bound
-B2                : P-1 B2 bound
-rB2               : ratio of B2 to B1. Default %u, used only if B2 is not explicitly set
-cleanup           : delete save files at end of run
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-proof <power>     : By default a proof of power 8 is generated, using 3GB of temporary disk space for a 100M exponent.
                     A lower power reduces disk space requirements but increases the verification cost.
                     A proof of power 9 uses 6GB of disk space for a 100M exponent and enables faster verification.
-tmpDir <dir>      : specify a folder with plenty of disk space where temporary proof checkpoints will be stored.
-results <file>    : name of results file, default 'results.txt'
-iters <N>         : run next PRP test for <N> iterations and exit. Multiple of 10000.
-maxAlloc <size>   : limit GPU memory usage to size, which is a value with suffix M for MB and G for GB.
                     e.g. -maxAlloc 2048M or -maxAlloc 3.5G
-save <N>          : specify the number of savefiles to keep (default 12).
-from <iteration>  : start at the given iteration instead of the most recent saved iteration
-yield             : enable work-around for CUDA busy wait taking up one CPU core
-nospin            : disable progress spinner
-use NEW_FFT8,OLD_FFT5,NEW_FFT10: comma separated list of defines, see the #if tests in gpuowl.cl (used for perf tuning)
-safeMath          : do not use -cl-unsafe-math-optimizations (OpenCL)
-binary <file>     : specify a file containing the compiled kernels binary
-device <N>        : select a specific device:
)", B2_B1_ratio);

  // Undocumented:
  // -D <value>         : specify the P2 "D" value, one of: 210, 330, 420, 462, 660, 770, 924, 1540, 2310.
  
  vector<cl_device_id> deviceIds = getAllDeviceIDs();
  for (unsigned i = 0; i < deviceIds.size(); ++i) {
    printf("%2u %s : %s %s\n", i, getUUID(i).c_str(), getLongInfo(deviceIds[i]).c_str(), isAmdGpu(deviceIds[i]) ? "AMD" : "not-AMD");
  }
  printf("\nFFT Configurations (specify with -fft <width>:<middle>:<height> from the set below):\n");
  
  vector<FFTConfig> configs = FFTConfig::genConfigs();
  configs.push_back(FFTConfig{}); // dummy guard for the loop below.
  string variants;
  u32 activeSize = 0;
  u32 activeMaxExp = 0;
  for (auto c : configs) {
    if (c.fftSize() != activeSize) {
      if (!variants.empty()) {
        printf("FFT %5s [%6.2fM - %7.2fM] %s\n",
               numberK(activeSize).c_str(),
               activeSize * 1.5 / 1'000'000, activeMaxExp / 1'000'000.0,
               variants.c_str());
        variants.clear();
      }
    }
    activeSize = c.fftSize();
    activeMaxExp = c.maxExp();
    variants += " "s + c.spec();
  }
}

static int getSeqId(const std::string& uid) {
  for (int i = 0;; ++i) {
    string foundId = getUUID(i);
    if (foundId == uid) {
      return i;
    } else if (foundId.empty()) {
      break;
    }
  }
  throw std::runtime_error("Could not find GPU with unique-id "s + uid);
}

void Args::parse(string line) {  
  auto args = splitArgLine(line);
  for (const auto& [key, s] : args) {
    // log("key '%s'\n", key.c_str());
    if (key == "-h" || key == "--help") { printHelp(); throw "help"; }
    else if (key == "-proof") {
      int power = 0;
      if (s.empty() || (power = stoi(s)) < 1 || power > 10) {
        log("-proof expects <power> 1 - 10 (found '%s')\n", s.c_str());
        throw "-proof <power>";
      }
      proofPow = power;
      assert(proofPow > 0);
    } else if (key == "-tmpDir" || key == "-tmpdir") {
      if (s.empty()) {
        log("-tmpDir needs <dir>\n");
        throw "-tmpDir needs <dir>";
      }
      tmpDir = s;
    } else if (key == "-keep") {
      if (s != "proof") {
        log("-keep requires 'proof'\n");
        throw "-keep without proof";
      }
      keepProof = true;
    } else if (key == "-verify") {
      if (s.empty()) {
        log("-verify needs <proof-file> or <exponent>\n");
        throw "-verify without proof-file";
      }
      verifyPath = s;
    }
    else if (key == "-pool") {
      masterDir = s;
      if (!masterDir.is_absolute()) {
        log("-pool <path> requires an absolute path\n");
        throw("-pool <path> requires an absolute path");
      }
    }
    else if (key == "-results") { resultsFile = s; }
    else if (key == "-maxAlloc" || key == "-maxalloc") {
      assert(!s.empty());
      u32 multiple = (s.back() == 'G') ? (1u << 30) : (1u << 20);
      maxAlloc = size_t(stod(s) * multiple + .5);
    }
    else if (key == "-log") { logStep = stoi(s); assert(logStep && (logStep % 10000 == 0)); }
    else if (key == "-iters") { iters = stoi(s); assert(iters && (iters % 10000 == 0)); }
    else if (key == "-prp" || key == "-PRP") { prpExp = stoll(s); }
    else if (key == "-B1" || key == "-b1") { B1 = stoi(s); }
    else if (key == "-B2" || key == "-b2") { B2 = stoi(s); }
    else if (key == "-rB2") { B2_B1_ratio = stoi(s); }
    else if (key == "-fft") { fftSpec = s; }
    else if (key == "-dump") { dump = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-cpu") { cpu = s; }
    else if (key == "-time") { timeKernels = true; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-uid") { device = getSeqId(s); }
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
        log("BlockSize %u must divide 10'000\n", blockSize);
        throw "invalid block size";
      }
    } else if (key == "-use") {
      string ss = s;
      std::replace(ss.begin(), ss.end(), ',', ' ');
      std::istringstream iss{ss};
      flags.insert(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{});
    } else if (key == "-safeMath") {
      safeMath = true;
    } else if (key == "-binary") {
      binaryFile = s;
    } else if (key == "-save") {
      nSavefiles = stoi(s);      
    } else if (key == "-from") {
      startFrom = stoi(s);
    } else if (key == "-D") {
      D = stoi(s);
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }

  if (logStep % 10000) {
    log("log step (%u) must be a multiple of 10'000\n", logStep);
    throw "invalid log step";
  }
}

void Args::setDefaults() {
  uid = getUUID(device);
  log("device %d, unique id '%s'\n", device, uid.c_str());
  
  if (cpu.empty()) {
    cpu = uid.empty() ? getShortInfo(getDevice(device)) + "-" + std::to_string(device) : uid;
  }

  if (!masterDir.empty()) {
    assert(masterDir.is_absolute());
    if (proofResultDir.is_relative()) { proofResultDir = masterDir / proofResultDir; }
    if (resultsFile.is_relative()) { resultsFile = masterDir / resultsFile; }
  }

  fs::create_directory(proofResultDir);

  if (!fs::exists(tmpDir)) {
    log("The tmpDir '%s' does not exist\n", tmpDir.string().c_str());
    throw "tmpDir does not exist";
  }

  File::openAppend(resultsFile);  // verify that it's possible to write results
}
