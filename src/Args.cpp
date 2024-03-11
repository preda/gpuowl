// Copyright (C) Mihai Preda

#include "Args.h"
#include "File.h"
#include "FFTConfig.h"
#include "clwrap.h"
#include "gpuid.h"
#include "Proof.h"

#include <vector>
#include <string>
#include <regex>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <algorithm>

int Args::value(const string& key, int valNotFound) const {
  auto it = flags.find(key);
  if (it == flags.end()) { return valNotFound; }
  return atoi(it->second.c_str());
}

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

void Args::readConfig(const fs::path& path) {
  if (File file = File::openRead(path)) {
    while (true) {
      string line = rstripNewline(file.readLine());
      if (line.empty()) { break; }
      // log("config: %s\n", line.c_str());
      parse(line);
    }
  }
}

u32 Args::getProofPow(u32 exponent) const {
  assert(proofPow >= -1);
  return (proofPow == -1) ? ProofSet::bestPower(exponent) : proofPow;
}

void Args::printHelp() {
  printf(R"(
PRPLL is "PRobable Prime and Lucas-Lehmer Cathegorizer", AKA "Purple-cat"
PRPLL is under active development and not ready for production use.

PRPLL is an OpenCL (GPU) program for primality testing Mersenne numbers (of the form 2^n - 1).

To check that OpenCL is installed correctly use the command "clinfo". If clinfo does not find any
devices or otherwise fails, this program will not run.

This program is tested on Linux/ROCm (AMD GPUs); it may also run on Windows and on Nvidia GPUs.

For information about Mersenne primes search see https://www.mersenne.org/

Run "prpll -h"; If this displays a list of OpenCL devices, it means that PRPLL is detecting the GPUs
and should be able to run.

To use PRPLL you need to create a file named "worktodo.txt" containing the exponent to be tested.
The tool primenet.py (found at tools/primenet.py) can be used to automatically obtain tasks
from the mersenne project.

The configuration options listed below can be passed on the command line or can be put in a file
named "config.txt" in the prpll run directory.


-dir <folder>      : specify local work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and results.txt
                     Multiple PRPLL instances, each in its own directory, can share a pool of assignments and report
                     the results back to the common pool.
-verbose           : print more log, useful for developers
-version           : print only the version and exit
-user <name>       : specify the mersenne.org user name (for result reporting)
-cpu  <name>       : specify a name of the GPU for result reporting to mersenne.org
-time              : display kernel profiling information.
-fft <spec>        : specify FFT e.g.: 1152K, 5M, 5.5M, 256:10:1K
-block <value>     : PRP error-check block size. Must divide 10'000.
-log <step>        : log every <step> iterations. Multiple of 10'000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-ll <exponent>     : run a single LL test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-proof <power>     : generate proof of power <power> (default: optimal depending on exponent).
                     A lower power reduces disk space requirements but increases the verification cost.
                     A higher power increases disk usage a lot.
                     e.g. proof power 10 for a 120M exponent uses about %.0fGB of disk space.
-autoverify <power> : Self-verify proofs generated with at least this power. Default %u.
-results <file>    : name of results file, default '%s'
-iters <N>         : run next PRP test for <N> iterations and exit. Multiple of 10000.
-save <N>          : specify the number of savefiles to keep (default %u).
-noclean           : do not delete data after the test is complete.
-yield             : enable work-around for Nvidia GPUs busy wait. Do not use on AMD GPUs!
-cache             : use binary kernel cache (for developers)
-profile           : enable queue profiling even when not timing kernels

-use <define>      : comma separated list of defines for configuring gpuowl.cl, such as:
  -use FAST_BARRIER: on AMD Radeon VII and older AMD GPUs, use a faster barrier(). Do not use
                     this option on Nvidia GPUs or on RDNA AMD GPUs where it produces errors
                     (which are nevertheless detected).
  -use NO_ASM      : do not use __asm() blocks (inline assembly)
  -use CARRY32     : force 32-bit carry (-use STATS=21 offers carry range statistics)
  -use CARRY64     : force 64-bit carry (a bit slower but no danger of carry overflow)
  -use TRIG_COMPUTE=0|1|2 : select sin/cos tradeoffs (compute vs. precomputed)
                     0 uses precomputed tables (more VRAM access, less DP compute)
                     2 uses more DP compute and less VRAM table access
  -use DEBUG       : enable asserts in OpenCL kernels (slow)
  -use STATS       : enable roundoff (ROE) or carry statistics logging.
                     Allows selecting among the the kernels CarryFused, CarryFusedMul, CarryA, CarryMul using the bit masks:
                     1 = CarryFused
                     2 = CarryFusedMul
                     4 = CarryA
                     8 = CarryMul
                    16 = analyze Carry instead of ROE
                     (the bit mask 16 selects Carry statistics, otherwise ROE statistics)
                     E.g. STATS=15 enables ROE stats for all the four kernels above.
                          STATs=21 enables Carry stats for the CarryFused and CarryA.
                     For carry, the range [0, 2^32] is mapped to [0.0, 1.0] float values; as such the max carry
                     that fits on 32bits (i.e. 31bits absolute value) is mapped to 0.5

-device <N>        : select the GPU at position N in the list of devices
-uid    <UID>      : select the GPU with the given UID (on ROCm/AMDGPU, Linux)
-pci    <BDF>      : select the GPU with the given PCI BDF, e.g. "0c:00.0"

Device selection : use one of -uid <UID>, -pci <BDF>, -device <N>, see the list below

)", ProofSet::diskUsageGB(120000000, 10), proofVerify, resultsFile.string().c_str(), nSavefiles);

  vector<cl_device_id> deviceIds = getAllDeviceIDs();
  if (!deviceIds.empty()) {
    printf(" N  : PCI BDF |   UID            |   Driver           |    Device\n");
  }
  for (unsigned i = 0; i < deviceIds.size(); ++i) {
    cl_device_id id = deviceIds[i];
    string bdf = getBdfFromDevice(id);
    printf("%2u  : %7s | %16s | %s | %s | %s\n",
           i,
           bdf.c_str(),
           getUidFromBdf(bdf).c_str(),
           getDriverVersion(id).c_str(),
           getDeviceName(id).c_str(),
           getBoardName(id).c_str()
           );

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
               activeSize * FFTConfig::MIN_BPW / 1'000'000, activeMaxExp / 1'000'000.0,
               variants.c_str());
        variants.clear();
      }
    }
    activeSize = c.fftSize();
    activeMaxExp = c.maxExp();
    variants += " "s + c.spec();
  }
}

void Args::parse(const string& line) {
  if (line.empty()) { return; }
  if (!silent) { log("config: %s\n", line.c_str()); }
  auto args = splitArgLine(line);
  for (const auto& [key, s] : args) {
    // log("key '%s'\n", key.c_str());
    if (key == "-h" || key == "--help") {
      printHelp();
      throw "help";
    } else if (key == "-version") {
      // log("PRPLL %s\n", VERSION);
      throw "version";
    } else if (key == "-verbose" || key == "-v") {
      verbose = true;
    } else if (key == "-profile") {
      forceProfile = true;
    } else if (key == "-cache") {
      useCache = true;
    } else if (key == "-noclean") {
      clean = false;
    } else if (key == "-proof") {
      int power = 0;
      if (s.empty() || (power = stoi(s)) < 1 || power > 12) {
        log("-proof expects <power> 1-12 (found '%s')\n", s.c_str());
        throw "-proof <power>";
      }
      proofPow = power;
      assert(proofPow > 0);
    } else if (key == "-autoverify") {
      if (s.empty()) {
        log("-autoverify expects <power>\n");
        throw "-autoverify <power>";
      }
      proofVerify = stoi(s);
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
    else if (key == "-ll" || key == "-LL") { llExp = stoll(s); }
    else if (key == "-fft") { fftSpec = s; }
    else if (key == "-dump") { dump = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-cpu") { cpu = s; }
    else if (key == "-time") { timeKernels = true; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-uid") { device = getPosFromUid(s); }
    else if (key == "-pci") { device = getPosFromBdf(s); }
    else if (key == "-dir") { dir = s; }
    else if (key == "-yield") { cudaYield = true; }
    else if (key == "-carry") {
      if (s == "short" || s == "long") {
        carry = s == "short" ? CARRY_SHORT : CARRY_LONG;
      } else {
        log("-carry expects short|long\n");
        throw "-carry expects short|long";
      }
    } else if (key == "-block") {
      blockSize = stoi(s);
      if (10000 % blockSize) {
        log("BlockSize %u must divide 10'000\n", blockSize);
        throw "invalid block size";
      }
    } else if (key == "-use") {
      string ss = s;
      std::replace(ss.begin(), ss.end(), ',', ' ');
      std::istringstream iss{ss};
      vector<string> uses{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
      for (const string &s : uses) {
        auto pos = s.find('=');
        string key = (pos == string::npos) ? s : s.substr(0, pos);
        string val = (pos == string::npos) ? "1"s : s.substr(pos+1);

        if (key == "STATS" && pos == string::npos) {
          // special-case the default value for STATS (=15) being a bit-field
          val = "15";
        }

        auto it = flags.find(key);
        if (it != flags.end() && it->second != val) {
          log("warning: -use %s=%s overrides %s=%s\n", key.c_str(), val.c_str(), it->first.c_str(), it->second.c_str());
        }
        flags[key] = val;
      }
    } else if (key == "-unsafeMath") {
      safeMath = false;
    } else if (key == "-binary") {
      binaryFile = s;
    } else if (key == "-save") {
      nSavefiles = stoi(s);      
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
  uid = getUidFromPos(device);
  log("device %d, unique id '%s', driver '%s'\n", device, uid.c_str(), getDriverVersionByPos(device).c_str());
  
  if (!masterDir.empty()) {
    assert(masterDir.is_absolute());
    for (filesystem::path* p : {&proofResultDir, &proofToVerifyDir, &resultsFile, &cacheDir}) {
      if (p->is_relative()) { *p = masterDir / *p; }
    }
  }

  for (auto& p : {proofResultDir, proofToVerifyDir, cacheDir}) { fs::create_directory(p); }

  File::openAppend(resultsFile);  // verify that it's possible to write results
}
