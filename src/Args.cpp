// Copyright (C) Mihai Preda

#include "Args.h"
#include "File.h"
#include "FFTConfig.h"
#include "clwrap.h"
#include "gpuid.h"
#include "Proof.h"

#include <vector>
#include <string>
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

vector<KeyVal> Args::splitArgLine(const string& inputLine) {
  vector<KeyVal> ret;

  string prev;
  for (const string& s : split(inputLine, ' ')) {
    if (s.empty()) { continue; }

    if (prev.empty()) {
      if (s[0] != '-') {
        log("Args: expected '-' before '%s'\n", s.c_str());
        throw "Argument syntax";
      }

      prev = s;
    } else {
      if (s[0] == '-') {
        ret.push_back({prev, {}});
        prev = s;
      } else {
        ret.push_back({prev, s});
        prev.clear();
      }
    }
  }
  if (!prev.empty()) {
    assert(prev[0] == '-');
    ret.push_back({prev, {}});
  }
  return ret;
}

// Splits a string of the form "Foo=bar,C,D=1" into key=value pairs, with value defaulting to "1".
vector<KeyVal> Args::splitUses(string ss) { // pass by value is intentional
  vector<KeyVal> ret;
  std::replace(ss.begin(), ss.end(), ',', ' ');
  std::istringstream iss{ss};
  vector<string> uses{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  for (const string &s : uses) {
    auto pos = s.find('=');
    string key = (pos == string::npos) ? s : s.substr(0, pos);
    string val = (pos == string::npos) ? "1"s : s.substr(pos+1);
    ret.push_back({key, val});
  }
  return ret;
}

void Args::readConfig(const fs::path& path) {
  if (File file = File::openRead(path)) {
    for (string line : file) {
      line = rstripNewline(line);
      parse(line);
    }
  }
}

u32 Args::getProofPow(u32 exponent) const {
  if (proofPow == -1) { return ProofSet::bestPower(exponent); }
  assert(proofPow >= 1);
  return proofPow;
}

string Args::tailDir() const { return fs::path{dir}.filename().string(); }

bool Args::hasFlag(const string& key) const { return flags.find(key) != flags.end(); }

void Args::printHelp() {
  printf(R"(
PRPLL is "PRobable Prime and Lucas-Lehmer Categorizer", AKA "Purple-cat"
PRPLL is under active development and not ready for production use.

PRPLL is an OpenCL (GPU) program for primality testing Mersenne numbers (of the form 2^n - 1).

To check that OpenCL is installed correctly use the command "clinfo". If clinfo does not find any
devices or otherwise fails, this program will not run.

This program is tested on Linux/ROCm (AMD GPUs); it may also run on Windows and on Nvidia GPUs.

For information about Mersenne primes search see https://www.mersenne.org/

Run "prpll -h"; If this displays a list of OpenCL devices, it means that PRPLL is detecting the GPUs
and should be able to run.


Worktodo:
PRPLL keeps the active tasks in per-worker files worktodo-0.txt, worktodo-1.txt etc in the local directory.
These per-worker files are supplied from the global worktodo.txt file if -pool is used.
In turn the global worktodo.txt can be supplied through the primenet.py script,
either the one located at gpuowl/tools/primenet.py or https://download.mersenne.ca/primenet.py

It is also possible to manually add exponents by adding lines of the form "PRP=118063003" to worktodo-<N>.txt


The configuration options listed below can be passed on the command line or can be put in a file
named "config.txt" in the prpll run directory.


-h                 : print general help, list of FFTs, list of devices
-info <fft>        : print detailed information about the given FFT; e.g. -h 1K:13:256
-dir <folder>      : specify local work directory (containing worktodo.txt, results.txt, config.txt, gpuowl.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and results.txt
                     Multiple PRPLL instances, each in its own directory, can share a pool of assignments and report
                     the results back to the common pool.
-verbose           : print more log, useful for developers
-version           : print only the version and exit
-user <name>       : specify the mersenne.org user name (for result reporting)
-workers <N>       : specify the number of parallel PRP tests to run (default 1)

-fft <spec>        : specify FFT or FFTs to use:
                     - a specific configuration: 256:13:1K
                     - a FFT size: 6.5M
                     - a size range: 7M-8M
                     - a list: 256:13:1K,8M
                     See the list of FFTs at the end.

-od <value>        : Overdrive the FFT range (ROE, CARRY32 limits). This allows to use a lower FFT for a given
                     exponent (thus faster), but increases the risk of errors. The presence of errors is detected,
                     but the errors are nevertheless costly computationally and better avoided.
                     A <value> of 1 extends the range by 0.1%% (and this would be acceptable); a value of 10
                     extends the range by 1%% (and this would be quite too much WRT errors).

-block <value>     : PRP block size, one of: 1000, 500, 200. Default 1000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-ll <exponent>     : run a single LL test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-proof <power>     : generate proof of power <power> (default: optimal depending on exponent).
                     A lower power reduces disk space requirements but increases the verification cost.
                     A higher power increases disk usage a lot.
                     e.g. proof power 10 for a 120M exponent uses about %.0fGB of disk space.
-iters <N>         : run next PRP test for <N> iterations and exit. Multiple of 10000.
-save <N>          : specify the number of savefiles to keep (default %u).
-noclean           : do not delete data after the test is complete.
-cache             : use binary kernel cache; useful with repeated use of -roeTune and -tune
-roe               : measure the Round-Off Error (Z) for more iterations (slow)

-use <define>      : comma separated list of defines for configuring gpuowl.cl, such as:
  -use FAST_BARRIER: on AMD Radeon VII and older AMD GPUs, use a faster barrier(). Do not use
                     this option on Nvidia GPUs or on RDNA AMD GPUs where it produces errors
                     (which are nevertheless detected).
  -use NO_ASM      : do not use __asm() blocks (inline assembly)
  -use STATS=<val> : enable carry statistics collection & logging, for the kernel according to <val>:
                     1 = CarryFused
                     2 = CarryFusedMul
                     4 = CarryA
                     8 = CarryMul
  -use TAIL_KERNELS=<val> : change how tailSquare operates according to <val>:
                     0 = single wide, single kernel
                     1 = single wide, two kernels
                     2 = double wide, single kernel
                     3 = double wide, two kernels
  -use TAIL_TRIGS=<val> : change how tailSquare computes final trig values according to <val>:
                     2 = calculate from scratch, no memory read
                     1 = calculate using one complex multiply from cached memory and uncached memory
                     0 = read trig values from memory
  -use PAD=<val>   : insert pad bytes to possibly improve memory access patterns.  Val is number bytes to pad.
  -use MIDDLE_IN_LDS_TRANSPOSE=0|1  : Transpose values in local memory before writing to global memory
  -use MIDDLE_OUT_LDS_TRANSPOSE=0|1 : Transpose values in local memory before writing to global memory
  -use TABMUL_CHAIN=<val>: Controls how trig values are obtained in WIDTH and HEIGHT when FFT-spec is 1.
                     0 = Read one trig value and compute the next 3 or 7.
                     1 = All trig values are pre-computed and read from memmory.

  -use DEBUG       : enable asserts in OpenCL kernels (slow, developers)

-tune              : measures the speed of the FFTs specified in -fft <spec> to find the best FFT for each exponent.

-ctune <configs>   : finds the best configuration for each FFT specified in -fft <spec>.
                     Prints the results in a form that can be incorporated in config.txt
                      -fft 6.5M  -ctune "OUT_SIZEX=32,8;OUT_WG=64,128,256"

                     It is possible to specify -ctune multiple times on the same command in order to define multiple
                     sets of parameters to be combined, e.g.:
                        -ctune "IN_WG=256,128,64" -ctune "OUT_WG=256,64;OUT_SIZEX=32,16,8"
                     which would try only 8 combinations among those two sets.

                     The tunable parameters (with the default value emphasized) are:
                       IN_WG, OUT_WG: 64, 128, *256*
                       IN_SIZEX, OUT_SIZEX: 4, 8, 16, *32*
                       UNROLL_W: *0*, 1
                       UNROLL_H: 0, 1

-device <N>        : select the GPU at position N in the list of devices
-uid    <UID>      : select the GPU with the given UID (on ROCm/AMDGPU, Linux)
-pci    <BDF>      : select the GPU with the given PCI BDF, e.g. "0c:00.0"

Device selection : use one of -uid <UID>, -pci <BDF>, -device <N>, see the list below

)", ProofSet::diskUsageGB(120000000, 10), nSavefiles);

  vector<cl_device_id> deviceIds = getAllDeviceIDs();
  if (!deviceIds.empty()) {
    printf(" N  : PCI BDF |   UID            |   Driver                 |    Device\n");
  }
  for (unsigned i = 0; i < deviceIds.size(); ++i) {
    cl_device_id id = deviceIds[i];
    string bdf = getBdfFromDevice(id);
    printf("%2u  : %7s | %16s | %-24s | %s | %s\n",
           i,
           bdf.c_str(),
           getUidFromBdf(bdf).c_str(),
           getDriverVersion(id).c_str(),
           getDeviceName(id).c_str(),
           getBoardName(id).c_str()
           );

  }
  printf("\nFFT Configurations (specify with -fft <width>:<middle>:<height> from the set below):\n"
         " Size   MaxExp   BPW    FFT\n");
  
  vector<FFTShape> configs = FFTShape::allShapes();
  configs.push_back(configs.front()); // dummy guard for the loop below.
  string variants;
  u32 activeSize = 0;
  double maxBpw = 0;
  for (auto c : configs) {
    if (c.size() != activeSize) {
      if (!variants.empty()) {
        printf("%5s  %7.2fM  %.2f  %s\n",
               numberK(activeSize).c_str(),
               // activeSize * FFTShape::MIN_BPW / 1'000'000,
               activeSize * maxBpw / 1'000'000.0,
               maxBpw,
               variants.c_str());
        variants.clear();
      }
      activeSize = c.size();
      maxBpw = 0;
    }
    maxBpw = max(maxBpw, c.maxBpw());
    if (!variants.empty()) { variants.push_back(','); }
    variants += c.spec();
  }
}

void Args::parse(const string& line) {
  if (line.empty() || line[0] == '#') { return; }

  if (line[0] == '!') {
    // conditional defines predicated on a FFT
    char fftBuf[32];
    char configBuf[256];
    sscanf(line.c_str(), "! %31s %255s", fftBuf, configBuf);
    string fft = fftBuf;
    string config = configBuf;
    perFftConfig[fft] = splitUses(config);
    return;
  }

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
    } else if (key == "-info") {
      if (s.empty()) {
        log("-info expects an FFT spec, e.g. -info 1K:13:256\n");
        throw "-info <fft>";
      }
      log(" FFT         | BPW   | Max exp (M)\n");
      for (const FFTShape& shape : FFTShape::multiSpec(s)) {
        for (u32 variant = 0; variant <= LAST_VARIANT; variant = next_variant (variant)) {
          FFTConfig fft{shape, variant, CARRY_AUTO};
          log("%12s | %.2f | %5.1f\n", fft.spec().c_str(), fft.maxBpw(), fft.maxExp() / 1'000'000.0);
        }
      }
      throw "info";
    } else if (key == "-od") {
      double od = stod(s);
      fftOverdrive = 1 + od / 1000;
    } else if (key == "-roe") {
      assert(s.empty());
      logROE = true;
    } else if (key == "-tune") {
      assert(s.empty());
      doTune = true;
    } else if (key == "-ctune") {
      doCtune = true;
      if (!s.empty()) { ctune.push_back(s); }
    } else if (key == "-ztune") {
      doZtune = true;
    } else if (key == "-carryTune") {
      carryTune = true;
    } else if (key == "-verbose" || key == "-v") {
      verbose = true;
    } else if (key == "-time") {
      profile = true;
    } else if (key == "-workers") {
      if (s.empty()) {
        log("-workers expects <N>\n");
        throw "-workers <N>";
      }
      workers = stoi(s);
      if (workers < 1 || workers > 4) {
        throw "Number of workers must be between 1 and 4";
      }
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
      assert(proofPow >= 1);
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
    else if (key == "-maxAlloc" || key == "-maxalloc") {
      assert(!s.empty());
      u32 multiple = (s.back() == 'G') ? (1u << 30) : (1u << 20);
      maxAlloc = size_t(stod(s) * multiple + .5);
    }
    else if (key == "-iters") { iters = stoi(s); assert(iters && (iters % 10000 == 0)); }
    else if (key == "-prp" || key == "-PRP") { prpExp = stoll(s); }
    else if (key == "-ll" || key == "-LL") { llExp = stoll(s); }
    else if (key == "-fft") { fftSpec = s; }
    else if (key == "-dump") { dump = s; }
    else if (key == "-user") { user = s; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-uid") { device = getPosFromUid(s); }
    else if (key == "-pci") { device = getPosFromBdf(s); }
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
      if (blockSize != 1000 && blockSize != 500 && blockSize != 200) {
        log("-block must be one of 1000, 500, 200\n");
        throw "invalid block size";
      }
    } else if (key == "-log") {
      logStep = stoi(s);
      if (logStep % 1000 != 0) {
        log("-log must be a multiple of 1000\n");
        throw "invalid log size";
      }
    } else if (key == "-use") {
      for (const auto& [key, val] : splitUses(s)) {
        auto it = flags.find(key);
        if (it != flags.end() && it->second != val) {
          log("warning: -use %s=%s overrides %s=%s\n", key.c_str(), val.c_str(), it->first.c_str(), it->second.c_str());
        }
        flags[key] = val;
      }
    } else if (key == "-unsafeMath") {
      safeMath = false;
    } else if (key == "-save") {
      nSavefiles = stoi(s);      
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }
}

void Args::setDefaults() {
  uid = getUidFromPos(device);
  log("device %d, OpenCL %s, unique id '%s'\n", device, getDriverVersionByPos(device).c_str(), uid.c_str());
  
  if (!masterDir.empty()) {
    assert(masterDir.is_absolute());
    for (filesystem::path* p : {&proofResultDir, &proofToVerifyDir, &cacheDir}) {
      if (p->is_relative()) { *p = masterDir / *p; }
    }
  }

  for (auto& p : {proofResultDir, proofToVerifyDir, cacheDir}) { fs::create_directory(p); }
}
