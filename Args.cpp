// GpuOwL, a Mersenne primality tester. Copyright (C) 2017-2018 Mihai Preda.

#include "args.h"
#include "file.h"
#include "clwrap.h"
#include "FFTConfig.h"

#include <vector>
#include <cstring>
#include <cassert>

bool Args::parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      printf(R"(
Command line options:

-user <name>       : specify the user name.
-cpu  <name>       : specify the hardware name.
-time              : display kernel profiling information.
-fft <size>        : specify FFT size, such as: 5000K, 4M, +2, -1.
-block <value>     : PRP GEC block size. Default 400. Smaller block is slower but detects errors sooner.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-device <N>        : select a specific device:
)");

      vector<cl_device_id> deviceIds = getDeviceIDs();
      for (unsigned i = 0; i < deviceIds.size(); ++i) { printf("%2d : %s\n", i, getLongInfo(deviceIds[i]).c_str()); }
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
      return false;
    } else if (!strcmp(arg, "-precompiled")) {
      usePrecompiled = true;
    } else if (!strcmp(arg, "-fft")) {
      if (i < argc - 1) {
        string s = argv[++i];
        fftSize = atoi(s.c_str()) * ((s.back() == 'K') ? 1024 : ((s.back() == 'M') ? 1024 * 1024 : 1));
      } else {
        log("-fft expects <size>\n");
        return false;
      }
    } else if (!strcmp(arg, "-dump")) {
      if (i < argc - 1 && argv[i + 1][0] != '-') {
        dump = argv[++i];
      } else {
        log("-dump expects name");
        return false;
      }
    } else if (!strcmp(arg, "-user")) {
      if (i < argc - 1) {
        user = argv[++i];
      } else {
        log("-user expects name\n");
        return false;
      }
    } else if (!strcmp(arg, "-cpu")) {
      if (i < argc - 1) {
        cpu = argv[++i];
      } else {
        log("-cpu expects name\n");
        return false;
      }
    } else if (!strcmp(arg, "-cl")) {
      if (i < argc - 1) {
        clArgs = argv[++i];
      } else {
        log("-cl expects options string to pass to CL compiler\n");
        return false;
      }
    } else if(!strcmp(arg, "-time")) {
      timeKernels = true;
    } else if (!strcmp(arg, "-carry")) {
      if (i < argc - 1) {
        std::string s = argv[++i];
        if (s == "short" || s == "long") {
          carry = s == "short" ? CARRY_SHORT : CARRY_LONG;
          continue;
        }
      }
      log("-carry expects short|long\n");
      return false;
    } else if (!strcmp(arg, "-block")) {
      if (i < argc - 1) {
        blockSize = atoi(argv[++i]);
        assert(blockSize > 0);
        if (10000 % blockSize) {
          log("Invalid blockSize %u, must divide 10000\n", blockSize);
          return false;
        }
        continue;
      } else {
        log("-block expects <value>\n");
        return false;
      }
    } else if (!strcmp(arg, "-device") || !strcmp(arg, "-d")) {
      if (i < argc - 1) {
        string s = argv[++i];
        size_t p = 0;
        while (true) {
          devices.push_back(stoi(s, &p));
          if (p >= s.size() || s[p] != ',') { break; }
          s = s.substr(p);
        }
      } else {
        log("-device expects <N> argument\n");
        return false;
      }
    } else {
      log("Argument '%s' not understood\n", arg);
      return false;
    }
  }
  
  return true;
}
