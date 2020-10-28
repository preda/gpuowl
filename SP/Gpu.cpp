// Copyright Mihai Preda and George Woltman.

#include "Gpu.h"
#include "AllocTrac.h"
#include "Queue.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <future>
#include <optional>
#include <numeric>
#include <bitset>
#include <cinttypes>

string SP_SRC =
#include "sp-wrap.cpp"
  
namespace {

[[maybe_unused]] string toLiteral(u32 value) { return to_string(value) + 'u'; }
[[maybe_unused]] string toLiteral(i32 value) { return to_string(value); }

[[maybe_unused]] string toLiteral(u64 value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "0x%016" PRIx64 "ul", value);
  return buf;
  // to_string(value) + "ul";
}

[[maybe_unused]] string toLiteral(double value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%a", value);
  return buf;
}

[[maybe_unused]] string toLiteral(float value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%af", value);
  return buf;
}

struct Define {
  const string str;

  template<typename T> Define(const string& label, T value) : str{label + '=' + toLiteral(value)} {
    assert(label.find('=') == string::npos);
  }

  Define(const string& labelAndVal) : str{labelAndVal} {
    assert(labelAndVal.find('=') != string::npos);
  }
  
  operator string() const { return str; }
};

cl_program compile(cl_context context, cl_device_id id, u32 ND, const string& source) {
  vector<Define> defines;

  // defines.push_back({"ND", ND});
  // defines.push_back({"TRIG_STEP", float(-1 / (long double) ND)});
  
  assert(ND == 1024 * 1024);
  defines.push_back({"WIDTH", 1024});
  defines.push_back({"MIDDLE", 1});
  defines.push_back({"SMALL_HEIGHT", 1024});

  vector<string> strDefines;
  strDefines.insert(strDefines.begin(), defines.begin(), defines.end());
  // log("%s\n", strDefines.c_str());
  cl_program program = compile(context, id, source, " -save-temps=tmp/ ", strDefines);
  if (!program) { throw "OpenCL compilation"; }
  return program;
}

template<typename To, typename From> To pun(From from) {
  union Pun {
    Pun(const From& f) : from{f} {}    
    From from;
    To to;
    static_assert(sizeof(To) == sizeof(From));
  };
  return Pun{from}.to;
}

}

// Specifies size in number of workgroups
// #define LOAD(name, nGroups) name{program.get(), queue, device, nGroups, #name}
// Specifies size in "work size": workSize == nGroups * groupSize
// #define LOAD_WS(name, workSize) name{program.get(), queue, device, #name, workSize}  

/*
string Gpu::makeTrigDelta() {
  string emptyTrigDelta = "global ulong trigDelta[] = {};\n";
  string source = emptyTrigDelta + SP_SRC;
  
  Holder<cl_program> program{compile(context.get(), device, ND, source)};
  Kernel readHwTrig{program.get(), queue, device, "readHwTrig", ND};  
  HostAccessBuffer<float2> bufReadTrig{queue, "readTrig", ND};
  readHwTrig(bufReadTrig);
  vector<float2> v = bufReadTrig.read();

  string trigDelta = "global ulong trigDelta[] = {\n";
  
  for (u32 k = 0; k < ND; ++k) {
    long double angle = - M_PIl * 2 * k / ND;
    long double ls = sinl(angle);
    long double lc = cosl(angle);
    float2 delta{lc - v[k].first, ls - v[k].second};

    u64 x = pun<u64>(delta);
    // printf("%3u %016" PRIx64 "\n", k, x);
    trigDelta += toLiteral(x) + ",";
    if ((k % 8) == 7) { trigDelta += '\n'; }
  }
  trigDelta += "\n};\n";
  return trigDelta;
}
*/

string Gpu::readTrigTable() {
  string emptyTrig = "global float2 TRIG[] = {};\n";
  string source = emptyTrig + SP_SRC;
  
  Holder<cl_program> program{compile(context.get(), device, ND, source)};
  Kernel readHwTrig{program.get(), queue, device, "readHwTrig", ND};  
  HostAccessBuffer<float2> bufReadTrig{queue, "readTrig", ND};
  readHwTrig(bufReadTrig);
  vector<float2> v = bufReadTrig.read();

  string table = "global float2 HW[] = {\n";
  
  for (u32 k = 0; k <= ND; ++k) {
    table += "{"s + toLiteral(v[k].first) + "," + toLiteral(v[k].second) + "},";
    if ((k % 8) == 7) { table += '\n'; }
  }
  table += "};\n";
  return table;
}
  
string makeTrigTableStr(u32 ND) {
  string table = "global float2 TRIG[] = {\n";
  
  for (u32 k = 0; k <= ND; ++k) {
    assert(ND % 2 == 0);
    long double angle = - M_PIl * k / (ND / 2);
    long double lc = cosl(angle);
    long double ls = sinl(angle);
    table += "{"s + toLiteral(float(lc)) + "," + toLiteral(float(ls)) + "},";
    if ((k % 8) == 7) { table += '\n'; }
  }
  table += "};\n";
  return table;
}

template<typename T>
pair<T, T> root1(u32 N, u32 k) {
  assert(k <= N);
  if (k >= N/2) {
    auto [c, s] = root1<T>(N, k - N/2);
    return {-c, -s};
  } else if (k > N/4) {
    auto [c, s] = root1<T>(N, N/2 - k);
    return {-c, s};
  } else if (k > N/8) {
    auto [c, s] = root1<T>(N, N/4 - k);
    return {-s, -c};
  } else {
    assert(!(N&7));
    assert(k <= N/8);
    N /= 2;
    long double angle = - M_PIl * k / N;
    return {cosl(angle), sinl(angle)};    
  }
}

vector<float2> makeDeltaTable(u32 ND, const vector<float2>& hwTrig) {
  assert(hwTrig.size() == ND);
  vector<float2> ret(ND);
  for (u32 k = 0; k < ND; ++k) {
    auto [c, s] = root1<long double>(ND, k);
    auto [cf, sf] = hwTrig[k];
    ret[k] = {c - cf, s - sf};
  }
  return ret;
}

vector<float2> makeTrigTable(u32 ND) {
  vector<float2> ret;
  for (u32 k = 0; k <= ND; ++k) { ret.push_back(root1<float>(ND, k)); }
  printf("Last %g %g\n", ret.back().first, ret.back().second);
  ret.pop_back();
  return ret;
}

vector<float2> init(u32 n, u32 bits) {
  assert(bits >= 2 && bits < 32);
  vector<float2> ret(n);
  for (u32 i = 0; i < n; ++i) {
    i32 x;
    do {
      x = random() & ((1 << bits) - 1);
    } while (x == (1 << (bits - 1)));
    
    x = (x << (32 - bits)) >> (32 - bits);
    ret[i] = {x, 0};
  }
  return ret;
}
  
Gpu::Gpu() :
  ND{1024 * 1024},
  device{getDevice(1)},
  context{device},
  queue{Queue::make(context, false, false)}
{
  // string hw = readTrigTable();
  // printf("%s\n\n", hw.c_str());
  
  // string trigTable = makeTrigTable(ND);
  // printf("%s\n", trigTable.c_str());

  string trigTable = ""s;
  
  string source = trigTable + SP_SRC;
  Holder<cl_program> program{compile(context.get(), device, ND, source)};

  Kernel copyTrig{program.get(), queue, device, "copyTrig",  1024};
  
  Kernel  transposeIn{program.get(), queue, device, "transposeIn",  ND / 16};
  Kernel transposeOut{program.get(), queue, device, "transposeOut", ND / 16};
  
  Kernel fftWin{program.get(), queue, device, "fftWin", ND / 4};
  Kernel fftWout{program.get(), queue, device, "fftWout", ND / 4};
  
  Kernel fftHIn{program.get(), queue, device, "fftHin", ND / 4};
  Kernel fftHout{program.get(), queue, device, "fftHout", ND / 4};
  
  Kernel fftMiddleIn{program.get(), queue, device, "fftMiddleIn", ND / 1};
  Kernel fftMiddleOut{program.get(), queue, device, "fftMiddleOut", ND / 1};

  Kernel square{program.get(), queue, device, "square", ND / 2};
  
  HostAccessBuffer<float2> buf1{queue, "buf1", ND * 2};
  HostAccessBuffer<float2> buf2{queue, "buf2", ND * 2};
  
  HostAccessBuffer<float2> trigBuf{queue, "trig", ND};

  Kernel readHwTrig{program.get(), queue, device, "readHwTrig", ND};
  readHwTrig(trigBuf);
  vector<float2> hwTrig = trigBuf.read();
  vector<float2> deltaTrig = makeDeltaTable(ND, hwTrig);
  trigBuf.write(deltaTrig);
  copyTrig(trigBuf);

#if 1
  srandom(3);
  vector<float2> initial = init(ND*2, 15);
#else
  vector<float2> initial(ND*2);
  initial[0] = {5, 0};
  initial[1] = {-3, 0};
  // initial[2] = -7;
  // initial[3] = 7;
  // initial[4] = -5;
#endif
  
  vector<float2> v = initial;

  // for (u32 i = 0; i < ND*2; ++i) { printf("%u %f\n", i, v[i]); }  
    
  buf1.write(v);

  transposeIn(buf2, buf1);  
  fftWin(buf1, buf2);
  fftMiddleIn(buf2, buf1);
  fftHIn(buf1, buf2);

  square(buf1);

  // v = buf1.read();
  // for (u32 i = 0; i < ND*2; ++i) { printf("%u %f\n", i, v[i]); }  
  // for (u32 i = 1; i < ND*2; i+=2) { v[i] = - v[i]; }
  // buf1.write(v);

  fftHout(buf1);
  fftMiddleOut(buf2, buf1);
  fftWout(buf1, buf2);
  transposeOut(buf2, buf1);

  v = buf2.read();

  for (u32 i = 1; i < ND*2; i+=2) {
    v[i] = {-v[i].first, -v[i].second};
  }

  double maxErr = 0;
  long double sumErr = 0;
  for (u32 i = 0; i < ND*2; ++i) {
    double x = (double(v[i].first) + v[i].second) / (ND * 4);

    // double err = abs(x - initial[i]);
    double err = abs(x - rint(x));
    sumErr += err;
    maxErr = max(err, maxErr);
    // if (abs(x) > 1e-3) { printf("%u %f\n", i, x); }
  }
  printf("max %g avg %g\n", maxErr, double(sumErr / (ND*2)));
}
