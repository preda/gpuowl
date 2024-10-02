// Copyright Mihai Preda

#include "Trig.h"

#include <cassert>

using namespace std;

namespace {

pair<u32, u32> splitTwos(u32 n) {
  int nTwos = 0;
  while (n % 2 == 0) {
    ++nTwos;
    n /= 2;
  }
  return {n, 1u << nTwos};
}


const u32 SCALE_TAB[] = {/*1*/ 29, /*3*/ 35, /*5*/ 21, /*7*/ 15, /*9*/ 11, /*11*/9, /*13*/31, /*15*/7};

std::array<double, 7> SIN_TAB[] = {
  /* 29*/ {0.10833078115826873,-0.00021188703022058996,1.2433062818062454e-07,-3.474022181027978e-11,5.6624423191207863e-15,-6.0408532039528796e-19,4.5052700358598892e-23,},
  /*105*/ {0.029919930034188507,-4.4640645978187822e-06,1.9981202540300318e-10,-4.25886112692796e-15,5.2951964931845789e-20,-4.309173347842599e-25,2.4515106035018415e-30,},
  /* 99*/ {0.031733259127169629,-5.3258972544029942e-06,2.6815885614626197e-10,-6.4294261183329357e-15,8.9922634380642729e-20,-8.2316934435642992e-25,5.2679397743958046e-30,},
  /*403*/ {0.0077955152694535812,-7.8955652921695967e-08,2.3990698162248796e-13,-3.4712288714405201e-19,2.9298160688497543e-25,-1.618529897213503e-31,6.250689919826233e-38,},
};

std::array<double, 7> COS_TAB[] = {
  {-0.0058677790731803559,5.7384718752755652e-06,-2.2448056787863548e-09,4.7042942092574513e-13,-6.1341684192006895e-17,5.4534831728892728e-21,-3.489859271117485e-25,},
  {-0.00044760110662536771,3.3391125108707998e-08,-9.9639363666339207e-13,1.5928103312110162e-17,-1.5843177337952864e-22,1.0742608501889767e-27,-5.1689920417524428e-33,},
  {-0.00050349986741604722,4.2252019414660429e-08,-1.4182590781906825e-12,2.5503330453732864e-17,-2.853533419993741e-22,2.1761884917715547e-27,-1.1623157743743642e-32,},
  {-3.038502915814197e-05,1.5387499949018801e-10,-3.1169975641321366e-16,3.3825022047128522e-22,-2.2839419012622157e-28,1.0513851275501036e-34,-3.4640322440572814e-41,},
};

template<size_t N>
array<double, N> scaleAux(const array<double, N>& v, double step, double f) {
  array<double, N> ret;
  for (u32 i = 0; i < N; ++i) {
    ret[i] = v[i] * f;
    f *= step;
  }
  return ret;
}

template<size_t N>
array<double, N> scaleOdd(const array<double, N>& v, double f) { return scaleAux(v, f * f, f); }

template<size_t N>
array<double, N> scaleEven(const array<double, N>& v, double f) { return scaleAux(v, f * f, f * f); }

} // namespace

TrigCoefs trigCoefs(u32 n) {
  auto [mid, twos] = splitTwos(n);
  assert(twos >= 1 && (twos & (twos - 1)) == 0);
  assert(mid % 2 != 0 && 1 <= mid && mid <= 15);

  u32 pos = (mid - 1) / 2;
  assert(pos < sizeof(SCALE_TAB) / sizeof(SCALE_TAB[0]));
  int scale = SCALE_TAB[pos];

  int factor = mid * scale;
  u32 sinPos = factor == 29 ? 0 : factor == 105 ? 1 : factor == 99 ? 2 : factor == 403 ? 3 : -1;
  assert(sinPos < sizeof(SIN_TAB) / sizeof(SIN_TAB[0]));

  return {scale, scaleOdd(SIN_TAB[sinPos], 1.0/twos), scaleEven(COS_TAB[sinPos], 1.0/twos)};
}
