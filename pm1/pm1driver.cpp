#include "pm1prob.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <utility>
#include <tuple>
#include <string>

using u32 = unsigned;
using namespace std;

static u32 parse(const string& s) {
  u32 len = s.size();
  assert(len);
  char c = s[len - 1];
  u32 multiple = (c == 'M' || c == 'm') ? 1'000'000 : ((c == 'K' || c == 'k') ? 1000 : 1);
  return atoi(s.c_str()) * multiple;
}

int main(int argc, char*argv[]) {
  if (argc < 5) {
    printf(R"(Usage: %s <exponent> <factoredTo> <B1> <B2>
Examples:
%s 102M 76 2.8M 150M
)", argv[0], argv[0]);
    return 1;
  }
  
  u32 exponent = parse(argv[1]);
  u32 factored = parse(argv[2]);
  u32 B1 = parse(argv[3]);
  u32 B2 = parse(argv[4]);

  auto [p1, p2] = pm1(exponent, factored, B1, B2);
  printf("%.3f%% (%.3f%% + %.3f%%)\n", (p1 + p2)*100, p1*100, p2*100);
}
