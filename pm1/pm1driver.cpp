#include "pm1prob.h"
#include <cstdio>
#include <cmath>
#include <utility>
#include <tuple>

using u32 = unsigned;
using namespace std;

int main(int argc, char*argv[]) {
  if (argc < 5) {
    printf("Usage: %s <exponent> <factoredTo> <B1> <B2>\nExample %s 100000000 77 500000 3000000", argv[0], argv[0]);
    return 1;
  }
  u32 exponent = atoi(argv[1]);  
  u32 factored = atoi(argv[2]);
  u32 B1 = atoi(argv[3]);
  u32 B2 = atoi(argv[4]);

  auto [p1, p2] = pm1(exponent, factored, B1, B2);
  printf("%.2f%% (first-stage %.2f%%, second-stage %.2f%%)\n", (p1 + p2)*100, p1*100, p2*100);
}
