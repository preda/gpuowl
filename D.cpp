#include "Pm1Plan.h"
#include "timeutil.h"

#include <vector>
#include <tuple>
#include <array>
#include <cassert>
#include <numeric>
#include <bitset>

int main(int argc, char** argv) {
  initLog();
  
  if (argc < 4) {
    printf("Use: D <B1> <B2> <nBuf>\nE.g. D 5000000 150000000 300\n");
    exit(-1);
  }
  
  u32 B1 = atoi(argv[1]);
  u32 B2 = atoi(argv[2]);
  u32 nBuf = atoi(argv[3]);

  Timer timer;
  vector<bool> primeBits{Pm1Plan::sieve(B1, B2)};
  // log("primes %.1fs\n", timer.deltaSecs());
  
  // for (u32 nBuf = 284; nBuf < 450; nBuf += nBuf < 100 ? 10 : 30) {
  printf("\nnBuf = %u\n", nBuf);
  for (u32 D : {210, 330, 420, 462, 660, 770, 924, 1540, 2310}) {    
    if (nBuf >= Pm1Plan::minBufsFor(D)) {
      Pm1Plan plan{D, nBuf, B1, B2, vector{primeBits}};
      PlanStats stats;
      plan.makePlan(&stats);
    }
  }  
}
