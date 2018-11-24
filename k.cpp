#include "Primes.h"

#include <cstdio>

int main() {
  Primes primes(1000000000);
  for (u32 p : primes.from(3)) {
    u32 z = primes.zn2(p);
    printf("%u\n", p / z);
  }
}
