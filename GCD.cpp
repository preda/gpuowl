// Copyright 2018 Mihai Preda

#include "GCD.h"

#include <gmp.h>
#include <cassert>

static string doGCD(u32 exp, const vector<u32> &bits, u32 sub = 0) {
  mpz_t b;
  mpz_init(b);
  mpz_import(b, bits.size(), -1 /*order: LSWord first*/, sizeof(u32), 0 /*endianess: native*/, 0 /*nails*/, bits.data());
  if (sub) { mpz_sub_ui(b, b, sub); }
  assert(mpz_sizeinbase(b, 2) <= exp);
  assert(mpz_cmp_ui(b, 0)); // b != 0.
  
  mpz_t m;
  // m := 2^exp - 1.
  mpz_init_set_ui(m, 1);
  mpz_mul_2exp(m, m, exp);
  mpz_sub_ui(m, m, 1);
  assert(mpz_sizeinbase(m, 2) == exp);
    
  mpz_gcd(m, m, b);
    
  mpz_clear(b);

  if (mpz_cmp_ui(m, 1) == 0) { return ""; }

  char *buf = mpz_get_str(nullptr, 10, m);
  string ret = buf;
  free(buf);

  mpz_clear(m);
  return ret;
}

void GCD::start(u32 E, const vector<u32> &bits, u32 sub) {
  bool on = isOngoing();
  assert(!on);
  timer.deltaMillis();
  this->E = E;
  gcdFuture = async(launch::async, doGCD, E, bits, sub);
}

string GCD::get() {
  string s = gcdFuture.get();
  log("%u GCD %s (%.2fs)\n", E, s.empty() ? "no factor" : s.c_str(), timer.deltaMillis() * 0.001);
  return s;
}
