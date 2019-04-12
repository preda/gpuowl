#include "GmpUtil.h"

#include <gmp.h>
#include <cmath>
#include <cassert>

using namespace std;

// return 2**exp - 1
static void mersenne(mpz_t m, u32 exp) {
  // m := 2^exp - 1.
  mpz_set_ui(m, 1);
  mpz_mul_2exp(m, m, exp);
  mpz_sub_ui(m, m, 1);
  assert(mpz_sizeinbase(m, 2) == exp);
}

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32> &bits, u32 sub) {
  mpz_t b;
  mpz_init(b);
  mpz_import(b, bits.size(), -1 /*order: LSWord first*/, sizeof(u32), 0 /*endianess: native*/, 0 /*nails*/, bits.data());
  if (sub) { mpz_sub_ui(b, b, sub); }
  assert(mpz_sizeinbase(b, 2) <= exp);
  assert(mpz_cmp_ui(b, 0)); // b != 0.
  
  mpz_t m;
  mpz_init(m);
  mersenne(m, exp);   // m := 2**exp - 1
    
  mpz_gcd(m, m, b);
    
  mpz_clear(b);

  string ret = "";
  
  if (mpz_cmp_ui(m, 1)) {
    char *buf = mpz_get_str(nullptr, 10, m);
    ret = buf;
    free(buf);
  }
  
  mpz_clear(m);
  return ret;
}

static void powerSmooth(mpz_t a, u32 exp, u32 B1) {
  mpz_set_ui(a, exp);
  mpz_mul_2exp(a, a, 8); // boost 2s.

  mpz_t b;
  mpz_init(b);
  
  for (int k = log2(B1); k >= 1; --k) {
    mpz_primorial_ui(b, u32(pow(B1, 1.0 / k)));
    mpz_mul(a, a, b);
  }  
  mpz_clear(b);
}

// "Rev" means: most significant bit first (at index 0).
vector<bool> powerSmoothBitsRev(u32 exp, u32 B1) {
  mpz_t a;
  mpz_init(a);
  powerSmooth(a, exp, B1);
  int nBits = mpz_sizeinbase(a, 2);
  vector<bool> bits;
  for (int i = nBits - 1; i >= 0; --i) { bits.push_back(mpz_tstbit(a, i)); }
  assert(int(bits.size()) == nBits);
  mpz_clear(a);
  return bits;
}

// Returns (x + 1/x) mod (2^exp - 1)
vector<u32> condition(u32 exp, const vector<u32>& x) {
  mpz_t a;
  mpz_init(a);
  mpz_import(a, x.size(), -1, sizeof(u32), 0, 0, x.data());

  mpz_t m;
  mpz_init(m);
  mersenne(m, exp);
  assert(mpz_cmp(a, m) < 0);
  
  mpz_t b;
  mpz_init(b);  
  
  bool hasInverse = mpz_invert(b, a, m);
  assert(hasInverse);

  // assert(mpz_cmp(b, m) < 0);
  mpz_add(b, b, a);
  if (mpz_cmp(b, m) >= 0) {
    mpz_sub(b, b, m);
    assert(mpz_cmp(b, m) < 0);
  }
  mpz_clear(m);
  mpz_clear(a);
  
  u32 nWords = (exp - 1) / 32 + 1;
  vector<u32> ret(nWords);

  assert(mpz_sizeinbase(b, 32) <= nWords); 
  
  size_t count = 0;
  mpz_export(ret.data(), &count, -1, sizeof(u32), 0, 0, b);
  assert(count == nWords || count == nWords - 1 || count == nWords - 2);
  mpz_clear(b);
  
  return ret;
}
