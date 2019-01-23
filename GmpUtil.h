#include "common.h"

#include <string>
#include <vector>

// return GCD(bits - sub, 2^exp - 1) as a decimal string if GCD!=1, or empty string otherwise.
std::string GCD(u32 exp, const std::vector<u32> &bits, u32 sub = 0);

// "BitsRev" means most significant bit at index 0.
vector<bool> powerSmoothBitsRev(u32 exp, u32 B1);

// Returns (x + 1/x) mod (2^exp - 1)
vector<u32> condition(u32 exp, const vector<u32>& x);
