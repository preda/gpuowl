#include <utility>

// Returns the probability of PM1(B1,B2) success for a Mersenne 2^exponent -1 already TF'ed to factoredUpTo, split into
// first and second stage.
std::pair<double, double> pm1(unsigned exponent, unsigned factoredUpTo, unsigned B1, unsigned B2);

// Dickman's "rho" function; rho(x) == F(1/x)
double rho(double x);
