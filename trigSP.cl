
float tmad(float x, float y, float z) {
#if 1
   return fma(x, y, z);
#else
   return x * y + z;
#endif
}

#if 0
// 1ULP on [0, Pi/4]; about 8% of values not correctly rounded.
float cosSP(float x, float y) {
   // Using cos(a) = 1 - 2*sin^2(a/2)
   float s = sinSP(x/2, y/2);
   return fma(-2*s, s, 1);
}
#endif

float2 trigReduced(u32 k, u32 tau) {
  assert(k <= tau/8);        // already reduced
  double slice = M_PI / (tau / 2); // compile-time

  // split k*slice into two floats such that: x + y == k*slice, and y much smaller than x.
  float x = k * (float)(slice);
  float y = k * slice - x;

  float x2 = x * x;
  float x3 = x * x2;
  float s = 0, c = 0;

  // sin(x + y)
  // y is expected to be much smaller than x
  // 1ULP on [0, Pi/4]; about 2% of values are not correctly rounded.
  {
    float S[] =  {-0.166666672, 0.0083334, -0.00019865, 2.98e-06};
    // for small 'x' we can use one fewer term
    bool shortcut = false;
    float r = shortcut ? S[2] : tmad(x2, S[3], S[2]);
    float t = tmad(tmad(tmad(r, x2, S[1]), x3, -y/2), x2, y);
    s = x + tmad(x3, S[0], t);
  }

  // cos(x+y): 1ULP on [0, Pi/4]; about 4% of values are not correctly rounded.
  {
    float C[] = {0.04166665f, -0.00138877f, 2.4476e-05f};

    float r = x2 / 2;
    float t = 1 - r;
    float u = 1 - t;
    float v = u - r;
    c = t + tmad(x2*x2, tmad(x2, tmad(x2, C[2], C[1]), C[0]), tmad(x, -y, v));
  }

  return (float2)(c, s);
}

// Compute the pair {cos, sin} for k/tau ratio of a full circle (2*Pi).
// It is expected that 'tau' and 'kBound' are compile-time constants, so that many conditions below
// can be evaluated at compile-time.
float2 slowTrig(u32 k, u32 bound, u32 tau) {
   assert(k < bound);
   assert(tau % 8 == 0);
   assert(bound <= 2 * tau);

   if (bound > tau && k >= tau) { k -= tau; } // at most one such reduction
   assert(k < tau);

   bool negate = bound > tau/2 && k >= tau/2;
   if (negate) { k -= tau/2; }

   bool negateCos = bound > tau/4 && k >= tau/4;
   if (negateCos) { k = tau/2 - k; }

   bool flip = bound > tau/8 + 1 && k > tau/8;
   if (flip) { k = tau/4 - k; }

   assert(k <= tau/8);

   float2 r = trigReduced(k, tau);
   if (flip) { r = swap(r); }
   if (negateCos) { r.x = -r.x; }
   if (negate) { r = -r; }

   // Use the "-i" direction for the forward FFT as a convention
   return conjugate(r); // U2(r.x, -r.y);
}

float2 slowTrig_2SH(u32 k, u32 bound) { return slowTrig(k, bound, 2 * SMALL_HEIGHT); }
float2 slowTrig_BH(u32 k, u32 bound)  { return slowTrig(k, bound, BIG_HEIGHT);       }
float2 slowTrig_N(u32 k, u32 bound)   { return slowTrig(k, bound, ND);               }
