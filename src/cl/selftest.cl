#include "base.cl"
#include "trig.cl"

KERNEL(256) testTrig(global double2* out) {
  for (i32 p = get_global_id(0); p < ND / 8; p += get_global_size(0)) { out[p] = slowTrig_N(p, ND/8); }
}
