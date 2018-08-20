#include "TF.h"

// Link in this file instead of OpenTF.cpp to disable TF compilation.

class Args;

bool TF::enabled() { return false; }
unique_ptr<TF> makeTF(Args &args) { return 0; }
