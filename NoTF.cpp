#include "TF.h"
#include "args.h"

// Link in this file instead of OpenTF.cpp to disable TF compilation.

bool TF::enabled() { return false; }
unique_ptr<TF> makeTF(Args &args) { return 0; }
