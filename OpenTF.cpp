#include "OpenTF.h"

bool TF::enabled() { return true; }
unique_ptr<TF> makeTF(Args &args) { return OpenTF::make(args); }
