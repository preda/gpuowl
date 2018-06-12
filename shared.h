// included from both C++ and OpenCL.

u32 bitposToWord(u32 E, u32 N, u32 offset) { return offset * ((u64) N) / E; }
u32 wordToBitpos(u32 E, u32 N, u32 word) { return (word * ((u64) E) + (N - 1)) / N; }
