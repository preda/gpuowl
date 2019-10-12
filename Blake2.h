#include <cstdint>
#include <cstring>
#include <vector>

// Blake2 hash: https://tools.ietf.org/html/rfc7693
class Blake2 {
  using u64 = uint64_t;
  using u8 = uint8_t;

  struct Data {
    const char* begin;
    const char* end;

    Data(const char* ptr, size_t size) : begin(ptr), end(ptr + size) {}
    Data(const void* ptr, size_t size) : Data{static_cast<const char*>(ptr), size} {}
    
    template<typename T>
    Data(const std::vector<T>& v) : Data{v.data(), v.size() * sizeof(T)} {}
    Data(const u64& seed) : Data{&seed, sizeof(seed)} {}
  };
  
public:
  template<typename T>
  static u64 hash64(const std::vector<T>& v) { return Blake2{8}.hash(v.data(), v.size() * sizeof(T)).h[0]; }
  
  static u64 hash64(const void* ptr, size_t size) { return Blake2{8}.hash(static_cast<const char*>(ptr), size).h[0]; }

  template<typename T>
  static u64 hash64(u64 seed, const std::vector<T>& v) {
    Blake2 hasher{8};
    hasher.push(&seed, sizeof(seed));
    hasher.push(v.data(), v.size() * 
  }
    
private:
  Blake2(u8 nOutputBytes) {
    for (int i = 0; i < 8; ++i) { h[i] = IV[i]; }
    h[0] ^= 0x01'01'00'00 | nOutputBytes;
  }
  
  Blake2& hash(const void* ptr, size_t size) {
    push(ptr, ptr + size);
    finish();
    return *this;
  }

  u64 h[8];
  u64 v[16];
  u64 m[16];
  u64 t = 0;
  char* const blockBegin = static_cast<char*>(m);
  char* const blockEnd = static_cast<char*>(m + 16);
  char* blockIt = blockBegin;

  // void push(const void* it, size_t size) { push(static_cast<const char*>(it), static_cast<const char*>(it) + size);  }

  void push(Data data) { push(static_cast<const char*>(data.ptr), static_cast<const char*>(it) + size);  }
  
  void push(const char* it, const char* end) {
    while (it < end) {
      if (blockIt == blockEnd) {
        compress<false>(blockEnd - blockBegin); // == 128
        blockIt = blockBegin;
      }
      
      unsigned nFit = std::min(end - it, blockEnd - blockIt);
      memcpy(blockIt, it, nFit);
      blockIt += nFit;
      it += nFit;
    }
  }

  void finish() {
    memset(blockIt, 0, blockEnd - blockIt);
    compress<true>(blockIt - blockBegin);
    blockIt = nullptr;
  }
    
  static constexpr u64 IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

  static constexpr u8 sigma[12][16] = {
  {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
  { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
  { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
  {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
  {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
  {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
  { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
  { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
  {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
  { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
  {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
  { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
  };
  
  template<unsigned c> u64 rotr(u64 w) { return (w >> c) | (w << (64 - c)); }

  template<int i> void G(int r, u64& a, u64& b, u64& c, u64& d) {
    a += b + m[sigma[r][2*i+0]];
    d = rotr<32>(d ^ a);
    c += d;
    b = rotr<24>(b ^ c);
    a += b + m[sigma[r][2*i+1]];
    d = rotr<16>(d ^ a);
    c += d;
    b = rotr<63>(b ^ c);
  }

  void round(int r) {
    G<0>(r, v[0], v[4], v[ 8], v[12]);
    G<1>(r, v[1], v[5], v[ 9], v[13]);
    G<2>(r, v[2], v[6], v[10], v[14]);
    G<3>(r, v[3], v[7], v[11], v[15]);
    G<4>(r, v[0], v[5], v[10], v[15]);
    G<5>(r, v[1], v[6], v[11], v[12]);
    G<6>(r, v[2], v[7], v[ 8], v[13]);
    G<7>(r, v[3], v[4], v[ 9], v[14]);
  }

  template<bool isFinal>
  void compress(unsigned nDeltaBytes) {
    t += nDeltaBytes;
    for (int i = 0; i < 8; ++i) { v[i] = h[i]; }
    for (int i = 0; i < 8; ++i) { v[8 + i] = IV[i]; }
    v[12] ^= t;
    if (isFinal) { v[14] = ~v[14]; }
    for (int i = 0; i < 12; ++i) { round(i); }
    for (int i = 0; i < 8; ++i) { h[i] ^= v[i] ^ v[i + 8]; }
  }
};
