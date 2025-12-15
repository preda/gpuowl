// Copyright (C) Mihai Preda and George Woltman

#if !IN_WG
#define IN_WG 128
#endif

#if !OUT_WG
#define OUT_WG 128
#endif

#if !IN_SIZEX
#define IN_SIZEX 16
#endif

#if !OUT_SIZEX
#define OUT_SIZEX 16
#endif

// The default is padding of 256 bytes for AMD GPUs, no padding otherwise.
#if !defined(PAD)
#if AMDGPU
#define PAD  256
#else
#define PAD  0
#endif
#endif
#define PAD_SIZE (PAD/16)          // Convert padding amount from bytes to number of T2 values

// The default setting for LDS transpose is on.  Only Intel battlemage is reported as faster without LDS transpose.
#if !defined(MIDDLE_IN_LDS_TRANSPOSE)
#define MIDDLE_IN_LDS_TRANSPOSE  1
#endif
#if !defined(MIDDLE_OUT_LDS_TRANSPOSE)
#define MIDDLE_OUT_LDS_TRANSPOSE 1
#endif

#if !INPLACE                       // Original implementation (not in place)

#if FFT_FP64 || NTT_GF61

//****************************************************************************************
// Pair of routines to write data from carryFused and read data into fftMiddleIn
//****************************************************************************************

// Optionally pad lines on output from fft_WIDTH in carryFused for input to fftMiddleIn.
// This lets fftMiddleIn read a more varied distribution of addresses.
// This can be faster on AMD GPUs, not certain about nVidia GPUs.

// writeCarryFusedLine writes:
//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)          (also known as 0...WG-1 and 0...NW-1)
//      line      ranges 0...BIG_HEIGHT-1 (multiples of one)
// fftMiddleIn reads:
//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)
//      u[i]      i ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...SMALL_HEIGHT-1 (multiples of one)

void OVERLOAD writeCarryFusedLine(T2 *u, P(T2) out, u32 line) {
#if PAD_SIZE > 0
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  out += line * WIDTH + line * PAD_SIZE + line / SMALL_HEIGHT * BIG_PAD_SIZE + (u32) get_local_id(0); // One pad every line + a big pad every SMALL_HEIGHT lines
  for (u32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W], u[i]); }
#else
  out += line * WIDTH + (u32) get_local_id(0);
  for (u32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W], u[i]); }
#endif
}

void OVERLOAD readMiddleInLine(T2 *u, CP(T2) in, u32 y, u32 x) {
#if PAD_SIZE > 0
  // Each work group reads successive y's which increments by one pad size.
  // Rather than having u[i] also increment by one, we choose a larger pad increment
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  in += y * WIDTH + y * PAD_SIZE + (y / SMALL_HEIGHT) * BIG_PAD_SIZE + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * (SMALL_HEIGHT * (WIDTH + PAD_SIZE) + BIG_PAD_SIZE)]); }
#else
  in += y * WIDTH + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SMALL_HEIGHT * WIDTH]); }
#endif
}

//****************************************************************************************
// Pair of routines to write data from fftMiddleIn and read data into tailFusedSquare/Mul
//****************************************************************************************

// fftMiddleIn processes:
//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)
//      u[i]      i ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...SMALL_HEIGHT-1 (multiples of one)
// tailFused reads:
//      x         ranges 0...SMALL_HEIGHT-1 (multiples of one)          (also known as 0...G_H-1 and 0...NH-1)
//      y         ranges 0...MIDDLE*WIDTH-1 (multiples of SMALL_HEIGHT)

void OVERLOAD writeMiddleInLine (P(T2) out, T2 *u, u32 chunk_y, u32 chunk_x)
{
  //u32 SIZEY = IN_WG / IN_SIZEX;
  //u32 num_x_chunks = WIDTH / IN_SIZEX;                // Number of x chunks
  //u32 num_y_chunks = SMALL_HEIGHT / SIZEY;            // Number of y chunks

#if PAD_SIZE > 0

  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * IN_WG + PAD_SIZE) +        // Write y chunks after middle chunks and a pad 
         chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX +  // num_y_chunks * (MIDDLE * IN_WG + PAD_SIZE)
                    SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE);
                                                        //              = SMALL_HEIGHT / SIZEY * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT / (IN_WG / IN_SIZEX) * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT * MIDDLE * IN_SIZEX + SMALL_HEIGHT / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * IN_WG], u[i]); }

#else

  // Output data such that readCarryFused lines are packed tightly together.  No padding.
  out += chunk_y * MIDDLE * IN_WG +                     // Write y chunks after middles
         chunk_x * MIDDLE * SMALL_HEIGHT * IN_SIZEX;    // num_y_chunks * IN_WG = SMALL_HEIGHT / SIZEY * MIDDLE * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT / (IN_WG / IN_SIZEX) * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT * IN_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * IN_WG], u[i]); }

#endif
}

// Read a line for tailFused or fftHin
// This reads partially transposed data as written by fftMiddleIn
void OVERLOAD readTailFusedLine(CP(T2) in, T2 *u, u32 line, u32 me) {
  u32 SIZEY = IN_WG / IN_SIZEX;

#if PAD_SIZE > 0

  // Adjust in pointer based on the x value used in writeMiddleInLine
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  u32 fftMiddleIn_x = line % WIDTH;                             // The fftMiddleIn x value
  u32 chunk_x = fftMiddleIn_x / IN_SIZEX;                       // The fftMiddleIn chunk_x value
  in += chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX + SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE); // Adjust in pointer the same way writeMiddleInLine did
  u32 x_within_in_wg = fftMiddleIn_x % IN_SIZEX;                // There were IN_SIZEX x values within IN_WG
  in += x_within_in_wg * SIZEY;                                 // Adjust in pointer the same way writeMiddleInLine wrote x values within IN_WG

  // Adjust in pointer based on the i value used in writeMiddleInLine
  u32 fftMiddleIn_i = line / WIDTH;                             // The i in fftMiddleIn's u[i]
  in += fftMiddleIn_i * IN_WG;                                  // Adjust in pointer the same way writeMiddleInLine did

  // Adjust in pointer based on the y value used in writeMiddleInLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleIn_y = me;                                       // The i=0 fftMiddleIn y value
  u32 chunk_y = fftMiddleIn_y / SIZEY;                          // The i=0 fftMiddleIn chunk_y value
  u32 fftMiddleIn_y_incr = G_H;                                 // The increment to next fftMiddleIn y value
  u32 chunk_y_incr = fftMiddleIn_y_incr / SIZEY;                // The increment to next fftMiddleIn chunk_y value
  for (i32 i = 0; i < NH; ++i) {
//    u32 fftMiddleIn_y = i * G_H + me;                         // The fftMiddleIn y value
//    u32 chunk_y = fftMiddleIn_y / SIZEY;                      // The fftMiddleIn chunk_y value
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * IN_WG + PAD_SIZE)]);   // Adjust in pointer the same way writeMiddleInLine did
    chunk_y += chunk_y_incr;
  }

#else                                                           // Read data that was not rotated or padded

  // Adjust in pointer based on the x value used in writeMiddleInLine
  u32 fftMiddleIn_x = line % WIDTH;                             // The fftMiddleIn x value
  u32 chunk_x = fftMiddleIn_x / IN_SIZEX;                       // The fftMiddleIn chunk_x value
  in += chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX);           // Adjust in pointer the same way writeMiddleInLine did
  u32 x_within_in_wg = fftMiddleIn_x % IN_SIZEX;                // There were IN_SIZEX x values within IN_WG
  in += x_within_in_wg * SIZEY;                                 // Adjust in pointer the same way writeMiddleInLine wrote x values within IN_WG

  // Adjust in pointer based on the i value used in writeMiddleInLine
  u32 fftMiddleIn_i = line / WIDTH;                             // The i in fftMiddleIn's u[i]
  in += fftMiddleIn_i * IN_WG;                                  // Adjust in pointer the same way writeMiddleInLine did

  // Adjust in pointer based on the y value used in writeMiddleInLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleIn_y = me;                                       // The i=0 fftMiddleIn y value
  u32 chunk_y = fftMiddleIn_y / SIZEY;                          // The i=0 fftMiddleIn chunk_y value
  u32 fftMiddleIn_y_incr = G_H;                                 // The increment to next fftMiddleIn y value
  u32 chunk_y_incr = fftMiddleIn_y_incr / SIZEY;                // The increment to next fftMiddleIn chunk_y value
  for (i32 i = 0; i < NH; ++i) {
    u32 fftMiddleIn_y = i * G_H + me;                           // The fftMiddleIn y value
    u32 chunk_y = fftMiddleIn_y / SIZEY;                        // The fftMiddleIn chunk_y value
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * IN_WG)]);              // Adjust in pointer the same way writeMiddleInLine did
    chunk_y += chunk_y_incr;
  }

#endif
}

//****************************************************************************************
// Pair of routines to write data from tailFusedSquare/Mul and read data into fftMiddleOut
//****************************************************************************************

// Optionally pad lines on output from fft_HEIGHT in tailFusedSquare/Mul for input to fftMiddleOut.
// This lets fftMiddleOut read a more varied distribution of addresses.
// This can be faster on AMD GPUs, not certain about nVidia GPUs.

// tailFused writes:
//      x         ranges 0...SMALL_HEIGHT-1 (multiples on one)          (also known as 0...G_H-1 and 0...NH-1)
//      y         ranges 0...MIDDLE*WIDTH-1 (multiples of SMALL_HEIGHT)
// fftMiddleOut reads:
//      x         ranges 0...SMALL_HEIGHT-1 (multiples of one)          (processed in batches of OUT_SIZEX)
//      i in u[i] ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)          (processed in batches of OUT_WG/OUT_SIZEX)

void OVERLOAD writeTailFusedLine(T2 *u, P(T2) out, u32 line, u32 me) {
#if PAD_SIZE > 0
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  out += line * (SMALL_HEIGHT + PAD_SIZE) + line / MIDDLE * BIG_PAD_SIZE + me; // Pad every output line plus every MIDDLE
#else
  out += line * (SMALL_HEIGHT + PAD_SIZE) + me;                         // Pad every output line
#endif
  for (u32 i = 0; i < NH; ++i) { NTSTORE(out[i * G_H], u[i]); }
#else                                                                   // No padding
  out += line * SMALL_HEIGHT + me;
  for (u32 i = 0; i < NH; ++i) { NTSTORE(out[i * G_H], u[i]); }
#endif
}

void OVERLOAD readMiddleOutLine(T2 *u, CP(T2) in, u32 y, u32 x) {
#if PAD_SIZE > 0
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  // Each u[i] increments by one pad size.
  // Rather than each work group reading successive y's also increment by one, we choose a larger pad increment.
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + y * BIG_PAD_SIZE + x;
#else
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + x;
#endif
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * (SMALL_HEIGHT + PAD_SIZE)]); }
#else                                                                   // No rotation, might be better on nVidia cards
  in += y * MIDDLE * SMALL_HEIGHT + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SMALL_HEIGHT]); }
#endif
}


//****************************************************************************************
// Pair of routines to write data from fftMiddleOut and read data into carryFused
//****************************************************************************************

// Write data from fftMiddleOut for consumption by carryFusedLine.
// We have the freedom to write the data anywhere in the output buffer,
// so we want to select locations that help speed up readCarryFusedLine.
//
// This gets complicated very fast, so I've documented my thought processes here.
// fftMiddleOut reads:
//      x         ranges 0...SMALL_HEIGHT-1 (multiples of one)          (processed in batches of OUT_SIZEX)
//      i in u[i] ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)          (processed in batches of OUT_WG/OUT_SIZEX)
// readCarryFusedLine reads:
//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)          (also known as 0...WG-1 and 0...NW-1)
//      line      ranges 0...BIG_HEIGHT-1 (multiples of one)
//
// Of note above, all the (y,x) values where x is unchanged are read by a single readCarryFusedLine call.
// Also, all the (y,x+1) values are read by the next readCarryFusedLine wavefront.  Since carryFused kernels are dispatched
// in ascending order, it is beneficial to group the (y,x+1) pairs immediately after the (y,x) pairs in case the (y,x) values
// are smaller than a full memory line.  The (y,x+1) pairs are then highly likely to be in the GPU's L2 cache.
//
// In the next sections we'll work through an example where WIDTH=1024, MIDDLE=15, SMALL_HEIGHT=256, OUT_WG=128, and OUT_SIZEX=16.
// The first order of business is for fftMiddleOut to contiguously write all data values that will be needed for a single readCarryFusedLine.
// That is, OUT_WG/OUT_SIZEX (y,x) values for the first x value (in our example this is 8 values - a value is data type T2 or 16 bytes, a total of 128 bytes).
// As noted above, we then output the (y,x+1) values, (y,x+2) and so on OUT_SIZEX times.  A total of 16 * 128 = 2KB in our example.
//
// The next memory layout decision is whether we should either a) output the next set of y values (readCarryFused lines are tightly packed together),
// or b) output the next set of x values (readCarryFused lines are spread out over a greater area) or c) the MIDDLE lines for sequential writes.
// In our example, readCarryFusedLine will read from WIDTH/8=128 different 2KB chunks.  128 2KB strides sounds scary to me.  To get a variety of
// strides we can rotate data within 2KB chunks or use a small padding less than 2KB.  A GPU is likely to prefer 128, 256, or 512 byte reads -- this
// limits the number of padding options in a 2KB chunk to 16, 8, or just 4.  If we go with option (b) or (c) we can rotate data over a larger
// area, but that is of no benefit as the stride will still be some multiple of 2KB.  If there are other bad stride values, (e.g. some CPUs don't like
// 64KB strides) that could impact our decision here (e.g. option (c) with MIDDLE=16 would result in a 32KB stride).
//
// If we output all MIDDLE i values after the x and y values, there will be a huge power-of-two stride between these writes.
// This is a problem on Radeon VII.  Another padding is necessary.
//
// After experimentation, we've chosen to output the MIDDLE values next with padding (padding is simpler code than rotation).
// Other options are workable with no measurable degradation in performance.

// Caller must either give us u values that are grouped by x values (i.e. the order in which they were read in) with the out pointer
// adjusted to effect a transpose.  Or caller must transpose the x and y values and send us an out pointer with thread_id added in.
// In other words, caller is responsible for deciding the best way to transpose x and y values.

void OVERLOAD writeMiddleOutLine (P(T2) out, T2 *u, u32 chunk_y, u32 chunk_x)
{
  //u32 SIZEY = OUT_WG / OUT_SIZEX;
  //u32 num_x_chunks = SMALL_HEIGHT / OUT_SIZEX;  // Number of x chunks
  //u32 num_y_chunks = WIDTH / SIZEY;             // Number of y chunks

#if PAD_SIZE > 0

  u32 SIZEY = OUT_WG / OUT_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * OUT_WG + PAD_SIZE) +       // Write y chunks after middle chunks and a pad 
         chunk_x * (WIDTH * MIDDLE * OUT_SIZEX +        // num_y_chunks * (MIDDLE * OUT_WG + PAD_SIZE)
                    WIDTH / SIZEY * PAD_SIZE + BIG_PAD_SIZE);//         = WIDTH / SIZEY * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH / (OUT_WG / OUT_SIZEX) * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH * MIDDLE * OUT_SIZEX + WIDTH / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * OUT_WG], u[i]); }

#else

  // Output data such that readCarryFused lines are packed tightly together.  No padding.
  out += chunk_y * MIDDLE * OUT_WG +             // Write y chunks after middles
         chunk_x * MIDDLE * WIDTH * OUT_SIZEX;   // num_y_chunks * OUT_WG = WIDTH / SIZEY * MIDDLE * OUT_WG
                                        //                       = MIDDLE * WIDTH / (OUT_WG / OUT_SIZEX) * OUT_WG
                                        //                       = MIDDLE * WIDTH * OUT_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * OUT_WG], u[i]); }

#endif
}

// Read a line for carryFused or FFTW.  This line was written by writeMiddleOutLine above.
void OVERLOAD readCarryFusedLine(CP(T2) in, T2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 SIZEY = OUT_WG / OUT_SIZEX;

#if PAD_SIZE > 0

  // Adjust in pointer based on the x value used in writeMiddleOutLine
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  u32 fftMiddleOut_x = line % SMALL_HEIGHT;                     // The fftMiddleOut x value
  u32 chunk_x = fftMiddleOut_x / OUT_SIZEX;                     // The fftMiddleOut chunk_x value
  in += chunk_x * (WIDTH * MIDDLE * OUT_SIZEX + WIDTH / SIZEY * PAD_SIZE + BIG_PAD_SIZE); // Adjust in pointer the same way writeMiddleOutLine did
  u32 x_within_out_wg = fftMiddleOut_x % OUT_SIZEX;             // There were OUT_SIZEX x values within OUT_WG
  in += x_within_out_wg * SIZEY;                                // Adjust in pointer the same way writeMiddleOutLine wrote x values within OUT_WG

  // Adjust in pointer based on the i value used in writeMiddleOutLine
  u32 fftMiddleOut_i = line / SMALL_HEIGHT;                     // The i in fftMiddleOut's u[i]
  in += fftMiddleOut_i * OUT_WG;                                // Adjust in pointer the same way writeMiddleOutLine did

  // Adjust in pointer based on the y value used in writeMiddleOutLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleOut_y = me;                                      // The i=0 fftMiddleOut y value
  u32 chunk_y = fftMiddleOut_y / SIZEY;                         // The i=0 fftMiddleOut chunk_y value
  u32 fftMiddleOut_y_incr = G_W;                                // The increment to next fftMiddleOut y value
  u32 chunk_y_incr = fftMiddleOut_y_incr / SIZEY;               // The increment to next fftMiddleOut chunk_y value
  for (i32 i = 0; i < NW; ++i) {
//    u32 fftMiddleOut_y = i * G_W + me;                        // The fftMiddleOut y value
//    u32 chunk_y = fftMiddleOut_y / SIZEY;                     // The fftMiddleOut chunk_y value
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * OUT_WG + PAD_SIZE)]);  // Adjust in pointer the same way writeMiddleOutLine did
    chunk_y += chunk_y_incr;
  }

#else                                                           // Read data that was not rotated or padded

  // Adjust in pointer based on the x value used in writeMiddleOutLine
  u32 fftMiddleOut_x = line % SMALL_HEIGHT;                     // The fftMiddleOut x value
  u32 chunk_x = fftMiddleOut_x / OUT_SIZEX;                     // The fftMiddleOut chunk_x value
  in += chunk_x * MIDDLE * WIDTH * OUT_SIZEX;                   // Adjust in pointer the same way writeMiddleOutLine did
  u32 x_within_out_wg = fftMiddleOut_x % OUT_SIZEX;             // There were OUT_SIZEX x values within OUT_WG
  in += x_within_out_wg * SIZEY;                                // Adjust in pointer the same way writeMiddleOutLine wrote x values with OUT_WG

  // Adjust in pointer based on the i value used in writeMiddleOutLine
  u32 fftMiddleOut_i = line / SMALL_HEIGHT;                     // The i in fftMiddleOut's u[i]
  in += fftMiddleOut_i * OUT_WG;                                // Adjust in pointer the same way writeMiddleOutLine did

  // Adjust in pointer based on the y value used in writeMiddleOutLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleOut_y = me;                                      // The i=0 fftMiddleOut y value
  u32 chunk_y = fftMiddleOut_y / SIZEY;                         // The i=0 fftMiddleOut chunk_y value
  u32 fftMiddleOut_y_incr = G_W;                                // The increment to next fftMiddleOut y value
  u32 chunk_y_incr = fftMiddleOut_y_incr / SIZEY;               // The increment to next fftMiddleOut chunk_y value
  for (i32 i = 0; i < NW; ++i) {
//    u32 fftMiddleOut_y = i * G_W + me;                          // The fftMiddleOut y value
//    u32 chunk_y = fftMiddleOut_y / SIZEY;                       // The fftMiddleOut chunk_y value
    u[i] = NTLOAD(in[chunk_y * MIDDLE * OUT_WG]);               // Adjust in pointer the same way writeMiddleOutLine did
    chunk_y += chunk_y_incr;
  }

#endif

}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32 || NTT_GF31

void OVERLOAD writeCarryFusedLine(F2 *u, P(F2) out, u32 line) {
#if PAD_SIZE > 0
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  out += line * WIDTH + line * PAD_SIZE + line / SMALL_HEIGHT * BIG_PAD_SIZE + (u32) get_local_id(0); // One pad every line + a big pad every SMALL_HEIGHT lines
  for (u32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W], u[i]); }
#else
  out += line * WIDTH + (u32) get_local_id(0);
  for (u32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W], u[i]); }
#endif
}

void OVERLOAD readMiddleInLine(F2 *u, CP(F2) in, u32 y, u32 x) {
#if PAD_SIZE > 0
  // Each work group reads successive y's which increments by one pad size.
  // Rather than having u[i] also increment by one, we choose a larger pad increment
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  in += y * WIDTH + y * PAD_SIZE + (y / SMALL_HEIGHT) * BIG_PAD_SIZE + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * (SMALL_HEIGHT * (WIDTH + PAD_SIZE) + BIG_PAD_SIZE)]); }
#else
  in += y * WIDTH + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SMALL_HEIGHT * WIDTH]); }
#endif
}

void OVERLOAD writeMiddleInLine (P(F2) out, F2 *u, u32 chunk_y, u32 chunk_x)
{
#if PAD_SIZE > 0
  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * IN_WG + PAD_SIZE) +        // Write y chunks after middle chunks and a pad 
         chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX +  // num_y_chunks * (MIDDLE * IN_WG + PAD_SIZE)
                    SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE);
                                                        //              = SMALL_HEIGHT / SIZEY * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT / (IN_WG / IN_SIZEX) * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT * MIDDLE * IN_SIZEX + SMALL_HEIGHT / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * IN_WG], u[i]); }
#else
  // Output data such that readCarryFused lines are packed tightly together.  No padding.
  out += chunk_y * MIDDLE * IN_WG +                     // Write y chunks after middles
         chunk_x * MIDDLE * SMALL_HEIGHT * IN_SIZEX;    // num_y_chunks * IN_WG = SMALL_HEIGHT / SIZEY * MIDDLE * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT / (IN_WG / IN_SIZEX) * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT * IN_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * IN_WG], u[i]); }
#endif
}

// Read a line for tailFused or fftHin
// This reads partially transposed data as written by fftMiddleIn
void OVERLOAD readTailFusedLine(CP(F2) in, F2 *u, u32 line, u32 me) {
  u32 SIZEY = IN_WG / IN_SIZEX;
#if PAD_SIZE > 0
  // Adjust in pointer based on the x value used in writeMiddleInLine
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  u32 fftMiddleIn_x = line % WIDTH;                             // The fftMiddleIn x value
  u32 chunk_x = fftMiddleIn_x / IN_SIZEX;                       // The fftMiddleIn chunk_x value
  in += chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX + SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE); // Adjust in pointer the same way writeMiddleInLine did
  u32 x_within_in_wg = fftMiddleIn_x % IN_SIZEX;                // There were IN_SIZEX x values within IN_WG
  in += x_within_in_wg * SIZEY;                                 // Adjust in pointer the same way writeMiddleInLine wrote x values within IN_WG

  // Adjust in pointer based on the i value used in writeMiddleInLine
  u32 fftMiddleIn_i = line / WIDTH;                             // The i in fftMiddleIn's u[i]
  in += fftMiddleIn_i * IN_WG;                                  // Adjust in pointer the same way writeMiddleInLine did
  // Adjust in pointer based on the y value used in writeMiddleInLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleIn_y = me;                                       // The i=0 fftMiddleIn y value
  u32 chunk_y = fftMiddleIn_y / SIZEY;                          // The i=0 fftMiddleIn chunk_y value
  u32 fftMiddleIn_y_incr = G_H;                                 // The increment to next fftMiddleIn y value
  u32 chunk_y_incr = fftMiddleIn_y_incr / SIZEY;                // The increment to next fftMiddleIn chunk_y value
  for (i32 i = 0; i < NH; ++i) {
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * IN_WG + PAD_SIZE)]);   // Adjust in pointer the same way writeMiddleInLine did
    chunk_y += chunk_y_incr;
  }
#else                                                           // Read data that was not rotated or padded
  // Adjust in pointer based on the x value used in writeMiddleInLine
  u32 fftMiddleIn_x = line % WIDTH;                             // The fftMiddleIn x value
  u32 chunk_x = fftMiddleIn_x / IN_SIZEX;                       // The fftMiddleIn chunk_x value
  in += chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX);           // Adjust in pointer the same way writeMiddleInLine did
  u32 x_within_in_wg = fftMiddleIn_x % IN_SIZEX;                // There were IN_SIZEX x values within IN_WG
  in += x_within_in_wg * SIZEY;                                 // Adjust in pointer the same way writeMiddleInLine wrote x values within IN_WG
  // Adjust in pointer based on the i value used in writeMiddleInLine
  u32 fftMiddleIn_i = line / WIDTH;                             // The i in fftMiddleIn's u[i]
  in += fftMiddleIn_i * IN_WG;                                  // Adjust in pointer the same way writeMiddleInLine did
  // Adjust in pointer based on the y value used in writeMiddleInLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleIn_y = me;                                       // The i=0 fftMiddleIn y value
  u32 chunk_y = fftMiddleIn_y / SIZEY;                          // The i=0 fftMiddleIn chunk_y value
  u32 fftMiddleIn_y_incr = G_H;                                 // The increment to next fftMiddleIn y value
  u32 chunk_y_incr = fftMiddleIn_y_incr / SIZEY;                // The increment to next fftMiddleIn chunk_y value
  for (i32 i = 0; i < NH; ++i) {
    u32 fftMiddleIn_y = i * G_H + me;                           // The fftMiddleIn y value
    u32 chunk_y = fftMiddleIn_y / SIZEY;                        // The fftMiddleIn chunk_y value
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * IN_WG)]);              // Adjust in pointer the same way writeMiddleInLine did
    chunk_y += chunk_y_incr;
  }
#endif
}

void OVERLOAD writeTailFusedLine(F2 *u, P(F2) out, u32 line, u32 me) {
#if PAD_SIZE > 0
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  out += line * (SMALL_HEIGHT + PAD_SIZE) + line / MIDDLE * BIG_PAD_SIZE + me; // Pad every output line plus every MIDDLE
#else
  out += line * (SMALL_HEIGHT + PAD_SIZE) + me;                         // Pad every output line
#endif
  for (u32 i = 0; i < NH; ++i) { NTSTORE(out[i * G_H], u[i]); }
#else                                                                   // No padding
  out += line * SMALL_HEIGHT + me;
  for (u32 i = 0; i < NH; ++i) { NTSTORE(out[i * G_H], u[i]); }
#endif
}

void OVERLOAD readMiddleOutLine(F2 *u, CP(F2) in, u32 y, u32 x) {
#if PAD_SIZE > 0
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  // Each u[i] increments by one pad size.
  // Rather than each work group reading successive y's also increment by one, we choose a larger pad increment.
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + y * BIG_PAD_SIZE + x;
#else
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + x;
#endif
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * (SMALL_HEIGHT + PAD_SIZE)]); }
#else                                                                   // No rotation, might be better on nVidia cards
  in += y * MIDDLE * SMALL_HEIGHT + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SMALL_HEIGHT]); }
#endif
}

void OVERLOAD writeMiddleOutLine (P(F2) out, F2 *u, u32 chunk_y, u32 chunk_x)
{
#if PAD_SIZE > 0
  u32 SIZEY = OUT_WG / OUT_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * OUT_WG + PAD_SIZE) +       // Write y chunks after middle chunks and a pad 
         chunk_x * (WIDTH * MIDDLE * OUT_SIZEX +        // num_y_chunks * (MIDDLE * OUT_WG + PAD_SIZE)
                    WIDTH / SIZEY * PAD_SIZE + BIG_PAD_SIZE);//         = WIDTH / SIZEY * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH / (OUT_WG / OUT_SIZEX) * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH * MIDDLE * OUT_SIZEX + WIDTH / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * OUT_WG], u[i]); }
#else
  // Output data such that readCarryFused lines are packed tightly together.  No padding.
  out += chunk_y * MIDDLE * OUT_WG +             // Write y chunks after middles
         chunk_x * MIDDLE * WIDTH * OUT_SIZEX;   // num_y_chunks * OUT_WG = WIDTH / SIZEY * MIDDLE * OUT_WG
                                        //                       = MIDDLE * WIDTH / (OUT_WG / OUT_SIZEX) * OUT_WG
                                        //                       = MIDDLE * WIDTH * OUT_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * OUT_WG], u[i]); }
#endif
}

void OVERLOAD readCarryFusedLine(CP(F2) in, F2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 SIZEY = OUT_WG / OUT_SIZEX;
#if PAD_SIZE > 0
  // Adjust in pointer based on the x value used in writeMiddleOutLine
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  u32 fftMiddleOut_x = line % SMALL_HEIGHT;                     // The fftMiddleOut x value
  u32 chunk_x = fftMiddleOut_x / OUT_SIZEX;                     // The fftMiddleOut chunk_x value
  in += chunk_x * (WIDTH * MIDDLE * OUT_SIZEX + WIDTH / SIZEY * PAD_SIZE + BIG_PAD_SIZE); // Adjust in pointer the same way writeMiddleOutLine did
  u32 x_within_out_wg = fftMiddleOut_x % OUT_SIZEX;             // There were OUT_SIZEX x values within OUT_WG
  in += x_within_out_wg * SIZEY;                                // Adjust in pointer the same way writeMiddleOutLine wrote x values within OUT_WG
  // Adjust in pointer based on the i value used in writeMiddleOutLine
  u32 fftMiddleOut_i = line / SMALL_HEIGHT;                     // The i in fftMiddleOut's u[i]
  in += fftMiddleOut_i * OUT_WG;                                // Adjust in pointer the same way writeMiddleOutLine did
  // Adjust in pointer based on the y value used in writeMiddleOutLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleOut_y = me;                                      // The i=0 fftMiddleOut y value
  u32 chunk_y = fftMiddleOut_y / SIZEY;                         // The i=0 fftMiddleOut chunk_y value
  u32 fftMiddleOut_y_incr = G_W;                                // The increment to next fftMiddleOut y value
  u32 chunk_y_incr = fftMiddleOut_y_incr / SIZEY;               // The increment to next fftMiddleOut chunk_y value
  for (i32 i = 0; i < NW; ++i) {
    u[i] = NTLOAD(in[chunk_y * (MIDDLE * OUT_WG + PAD_SIZE)]);  // Adjust in pointer the same way writeMiddleOutLine did
    chunk_y += chunk_y_incr;
  }
#else                                                           // Read data that was not rotated or padded
  // Adjust in pointer based on the x value used in writeMiddleOutLine
  u32 fftMiddleOut_x = line % SMALL_HEIGHT;                     // The fftMiddleOut x value
  u32 chunk_x = fftMiddleOut_x / OUT_SIZEX;                     // The fftMiddleOut chunk_x value
  in += chunk_x * MIDDLE * WIDTH * OUT_SIZEX;                   // Adjust in pointer the same way writeMiddleOutLine did
  u32 x_within_out_wg = fftMiddleOut_x % OUT_SIZEX;             // There were OUT_SIZEX x values within OUT_WG
  in += x_within_out_wg * SIZEY;                                // Adjust in pointer the same way writeMiddleOutLine wrote x values with OUT_WG
  // Adjust in pointer based on the i value used in writeMiddleOutLine
  u32 fftMiddleOut_i = line / SMALL_HEIGHT;                     // The i in fftMiddleOut's u[i]
  in += fftMiddleOut_i * OUT_WG;                                // Adjust in pointer the same way writeMiddleOutLine did
  // Adjust in pointer based on the y value used in writeMiddleOutLine.  This code is a little obscure as rocm compiler has trouble optimizing commented out code.
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  u32 fftMiddleOut_y = me;                                      // The i=0 fftMiddleOut y value
  u32 chunk_y = fftMiddleOut_y / SIZEY;                         // The i=0 fftMiddleOut chunk_y value
  u32 fftMiddleOut_y_incr = G_W;                                // The increment to next fftMiddleOut y value
  u32 chunk_y_incr = fftMiddleOut_y_incr / SIZEY;               // The increment to next fftMiddleOut chunk_y value
  for (i32 i = 0; i < NW; ++i) {
    u[i] = NTLOAD(in[chunk_y * MIDDLE * OUT_WG]);               // Adjust in pointer the same way writeMiddleOutLine did
    chunk_y += chunk_y_incr;
  }
#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Since F2 and GF31 are the same size we can simply call the floats based code

void OVERLOAD writeCarryFusedLine(GF31 *u, P(GF31) out, u32 line) {
  writeCarryFusedLine((F2 *) u, (P(F2)) out, line);
}

void OVERLOAD readMiddleInLine(GF31 *u, CP(GF31) in, u32 y, u32 x) {
  readMiddleInLine((F2 *) u, (CP(F2)) in, y, x);
}

void OVERLOAD writeMiddleInLine (P(GF31) out, GF31 *u, u32 chunk_y, u32 chunk_x) {
  writeMiddleInLine ((P(F2)) out, (F2 *) u, chunk_y, chunk_x);
}

void OVERLOAD readTailFusedLine(CP(GF31) in, GF31 *u, u32 line, u32 me) {
  readTailFusedLine((CP(F2)) in, (F2 *) u, line, me);
}

void OVERLOAD writeTailFusedLine(GF31 *u, P(GF31) out, u32 line, u32 me) {
  writeTailFusedLine((F2 *) u, (P(F2)) out, line, me);
}

void OVERLOAD readMiddleOutLine(GF31 *u, CP(GF31) in, u32 y, u32 x) {
  readMiddleOutLine((F2 *) u, (CP(F2)) in, y, x);
}

void OVERLOAD writeMiddleOutLine (P(GF31) out, GF31 *u, u32 chunk_y, u32 chunk_x) {
  writeMiddleOutLine ((P(F2)) out, (F2 *) u, chunk_y, chunk_x);
}

void OVERLOAD readCarryFusedLine(CP(GF31) in, GF31 *u, u32 line) {
  readCarryFusedLine((CP(F2)) in, (F2 *) u, line);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Since T2 and GF61 are the same size we can simply call the doubles based code

void OVERLOAD writeCarryFusedLine(GF61 *u, P(GF61) out, u32 line) {
  writeCarryFusedLine((T2 *) u, (P(T2)) out, line);
}

void OVERLOAD readMiddleInLine(GF61 *u, CP(GF61) in, u32 y, u32 x) {
  readMiddleInLine((T2 *) u, (CP(T2)) in, y, x);
}

void OVERLOAD writeMiddleInLine (P(GF61) out, GF61 *u, u32 chunk_y, u32 chunk_x) {
  writeMiddleInLine ((P(T2)) out, (T2 *) u, chunk_y, chunk_x);
}

void OVERLOAD readTailFusedLine(CP(GF61) in, GF61 *u, u32 line, u32 me) {
  readTailFusedLine((CP(T2)) in, (T2 *) u, line, me);
}

void OVERLOAD writeTailFusedLine(GF61 *u, P(GF61) out, u32 line, u32 me) {
  writeTailFusedLine((T2 *) u, (P(T2)) out, line, me);
}

void OVERLOAD readMiddleOutLine(GF61 *u, CP(GF61) in, u32 y, u32 x) {
  readMiddleOutLine((T2 *) u, (CP(T2)) in, y, x);
}

void OVERLOAD writeMiddleOutLine (P(GF61) out, GF61 *u, u32 chunk_y, u32 chunk_x) {
  writeMiddleOutLine ((P(T2)) out, (T2 *) u, chunk_y, chunk_x);
}

void OVERLOAD readCarryFusedLine(CP(GF61) in, GF61 *u, u32 line) {
  readCarryFusedLine((CP(T2)) in, (T2 *) u, line);
}

#endif






#else                           // New implementation (in-place)

// Goals:
// 1) In-place transpose.  Rather than "ping-pong"ing buffers, an in-place transpose uses half as much memory.  This may allow
//    the entire FFT/NTT data set to reside in the L2 cache on upper end consumer GPUs (circa 2025) which can have 64MB or larger L2 caches.
// 2) We want to have distribute the carryFused and/or tailSquare memory in the L2 cache with minimal cache line collisions.  The hope is to (one day) do
//    fftMiddleOut/carryFused/fftMiddleIn or fftMiddleIn/tailSquare/fftMiddleOut in L2 cache-sized chunks to minimize the slowest memory accesses.
//    The cost of extra kernel launches may negate any L2 cache benefits.
// 3) We use swizzling and/or modest padding to reduce carryFused L2 cache line collisions.  Several different memory layouts and padding were tried
//    on nVidia Titan V and AMD Radeon VII to find the fastest in-place layout and padding scheme.  Hopefully, these will schemes will work well
//    on later generation GPUs with different L2 cache dimensions (size and "number-of-ways").
// 4) Apparently cache line collisions in the L1 cache also adversely affect timings.  The L1 cache may have a different cache line size and number-of-ways
//    which makes padding tuning a bit difficult.  This is especially true on AMD which has a very strange channel & banks partitioning of memory accesses.
// 5) Manufacturers are not very good about documenting L1 & L2 cache configurations.  nVidia GPUs seem to have a L2 cache line size of 128 bytes.
//    AMD documentation seems to indicate 64 or 128 byte cache lines.  However, other documentation indicates 256 byte reads are to be preferred.
//    Clinfo on an rx9070 says the L2 cache line size is 256 bytes.  Thus, our goal is to target cache line size of 256 bytes.
//
// Here is the proposed memory layout for a 512 x 8 x 512 = 2M complex FFT using two doubles (16 bytes).  PRPLL calls this a 4M FFT for the end user.
// The in-place transpose done by fftMiddleIn and fftMiddleOut works on a grid of 16 values from carryFused (multiples of 4K) and 16 values
// from tailSquare (multiples of 1).  Sixteen 16-byte values = one 256 byte cache line or two 128 byte cache lines.
//
// tailSquare memory layout (also fftMiddleIn output layout and fftMiddleOut input layout).
// A HEIGHT=512 tail line is 512*16 bytes = 8KB.  There are 2M/512=4K lines.  Lines k and N-k are processed together.
// A 2MB L2 cache can fit 256 tail lines.  So the first group of tail lines output by fftMiddleIn can be (mostly) paired with the last group of
// tail lines output by fftMiddleIn for tailSquare to process.  128 tail lines = 64K FFT values = 1MB.
// Here is the memory layout of FFT data.
//      0..511        The first tailSquare line (tail line 0).  SMALL_HEIGHT values * 16 bytes (8KB).
//      4K            The tailSquare line starting with FFT data element 4K (tail line 1).
//      ..
//      60K           These 16 lines will form 16x16 "transpose blocks".
//      64K           Follow the same pattern starting at FFT data element 64K (tail line 16).  Note: We have tried placing the "middle" lines next.
//      ...
//      2M-4K         The last "width" line (tail line MIDDLE*(WIDTH-1)).
//      512..         The tailSquare line starting with the first "middle" value (tail line WIDTH, i.e. 512), followed again by WIDTH lines 4K apart.
//      ...
//      3.5K          The tailSquare line containing the last middle value that fftMiddleOut will need (tail line (MIDDLE-1)*WIDTH)
//      ...
//
// fftMiddleOut uses 16 sets of 16 threads to transpose 16x16 blocks of 16-byte values.  Those 256 threads also process the MIDDLE blocks. For example,
// the 256 threads read 0,1..15, 4K+0,4K+1..4K+15, ..., 60K+0..60K+15 to form one 16x16 block.  For MIDDLE processing, those 256 threads also read
// the seven 16x16 blocks beginning at 512, 1024, ... 3584.
//
// After transposing, the memory layout of fftMiddleOut output (also carryFused input & output and fftMiddleIn layout) is:
//      0,4K..60K  16..16+60K, 32..32+60K, ..., 496..496+60K       (SMALL_HEIGHT/16=32) groups of 16 values * 16 bytes (8KB)
//      ...
//      15..
//      64K           After the above 16 lines (128KB), follow the same pattern starting at FFT data element 64K
//      ...
//      2.5M-64K
//      512..         After WIDTH lines above, follow the same pattern starting at the first "middle" value
//      ...
//      3.5K          The last "middle" value
//      ...
//
// carryFused reads FFT data values that are 4K apart.  The first 16 are in a single 256 byte cache line.  The next 16 (starting at 64K) occurs 128KB later.
// This large power of two stride could cause cache collisions depending on the cache layout.  If so, padding or swizzling MAY be very useful.
// carryFused does not reuse any read in data, so in theory its no big deal if a cache line is evicted before writing the result back out.
// I'm not sure if cache hardware cares about having multiple reads to the same cache line in-flight.  NOTE: One minor advantage to eliminating cache
// conflicts is that carryFused likely processes lines in order or close to it.  If we next process MiddleIn/tailSquare/MiddleOut in a 2MB cache as
// described above, it will appreciate the last 128 lines being in the L2 cache from carryfused.
//
// Back to carryfused.  What might be an optimal padding scheme?  Say that 64 carryfuseds are active simultaneously (probably more).
//      The +1s are +8KB
//      The +16s are +256B
//      The +64Ks are at +128KB
// This leaves the "columns" starting at +1KB unused - suggesting a pad of +1KB before the 64Ks would yield a better distribution in the L2 cache.
// However, if we have say an 8-way 16MB L2 cache then each way contains 2MB.  If so, we'd want to pad 1KB before the 16th 64K FFT data value.

#if FFT_FP64 || NTT_GF61

//****************************************************************************************
// Pair of routines to read/write data to/from carryFused
//****************************************************************************************

#if INPLACE == 1                                               // nVidia friendly padding
// Place middle rows after first 16 rows
//#define SIZEBLK (SMALL_HEIGHT + 16)                            // Pad 256 bytes
//#define SIZEW   (16 * SIZEA + 16)                              // Pad 256 bytes
//#define SIZEM   (MIDDLE * SIZEB + (1 - (MIDDLE & 1)) * 16)     // Pad 256 bytes if MIDDLE is even
// Place middle rows after all width rows
#define SIZEBLK   (SMALL_HEIGHT + 0)                           // No pad needed when swizzling
#define SIZEW   (16 * SIZEBLK + 16)                            // Pad 256 bytes
#define SIZEM   (WIDTH / 16 * SIZEW + 16)                      // Pad 256 bytes
#define SWIZ(a,m) ((m) ^ (a))                                  // Swizzle 16 rows (remove "^ (a)" to turn swizzling off)
#else                                                          // AMD friendly padding
// Place middle rows after first 16 rows
//#define SIZEBLK (SMALL_HEIGHT + 16)                            // Pad 256 bytes
//#define SIZEM   (16 * SIZEBLK + 16)                            // Pad 256 bytes
//#define SIZEW   (MIDDLE * SIZEM + (1 - (MIDDLE & 1)) * 16)     // Pad 256 bytes if MIDDLE is even
// Place middle rows after all width rows
#define SIZEBLK (SMALL_HEIGHT + 0)                             // No pad needed when swizzling
#define SIZEW   (16 * SIZEBLK + 16)                            // Pad 256 bytes
#define SIZEM   (WIDTH / 16 * SIZEW + 0)                       // Pad 0 bytes
#define SWIZ(a,m) ((m) ^ (a))                                  // Swizzle 16 rows (remove "^ (a)" to turn swizzling off)
#endif

//      me        ranges 0...WIDTH/NW-1 (multiples of BIG_HEIGHT)
//      u[i]      ranges 0..NW-1 (big multiples of BIG_HEIGHT) 
//      line      ranges 0...BIG_HEIGHT-1 (multiples of one)

// Read a line for carryFused or FFTW.  This line was written by writeMiddleOutLine above.
void OVERLOAD readCarryFusedLine(CP(T2) in, T2 *u, u32 line) {
  u32 me = get_local_id(0);             // Multiples of BIG_HEIGHT
  u32 middle = line / SMALL_HEIGHT;     // Multiples of SMALL_HEIGHT
  line = line % SMALL_HEIGHT;           // Multiples of one
  in += (me / 16 * SIZEW) + (middle * SIZEM) + (line % 16 * SIZEBLK) + SWIZ(line % 16, line / 16) * 16 + (me % 16);
  for (u32 i = 0; i < NW; ++i) { u[i] = NTLOAD(in[i * G_W / 16 * SIZEW]); }
}

// Write a line from carryFused.  This data will be read by fftMiddleIn.
void OVERLOAD writeCarryFusedLine(T2 *u, P(T2) out, u32 line) {
  u32 me = get_local_id(0);             // Multiples of BIG_HEIGHT
  u32 middle = line / SMALL_HEIGHT;     // Multiples of SMALL_HEIGHT
  line = line % SMALL_HEIGHT;           // Multiples of one
  out += (me / 16 * SIZEW) + (middle * SIZEM) + (line % 16 * SIZEBLK) + SWIZ(line % 16, line / 16) * 16 + (me % 16);
  for (i32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W / 16 * SIZEW], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from fftMiddleIn
//****************************************************************************************

//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)
//      u[i]      ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...SMALL_HEIGHT-1 (multiples of one)

void OVERLOAD readMiddleInLine(T2 *u, CP(T2) in, u32 y, u32 x) {
  in += (x / 16 * SIZEW) + (y % 16 * SIZEBLK) + (SWIZ(y % 16, y / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SIZEM]); }
}

// NOTE:  writeMiddleInLine uses the same definition of x,y as readMiddleInLine.  Caller transposes 16x16 blocks of FFT data before calling writeMiddleInLine.
void OVERLOAD writeMiddleInLine (P(T2) out, T2 *u, u32 y, u32 x)
{
  out += (x / 16 * SIZEW) + (y % 16 * SIZEBLK) + (SWIZ(y % 16, y / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * SIZEM], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from tailSquare/Mul
//****************************************************************************************

//      me        ranges 0...SMALL_HEIGHT/NH-1 (multiples of one)
//      u[i]      ranges 0...NH-1 (big multiples of one)
//      line      ranges 0...MIDDLE*WIDTH-1 (multiples of SMALL_HEIGHT)

// Read a line for tailSquare/Mul or fftHin
void OVERLOAD readTailFusedLine(CP(T2) in, T2 *u, u32 line, u32 me) {
  u32 width = line % WIDTH;            // Multiples of BIG_HEIGHT
  u32 middle = line / WIDTH;           // Multiples of SMALL_HEIGHT
  in += (width / 16 * SIZEW) + (middle * SIZEM) + (width % 16 * SIZEBLK) + (me % 16);
  for (i32 i = 0; i < NH; ++i) { u[i] = NTLOAD(in[SWIZ(width % 16, (i * SMALL_HEIGHT / NH + me) / 16) * 16]); }
}

void OVERLOAD writeTailFusedLine(T2 *u, P(T2) out, u32 line, u32 me) {
  u32 width = line % WIDTH;            // Multiples of BIG_HEIGHT
  u32 middle = line / WIDTH;           // Multiples of SMALL_HEIGHT
  out += (width / 16 * SIZEW) + (middle * SIZEM) + (width % 16 * SIZEBLK) + (me % 16);
  for (i32 i = 0; i < NH; ++i) { NTSTORE(out[SWIZ(width % 16, (i * SMALL_HEIGHT / NH + me) / 16) * 16], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from fftMiddleOut
//****************************************************************************************

//      x         ranges 0...SMALL_HEIGHT-1 (multiples of one)
//      u[i]      ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)

void OVERLOAD readMiddleOutLine(T2 *u, CP(T2) in, u32 y, u32 x) {
  in += (y / 16 * SIZEW) + (y % 16 * SIZEBLK) + (SWIZ(y % 16, x / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SIZEM]); }
}

// NOTE:  writeMiddleOutLine uses the same definition of x,y as readMiddleOutLine.  Caller transposes 16x16 blocks of FFT data before calling writeMiddleOutLine.
void OVERLOAD writeMiddleOutLine (P(T2) out, T2 *u, u32 y, u32 x)
{
  out += (y / 16 * SIZEW) + (y % 16 * SIZEBLK) + (SWIZ(y % 16, x / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * SIZEM], u[i]); }
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32 || NTT_GF31

//****************************************************************************************
// Pair of routines to read/write data to/from carryFused
//****************************************************************************************

// NOTE: I hven't studied best padding/swizzling/memory layout for 32-bit values.  I'm assuming the 64-bit values scheme will be pretty good.
#if INPLACE == 1                                               // nVidia friendly padding
// Place middle rows after first 16 rows
//#define SIZEBLK32 (SMALL_HEIGHT + 16)                          // Pad 128 bytes
//#define SIZEW32   (16 * SIZEA + 16)                            // Pad 128 bytes
//#define SIZEM32   (MIDDLE * SIZEB + (1 - (MIDDLE & 1)) * 16)   // Pad 128 bytes if MIDDLE is even
// Place middle rows after all width rows
#define SIZEBLK32   (SMALL_HEIGHT + 0)                         // No pad needed when swizzling
#define SIZEW32   (16 * SIZEBLK + 16)                          // Pad 128 bytes
#define SIZEM32   (WIDTH / 16 * SIZEW + 16)                    // Pad 128 bytes
#define SWIZ32(a,m) ((m) ^ (a))                                // Swizzle 16 rows (remove "^ (a)" to turn swizzling off)
#else                                                          // AMD friendly padding
// Place middle rows after first 16 rows
//#define SIZEBLK32 (SMALL_HEIGHT + 16)                          // Pad 128 bytes
//#define SIZEM32   (16 * SIZEBLK + 16)                          // Pad 128 bytes
//#define SIZEW32   (MIDDLE * SIZEM + (1 - (MIDDLE & 1)) * 16)   // Pad 128 bytes if MIDDLE is even
// Place middle rows after all width rows
#define SIZEBLK32 (SMALL_HEIGHT + 0)                           // No pad needed when swizzling
#define SIZEW32   (16 * SIZEBLK + 16)                          // Pad 128 bytes
#define SIZEM32   (WIDTH / 16 * SIZEW + 0)                     // Pad 0 bytes
#define SWIZ32(a,m) ((m) ^ (a))                                // Swizzle 16 rows (remove "^ (a)" to turn swizzling off)
#endif

//      me        ranges 0...WIDTH/NW-1 (multiples of BIG_HEIGHT)
//      u[i]      ranges 0..NW-1 (big multiples of BIG_HEIGHT) 
//      line      ranges 0...BIG_HEIGHT-1 (multiples of one)

// Read a line for carryFused or FFTW.  This line was written by writeMiddleOutLine above.
void OVERLOAD readCarryFusedLine(CP(F2) in, F2 *u, u32 line) {
  u32 me = get_local_id(0);             // Multiples of BIG_HEIGHT
  u32 middle = line / SMALL_HEIGHT;     // Multiples of SMALL_HEIGHT
  line = line % SMALL_HEIGHT;           // Multiples of one
  in += (me / 16 * SIZEW32) + (middle * SIZEM32) + (line % 16 * SIZEBLK32) + SWIZ32(line % 16, line / 16) * 16 + (me % 16);
  for (u32 i = 0; i < NW; ++i) { u[i] = NTLOAD(in[i * G_W / 16 * SIZEW32]); }
}

// Write a line from carryFused.  This data will be read by fftMiddleIn.
void OVERLOAD writeCarryFusedLine(F2 *u, P(F2) out, u32 line) {
  u32 me = get_local_id(0);             // Multiples of BIG_HEIGHT
  u32 middle = line / SMALL_HEIGHT;     // Multiples of SMALL_HEIGHT
  line = line % SMALL_HEIGHT;           // Multiples of one
  out += (me / 16 * SIZEW32) + (middle * SIZEM32) + (line % 16 * SIZEBLK32) + SWIZ32(line % 16, line / 16) * 16 + (me % 16);
  for (i32 i = 0; i < NW; ++i) { NTSTORE(out[i * G_W / 16 * SIZEW32], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from fftMiddleIn
//****************************************************************************************

//      x         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)
//      u[i]      ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...SMALL_HEIGHT-1 (multiples of one)

void OVERLOAD readMiddleInLine(F2 *u, CP(F2) in, u32 y, u32 x) {
  in += (x / 16 * SIZEW32) + (y % 16 * SIZEBLK32) + (SWIZ32(y % 16, y / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SIZEM32]); }
}

// NOTE:  writeMiddleInLine uses the same definition of x,y as readMiddleInLine.  Caller transposes 16x16 blocks of FFT data before calling writeMiddleInLine.
void OVERLOAD writeMiddleInLine (P(F2) out, F2 *u, u32 y, u32 x)
{
  out += (x / 16 * SIZEW32) + (y % 16 * SIZEBLK32) + (SWIZ32(y % 16, y / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * SIZEM32], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from tailSquare/Mul
//****************************************************************************************

//      me        ranges 0...SMALL_HEIGHT/NH-1 (multiples of one)
//      u[i]      ranges 0...NH-1 (big multiples of one)
//      line      ranges 0...MIDDLE*WIDTH-1 (multiples of SMALL_HEIGHT)

// Read a line for tailSquare/Mul or fftHin
void OVERLOAD readTailFusedLine(CP(F2) in, F2 *u, u32 line, u32 me) {
  u32 width = line % WIDTH;            // Multiples of BIG_HEIGHT
  u32 middle = line / WIDTH;           // Multiples of SMALL_HEIGHT
  in += (width / 16 * SIZEW32) + (middle * SIZEM32) + (width % 16 * SIZEBLK32) + (me % 16);
  for (i32 i = 0; i < NH; ++i) { u[i] = NTLOAD(in[SWIZ32(width % 16, (i * SMALL_HEIGHT / NH + me) / 16) * 16]); }
}

void OVERLOAD writeTailFusedLine(F2 *u, P(F2) out, u32 line, u32 me) {
  u32 width = line % WIDTH;            // Multiples of BIG_HEIGHT
  u32 middle = line / WIDTH;           // Multiples of SMALL_HEIGHT
  out += (width / 16 * SIZEW32) + (middle * SIZEM32) + (width % 16 * SIZEBLK32) + (me % 16);
  for (i32 i = 0; i < NH; ++i) { NTSTORE(out[SWIZ32(width % 16, (i * SMALL_HEIGHT / NH + me) / 16) * 16], u[i]); }
}

//****************************************************************************************
// Pair of routines to read/write data to/from fftMiddleOut
//****************************************************************************************

//      x         ranges 0...SMALL_HEIGHT-1 (multiples of one)
//      u[i]      ranges 0...MIDDLE-1 (multiples of SMALL_HEIGHT)
//      y         ranges 0...WIDTH-1 (multiples of BIG_HEIGHT)

void OVERLOAD readMiddleOutLine(F2 *u, CP(F2) in, u32 y, u32 x) {
  in += (y / 16 * SIZEW32) + (y % 16 * SIZEBLK32) + (SWIZ32(y % 16, x / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = NTLOAD(in[i * SIZEM32]); }
}

// NOTE:  writeMiddleOutLine uses the same definition of x,y as readMiddleOutLine.  Caller transposes 16x16 blocks of FFT data before calling writeMiddleOutLine.
void OVERLOAD writeMiddleOutLine (P(F2) out, F2 *u, u32 y, u32 x)
{
  out += (y / 16 * SIZEW32) + (y % 16 * SIZEBLK32) + (SWIZ32(y % 16, x / 16) * 16) + (x % 16);
  for (i32 i = 0; i < MIDDLE; ++i) { NTSTORE(out[i * SIZEM32], u[i]); }
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

// Since F2 and GF31 are the same size we can simply call the floats based code

void OVERLOAD readCarryFusedLine(CP(GF31) in, GF31 *u, u32 line) {
  readCarryFusedLine((CP(F2)) in, (F2 *) u, line);
}

void OVERLOAD writeCarryFusedLine(GF31 *u, P(GF31) out, u32 line) {
  writeCarryFusedLine((F2 *) u, (P(F2)) out, line);
}

void OVERLOAD readMiddleInLine(GF31 *u, CP(GF31) in, u32 y, u32 x) {
  readMiddleInLine((F2 *) u, (CP(F2)) in, y, x);
}

void OVERLOAD writeMiddleInLine (P(GF31) out, GF31 *u, u32 y, u32 x) {
  writeMiddleInLine ((P(F2)) out, (F2 *) u, y, x);
}

void OVERLOAD readTailFusedLine(CP(GF31) in, GF31 *u, u32 line, u32 me) {
  readTailFusedLine((CP(F2)) in, (F2 *) u, line, me);
}

void OVERLOAD writeTailFusedLine(GF31 *u, P(GF31) out, u32 line, u32 me) {
  writeTailFusedLine((F2 *) u, (P(F2)) out, line, me);
}

void OVERLOAD readMiddleOutLine(GF31 *u, CP(GF31) in, u32 y, u32 x) {
  readMiddleOutLine((F2 *) u, (CP(F2)) in, y, x);
}

void OVERLOAD writeMiddleOutLine (P(GF31) out, GF31 *u, u32 y, u32 x) {
  writeMiddleOutLine ((P(F2)) out, (F2 *) u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

// Since T2 and GF61 are the same size we can simply call the doubles based code

void OVERLOAD readCarryFusedLine(CP(GF61) in, GF61 *u, u32 line) {
  readCarryFusedLine((CP(T2)) in, (T2 *) u, line);
}

void OVERLOAD writeCarryFusedLine(GF61 *u, P(GF61) out, u32 line) {
  writeCarryFusedLine((T2 *) u, (P(T2)) out, line);
}

void OVERLOAD readMiddleInLine(GF61 *u, CP(GF61) in, u32 y, u32 x) {
  readMiddleInLine((T2 *) u, (CP(T2)) in, y, x);
}

void OVERLOAD writeMiddleInLine (P(GF61) out, GF61 *u, u32 y, u32 x) {
  writeMiddleInLine ((P(T2)) out, (T2 *) u, y, x);
}

void OVERLOAD readTailFusedLine(CP(GF61) in, GF61 *u, u32 line, u32 me) {
  readTailFusedLine((CP(T2)) in, (T2 *) u, line, me);
}

void OVERLOAD writeTailFusedLine(GF61 *u, P(GF61) out, u32 line, u32 me) {
  writeTailFusedLine((T2 *) u, (P(T2)) out, line, me);
}

void OVERLOAD readMiddleOutLine(GF61 *u, CP(GF61) in, u32 y, u32 x) {
  readMiddleOutLine((T2 *) u, (CP(T2)) in, y, x);
}

void OVERLOAD writeMiddleOutLine (P(GF61) out, GF61 *u, u32 y, u32 x) {
  writeMiddleOutLine ((P(T2)) out, (T2 *) u, y, x);
}

#endif

#endif
