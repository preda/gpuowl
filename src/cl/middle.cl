// Copyright (C) Mihai Preda and George Woltman

#if !IN_WG
#define IN_WG 256
#endif

#if !OUT_WG
#define OUT_WG 256
#endif

#if !IN_SIZEX
#if AMDGPU
#define IN_SIZEX 32
#else // !AMDGPU
#if G_W >= 64
#define IN_SIZEX 4
#else
#define IN_SIZEX 32
#endif
#endif
#endif

#if !OUT_SIZEX
#if AMDGPU
// We realized that these (OUT_WG, OUT_SIZEX) combinations work well: (256, 32) and (64, 8)
// so default OUT_SIZEX relative to OUT_WG
#define OUT_SIZEX (OUT_WG / 8)
#else
#if G_W >= 64
#define OUT_SIZEX 4
#else
#define OUT_SIZEX 32
#endif
#endif
#endif

// Parameters we may want to let user tune.  WIDTH other than 512 and 1K is untested.  SMALL_HEIGHT other than 256 and 512 is untested.
#if AMDGPU
#define PADDING 1                                       // Prefer padding to avoid bad strides
#define MIDDLE_IN_LDS_TRANSPOSE (IN_WG >= 128)          // Radeon VII likes LDS transpose for larger workgroups
#define MIDDLE_OUT_LDS_TRANSPOSE (OUT_WG >= 128)        // Radeon VII likes LDS transpose for larger workgroups
#define PAD_SIZE 16                                     // Radeon VII likes 16 T2 values = 256 bytes 
#endif

// nVidia Titan V see no padding benefit, likes LDS transposes
#if !AMDGPU
#define PADDING 0                                       // Don't prefer padding to avoid bad strides
#define MIDDLE_IN_LDS_TRANSPOSE 1                       // nVidia likes LDS transpose
#define MIDDLE_OUT_LDS_TRANSPOSE 1                      // nVidia likes LDS transpose
#define PAD_SIZE 8                                      // nVidia documentation indicates 8 T2 values = 128 bytes should be best
#endif

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

void writeCarryFusedLine(T2 *u, P(T2) out, u32 line) {
#if PADDING
  out += line * WIDTH + line / SMALL_HEIGHT * PAD_SIZE + (u32) get_local_id(0); // One padding every SMALL_HEIGHT lines
  for (u32 i = 0; i < NW; ++i) { out[i * G_W] = u[i]; }
#else
  out += line * WIDTH + (u32) get_local_id(0);
  for (u32 i = 0; i < NW; ++i) { out[i * G_W] = u[i]; }
#endif
}

void readMiddleInLine(T2 *u, CP(T2) in, u32 y, u32 x) {
#if PADDING
  in += y * WIDTH + y / SMALL_HEIGHT * PAD_SIZE + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * (SMALL_HEIGHT * WIDTH + PAD_SIZE)]; }
#else
  in += y * WIDTH + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT * WIDTH]; }
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

void writeMiddleInLine (P(T2) out, T2 *u, u32 chunk_y, u32 chunk_x)
{
  //u32 SIZEY = IN_WG / IN_SIZEX;
  //u32 num_x_chunks = WIDTH / IN_SIZEX;                // Number of x chunks
  //u32 num_y_chunks = SMALL_HEIGHT / SIZEY;            // Number of y chunks

#if PADDING

  u32 SIZEY = IN_WG / IN_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * IN_WG + PAD_SIZE) +        // Write y chunks after middle chunks and a pad 
         chunk_x * (SMALL_HEIGHT * MIDDLE * IN_SIZEX +  // num_y_chunks * (MIDDLE * IN_WG + PAD_SIZE)
                    SMALL_HEIGHT / SIZEY * PAD_SIZE + BIG_PAD_SIZE);
                                                        //              = SMALL_HEIGHT / SIZEY * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT / (IN_WG / IN_SIZEX) * (MIDDLE * IN_WG + PAD_SIZE)
                                                        //              = SMALL_HEIGHT * MIDDLE * IN_SIZEX + SMALL_HEIGHT / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { out[i * IN_WG] = u[i]; }

#else

  // Output data such that readCarryFused lines are packed tightly together.  No rotations or padding.
  out += chunk_y * MIDDLE * IN_WG +                     // Write y chunks after middles
         chunk_x * MIDDLE * SMALL_HEIGHT * IN_SIZEX;    // num_y_chunks * IN_WG = SMALL_HEIGHT / SIZEY * MIDDLE * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT / (IN_WG / IN_SIZEX) * IN_WG
                                                        //                       = MIDDLE * SMALL_HEIGHT * IN_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { out[i * IN_WG] = u[i]; }

#endif
}

// Read a line for tailFused or fftHin
// This reads partially transposed data as written by fftMiddleIn
void readTailFusedLine(CP(T2) in, T2 *u, u32 line, u32 me) {
  u32 SIZEY = IN_WG / IN_SIZEX;

#if PADDING

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

  // Adjust in pointer based on the y value used in writeMiddleInLine
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  for (i32 i = 0; i < NH; ++i) {
    u32 fftMiddleIn_y = i * G_H + me;                           // The fftMiddleIn y value
    u32 chunk_y = fftMiddleIn_y / SIZEY;                        // The fftMiddleIn chunk_y value
    u[i] = in[chunk_y * (MIDDLE * IN_WG + PAD_SIZE)];           // Adjust in pointer the same way writeMiddleInLine did
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

  // Adjust in pointer based on the y value used in writeMiddleInLine
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  for (i32 i = 0; i < NH; ++i) {
    u32 fftMiddleIn_y = i * G_H + me;                           // The fftMiddleIn y value
    u32 chunk_y = fftMiddleIn_y / SIZEY;                        // The fftMiddleIn chunk_y value
    u[i] = in[chunk_y * (MIDDLE * IN_WG)];                      // Adjust in pointer the same way writeMiddleInLine did
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

void writeTailFusedLine(T2 *u, P(T2) out, u32 line, u32 me) {
#if PADDING
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  out += line * (SMALL_HEIGHT + PAD_SIZE) + line / MIDDLE * BIG_PAD_SIZE + me; // Pad every output line plus every MIDDLE
#else
  out += line * (SMALL_HEIGHT + PAD_SIZE) + me;                         // Pad every output line
#endif
  for (u32 i = 0; i < NH; ++i) { out[i * G_H] = u[i]; }
#else                                                                   // No padding, might be better on nVidia cards
  out += line * SMALL_HEIGHT + me;
  for (u32 i = 0; i < NH; ++i) { out[i * G_H] = u[i]; }
#endif
}

void readMiddleOutLine(T2 *u, CP(T2) in, u32 y, u32 x) {
#if PADDING
#if MIDDLE == 4 || MIDDLE == 8 || MIDDLE == 16
  // Each u[i] increments by one pad size.
  // Rather than each work group reading successive y's also increment by one, we choose a larger pad increment.
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + y * BIG_PAD_SIZE + x;
#else
  in += y * MIDDLE * (SMALL_HEIGHT + PAD_SIZE) + x;
#endif
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * (SMALL_HEIGHT + PAD_SIZE)]; }
#else                                                                   // No rotation, might be better on nVidia cards
  in += y * MIDDLE * SMALL_HEIGHT + x;
  for (i32 i = 0; i < MIDDLE; ++i) { u[i] = in[i * SMALL_HEIGHT]; }
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
// limits the number of rotation/padding options in a 2KB chunk to 16, 8, or just 4.  If we go with option (b) or (c) we can rotate data over a larger
// area, but that is of no benefit as the stride will still be some multiple of 2KB.  If there are other bad stride values, (e.g. some CPUs don't like
// 64KB strides) that could impact our decision here (e.g. option (c) with MIDDLE=16 would result in a 32KB stride).
//
// If we output all MIDDLE i values after the x and y values, there will be a huge power-of-two stride between these writes.
// This is a problem on Radeon VII.  Another rotation or padding would be necessary.
//
// After experimentation, we've chosen to output the MIDDLE values next with padding (padding is simpler code than rotation).
// Other options are workable with no measurable degradation in performance.

// Caller must either give us u values that are grouped by x values (i.e. the order in which they were read in) with the out pointer
// adjusted to effect a transpose.  Or caller must transpose the x and y values and send us an out pointer with thread_id added in.
// In other words, caller is responsible for deciding the best way to transpose x and y values.

void writeMiddleOutLine (P(T2) out, T2 *u, u32 chunk_y, u32 chunk_x)
{
  //u32 SIZEY = OUT_WG / OUT_SIZEX;
  //u32 num_x_chunks = SMALL_HEIGHT / OUT_SIZEX;  // Number of x chunks
  //u32 num_y_chunks = WIDTH / SIZEY;             // Number of y chunks

#if PADDING

  u32 SIZEY = OUT_WG / OUT_SIZEX;
  u32 BIG_PAD_SIZE = (PAD_SIZE/2+1)*PAD_SIZE;

  out += chunk_y * (MIDDLE * OUT_WG + PAD_SIZE) +       // Write y chunks after middle chunks and a pad 
         chunk_x * (WIDTH * MIDDLE * OUT_SIZEX +        // num_y_chunks * (MIDDLE * OUT_WG + PAD_SIZE)
                    WIDTH / SIZEY * PAD_SIZE + BIG_PAD_SIZE);//         = WIDTH / SIZEY * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH / (OUT_WG / OUT_SIZEX) * (MIDDLE * OUT_WG + PAD_SIZE)
                                                        //              = WIDTH * MIDDLE * OUT_SIZEX + WIDTH / SIZEY * PAD_SIZE
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { out[i * OUT_WG] = u[i]; }

#else

  // Output data such that readCarryFused lines are packed tightly together.  No rotations or padding.
  out += chunk_y * MIDDLE * OUT_WG +             // Write y chunks after middles
         chunk_x * MIDDLE * WIDTH * OUT_SIZEX;   // num_y_chunks * OUT_WG = WIDTH / SIZEY * MIDDLE * OUT_WG
                                        //                       = MIDDLE * WIDTH / (OUT_WG / OUT_SIZEX) * OUT_WG
                                        //                       = MIDDLE * WIDTH * OUT_SIZEX
  // Write each u[i] sequentially
  for (int i = 0; i < MIDDLE; ++i) { out[i * OUT_WG] = u[i]; }

#endif
}

// Read a line for carryFused or FFTW.  This line was written by writeMiddleOutLine above.
void readCarryFusedLine(CP(T2) in, T2 *u, u32 line) {
  u32 me = get_local_id(0);
  u32 SIZEY = OUT_WG / OUT_SIZEX;

#if PADDING

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

  // Adjust in pointer based on the y value used in writeMiddleOutLine
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  for (i32 i = 0; i < NW; ++i) {
    u32 fftMiddleOut_y = i * G_W + me;                          // The fftMiddleOut y value
    u32 chunk_y = fftMiddleOut_y / SIZEY;                       // The fftMiddleOut chunk_y value
    u[i] = in[chunk_y * (MIDDLE * OUT_WG + PAD_SIZE)];          // Adjust in pointer the same way writeMiddleOutLine did
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

  // Adjust in pointer based on the y value used in writeMiddleOutLine
  in += me % SIZEY;                                             // Adjust in pointer to read SIZEY consecutive values
  for (i32 i = 0; i < NW; ++i) {
    u32 fftMiddleOut_y = i * G_W + me;                          // The fftMiddleOut y value
    u32 chunk_y = fftMiddleOut_y / SIZEY;                       // The fftMiddleOut chunk_y value
    u[i] = in[chunk_y * MIDDLE * OUT_WG];                       // Adjust in pointer the same way writeMiddleOutLine did
  }

#endif

}
