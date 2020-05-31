/*
** Copyright (c) 2017 D. Richard Hipp
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the Simplified BSD License (also
** known as the "2-Clause License" or "FreeBSD License".)
**
** This program is distributed in the hope that it will be useful,
** but without any warranty; without even the implied warranty of
** merchantability or fitness for a particular purpose.
**
** Author contact information:
**   drh@hwaci.com
**   http://www.hwaci.com/drh/
**
*******************************************************************************
**
** This file contains an implementation of SHA3 (Keccak) hashing.
*/

#pragma once

#include "common.h"

struct SHA3Context {
  union {
    u64 s[25];                /* Keccak state. 5x5 lines of 64 bits each */
    unsigned char x[1600];    /* ... or 1600 bytes */
  } u;
  unsigned nRate;        /* Bytes of input accepted per Keccak iteration */
  unsigned nLoaded;      /* Input bytes loaded into u.x[] so far this cycle */
  unsigned ixMask;       /* Insert next input into u.x[nLoaded^ixMask]. */
};

/*
** Initialize a new hash.  iSize determines the size of the hash
** in bits and should be one of 224, 256, 384, or 512.  Or iSize
** can be zero to use the default hash size of 256 bits.
*/
void SHA3Init(SHA3Context *p, int iSize);

/*
** Make consecutive calls to the SHA3Update function to add new content
** to the hash
*/
void SHA3Update(SHA3Context *p, const unsigned char *aData, unsigned int nData);

/*
** After all content has been added, invoke SHA3Final() to compute
** the final hash.  The function returns a pointer to the binary
** hash value.
*/
unsigned char *SHA3Final(SHA3Context *p);
