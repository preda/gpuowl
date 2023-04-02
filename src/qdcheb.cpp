/*******************************************************************************
*                                                                              *
*   (C) 1997-2020 by Ernst W. Mayer.                                           *
*                                                                              *
*  This program is free software; you can redistribute it and/or modify it     *
*  under the terms of the GNU General Public License as published by the       *
*  Free Software Foundation; either version 2 of the License, or (at your      *
*  option) any later version.                                                  *
*                                                                              *
*  This program is distributed in the hope that it will be useful, but WITHOUT *
*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
*  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for   *
*  more details.                                                               *
*                                                                              *
*  You should have received a copy of the GNU General Public License along     *
*  with this program; see the file GPL.txt.  If not, you may view one at       *
*  http://www.fsf.org/licenses/licenses.html, or obtain one by writing to the  *
*  Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA     *
*  02111-1307, USA.                                                            *
*                                                                              *
*******************************************************************************/

// Modified by George Woltman
//
// To compile (on Mac):   g++ qdcheb.cpp -lqd

#define x86
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "qd/qd_real.h"		// David Bailey's quad-double library (the latest version available on 4/15/2020)

#define STR_MAX_LEN 1024
char cbuf[STR_MAX_LEN];
typedef long int64;

long as_long(double x) {
	union {
		double d;
		long l;
	} foo;
	foo.d = x;
	return foo.l;
}

// Generate Chebyshev approximation to f(x) = sin(x) on x in [-1,+1]
int 	main(int argc, char *argv[])
{
	int nn = 15;

//	ASSERT(HERE, IS_EVEN(n), "For silly reasons, n must be even!");
	printf("Using n = %d\n", nn);
	nn += 1;
	
	double maxerr, derr, xmaxerr, maxrelerr, xmaxrelerr, sumerr, sumrelerr;
	double tf, x, x2;
	int i,j,k,n;
	// c = Cheb-coeffs, d = corresponding raw-monomial coeffs, t = Tn(x):
	qd_real qfunc,qfuncscale,qerr,qn,qninv,qpin,qx,q2x,qy,qscale,qscinv,qt,qk,qtmp, *c = 0x0, *d = 0x0, *t = 0x0;
	int64 *xpow_mults = 0x0; xpow_mults = (int64 *)malloc(nn*nn*sizeof(int64));	// Coeffs of various powers of x in the Tn
	// Alloc basis and coeffs (Cheb-basis and raw-monomials) arrays, add a pad element to each to allow for 1st neglected basis function:
	c = (qd_real *)malloc((3*nn+2)*sizeof(qd_real)); d = c + nn + 1; t = d + nn + 1;

double SCALER;
// This for loop was my first investigation into different multipliers for better sine/cosine accuracy
// Final choice was to use SCALER = 4.0, and investigate multipliers in sine_compare.cpp.
for (int qqqq = -5; qqqq <= 100; qqqq++) {		
if (qqqq == -5) continue;//SCALER = 1.0;
else if (qqqq == -4) SCALER = 4.0;
else if (qqqq == -3) break;//SCALER = 510.0 / 128.0;
else if (qqqq == -2) SCALER = 505.0 / 128.0;
else if (qqqq == -1) SCALER = 500.0 / 128.0;
else SCALER = (2147475330.0 - (qqqq * 45045.0)) * pow (2.0, -29.0);   // 2147475330 - (i*45045)  * 2^22 / 2^51
printf("Using scale = %25.17g, %lX\n", SCALER, as_long(SCALER));

	// Scale factor for mapping target-function half-interval y in [0,ymax] to Chebyshev half-interval x in [0,+1]:
if (SCALER == 1.0) {
	qscale = qd_real::_pi4;
	qfuncscale = 1.0;
} else {
	qscale = 1.0;
	qfuncscale = qd_real::_pi / SCALER;
}

	/*****************************************************************/
	/***************** Cos(y) for y in [-Pi/4,Pi/4]: *****************/
	/*****************************************************************/
	n = nn & ~1;
	qn = n;
	qpin = qd_real::_pi / qn;	// Pi/n
	qninv = 4.0 / qn;		// 4/n
	qscinv = 1.0 / qscale;
	// cos(x) an even function so only need even-order Tj. Using same symmetry we only need the half-interval x in [0,+1]:
	printf("Computing Chebyshev-approximation to Cos(y) for |y| <= %18.16f, using first %d T_j(x):\n", to_double(qscale), n);
	t[0] = 1.0;	// The DC term
	for(j = 0; j < n; j++) { c[j] = 0.0; }
	for(i = 0, qk = 0.5; i < (n>>1); i++) {
		qx = qk * qpin;		// theta = (k + 1/2)*Pi/n
		qx = cos(qx);		// x = cos(theta)
		qy = qx * qscale;
//		qfunc = qfcos(qy);	// f(x) = cos(y(x)) = cos(x*scale)
qfunc = cos(qy * qfuncscale);		// f(x) = cos(y(x*pi/SCALER))
		q2x = 2.0 * qx;		// 2x
	//	fprintf(stderr,"x[%2d] = %20.15f, cos(x) = %20.15f\n",i,qfdbl(qx),qfdbl(qfunc));
		c[0] = c[0] + qfunc * t[0];	// cj += f(x)*Tj(x)
		t[1] = qx;		// T1 done separately to complete init of 3-term recurrence
		for(j = 2; j < n; j+=2) {
			t[j] = q2x * t[j-1] - t[j-2];	// Next odd  term: T[j] = 2.x.T[j-1] - T[j-2]
			t[j+1] = q2x * t[j] - t[j-1];	// Next even term: T[j+1] = 2.x.T[j] - T[j-1]
			c[j] = c[j] + qfunc * t[j];	// cj += f(x)*Tj(x)
		}
		qk = qk + 1.0;		// k += 1
	}
//	fprintf(stderr,"Chebyshev basis function coefficients, as double and [exact uint64 bitfield]:\n");
	for(j = 0; j < n; j+=2) {
		c[j] = qninv * c[j];
		if (!j) c[0] = c[0] * 0.5;	// DC term normalizer is 1/n, half of the 2/n used for the remaining coefficients:
//		fprintf(stderr,"c[%2d] = %25.15e [0x%16llx]\n",j,qfdbl(c[j]),qfdbl_as_uint64(c[j]));	// (4/n)*[half-interval sum]
	}

	// Convert Cheb-coeffs to ones of underlying raw monomials. First init the x-powers coefficients array,
	// treated as a 2D array with each T-poly having n entries, even though only the highest-term one, Tn-1 needs that many:
	for(i = 0; i < n; i++) { d[i] = 0.0; }
	for(i = 0; i < n*n; i++) { xpow_mults[i] = 0ull; }	// init = 0
	xpow_mults[  0] = 1ull;	// T0 = 1 = 1.x^0
	xpow_mults[n+1] = 1ull;	// T1 = x = 0.x^0 + 1.x^1
	for(i = 2; i < n; i++) {
		j = i*n;	// Ti's coeffs start at array elt i*n
		for(k = j; k < (j+i); k++) { xpow_mults[k+1]  = 2*xpow_mults[k-n  ]; } 	// T[j] = 2.x.T[j-1] ...
		for(k = j; k < (j+i); k++) { xpow_mults[k  ] -=   xpow_mults[k-2*n]; } 	// T[j] = 2.x.T[j-1] - T[j-2]
		// Sanity-check each T's coeffs that they sum to 1:
		int64 csum = 0ull;
		//fprintf(stderr,"Coeffs of T[%2d] = ",i);
		for(k = j; k < (j+i+1); k++) {
			csum += xpow_mults[k];
		//	fprintf(stderr,"%lld,",xpow_mults[k]);
		}// fprintf(stderr,"\n");
		//assert(csum == 1ull, "Chebyshev basis function coefficients fail to sum = 1!");
	}
	// Now sum the weighted expansion coefficients to get the resulting raw-monomial coefficients:
	for(i = 0; i < n; i++) { d[i] = 0.0; }
	for(i = 0; i < n; i++) {	// This loop still over Cheb-basis index!
		j = i*n;	// Ti's coeffs start at array elt i*n
		// The coeffs of the various monomials terms the current (i)th basis function get weighted-added to the respective monomial-coeff accumulators:
		qt = 1.0;	// Need to multiply each power of x by 1/(scale factor) raised to same power
		for(k = j; k < (j+i+1); k++) {
		//	fprintf(stderr,"d[%2d] += %lld * %25.15e:\n",k-j,xpow_mults[k],qfdbl(c[i]));
			d[k-j] = d[k-j] + qt * double(xpow_mults[k]) * c[i];
			qt = qt * qscinv;	// up-multiply inverse coordinate scaling in preparation for next loop pass
		}
	}
	printf("Raw polynomial coefficients, as double and [exact uint64 bitfield]:\n");
	for(j = 0; j < n; j++) {
		char buf[80];
		d[j].write(buf, 72);
		qd_real tmp;
		tmp = d[j] * pow(SCALER, j);
		tmp.write(buf, 72);
		if(d[j] != 0.0) printf("d[%2d] = %s, rel.err: %g\n",
					       j, buf, //to_double(d[j]), as_long(to_double(d[j])),
//					       qfdbl(qfdiv(qfsub(d[j],dbl_to_q(qfdbl(d[j]))),d[j])));
//					       qfdbl(qfsub(d[j],dbl_to_q(qfdbl(d[j])))));
					       fabs(to_double((d[j] - to_double(d[j])) / d[j])));
	}

	// Compute maxerr over [0,1], incrementing by 10^6:
	maxerr = 0, sumerr = 0;;
	maxrelerr = 0, sumrelerr = 0;
	for(i = 0; i < 1000001; i++) {
		x = (double) i / 1000000.0;
qfunc = cos(x * qfuncscale);	// f(x) = cos(x*pi/SCALER)
		tf = to_double(d[n-2]);
		x2 = x * x;
		for(j = n - 4; ; j -= 2) {
			tf = fma(x2, tf, to_double(d[j]));
			if (j == 0) break;
		}
		derr = fabs(to_double(qfunc - tf)); sumerr += derr;
		if(derr > maxerr) { maxerr = derr; xmaxerr = x; }
		derr = fabs(to_double(derr / qfunc)); sumrelerr += derr;
		if(derr > maxrelerr) { maxrelerr = derr; xmaxrelerr = x; }
	}
	printf("avgerr = %25.15e, maxerr = %25.15e at x = %20.15f\n",sumerr/1000001.0,maxerr,xmaxerr);
	printf("avgrelerr = %25.15e, maxrelerr = %25.15e at x = %20.15f\n",sumrelerr/1000001.0,maxrelerr,xmaxrelerr);

	/*****************************************************************/
	/***************** Sin(y) for y in [-Pi/4,Pi/4]: *****************/
	/*****************************************************************/
	n = (nn - 1) & ~1;
	qn = n;
	qpin = qd_real::_pi / qn;	// Pi/n
	qninv = 4.0 / qn;		// 4/n
	// sin(x) an odd function so only need odd-order Tj. Using same symmetry we only need the half-interval x in [0,+1]:
	printf("Computing Chebyshev-approximation to Sin(y) for |y| <= %18.16f, using first %d T_j(x):\n",to_double(qscale),n);
	t[0] = 1.0;	// The DC term - with a few further optimizations we wouldn't need this for the Sin(x) half of the run
	for(j = 0; j < n; j++) { c[j] = 0.0; }
	for(i = 0, qk = 0.5; i < (n>>1); i++) {
		qx = qk * qpin;		// theta = (k + 1/2)*Pi/n
		qx = sin(qx);		// x = sin(theta)
		qy = qx * qscale;
//		qfunc = qfsin(qy);	// f(x) = sin(y(x)) = sin(x*scale)
qfunc = sin(qy * qfuncscale);		// f(x) = sin(y(x*pi/SCALER))
		q2x = 2.0 * qx;		// 2x
	//	fprintf(stderr,"x[%2d] = %20.15f, sin(x) = %20.15f\n",i,qfdbl(qx),qfdbl(qfunc));
		t[1] = qx;		// T1 done separately to complete init of 3-term recurrence
		for(j = 1; j < n; j+=2) {
			if(j > 1) t[j] = q2x * t[j-1] - t[j-2];	// Next odd  term: T[j] = 2.x.T[j-1] - T[j-2]
			t[j+1] = q2x * t[j] - t[j-1];		// Next even term: T[j+1] = 2.x.T[j] - T[j-1]
			c[j] = c[j] + qfunc * t[j];		// cj += f(x)*Tj(x)
		}
		qk = qk + 1.0;		// k += 1
	}
//	fprintf(stderr,"Chebyshev basis function coefficients, as double and [exact uint64 bitfield]:\n");
	for(j = 1; j < n; j+=2) {
		c[j] = qninv * c[j];
//		fprintf(stderr,"c[%2d] = %25.15e [0x%16llx]\n",j,qfdbl(c[j]),qfdbl_as_uint64(c[j]));	// (4/n)*[half-interval sum]
	}

	// Convert Cheb-coeffs to ones of underlying raw monomials. First init the x-powers coefficients array,
	// treated as a 2D array with each T-poly having n entries, even though only the highest-term one, Tn-1 needs that many:
	for(i = 0; i < n; i++) { d[i] = 0.0; }
	for(i = 0; i < n*n; i++) { xpow_mults[i] = 0ull; }	// init = 0
	xpow_mults[  0] = 1ull;	// T0 = 1 = 1.x^0
	xpow_mults[n+1] = 1ull;	// T1 = x = 0.x^0 + 1.x^1
	for(i = 2; i < n; i++) {
		j = i*n;	// Ti's coeffs start at array elt i*n
		for(k = j; k < (j+i); k++) { xpow_mults[k+1]  = 2*xpow_mults[k-n  ]; } 	// T[j] = 2.x.T[j-1] ...
		for(k = j; k < (j+i); k++) { xpow_mults[k  ] -=   xpow_mults[k-2*n]; } 	// T[j] = 2.x.T[j-1] - T[j-2]
		// Sanity-check each T's coeffs that they sum to 1:
		int64 csum = 0ull;
		//fprintf(stderr,"Coeffs of T[%2d] = ",i);
		for(k = j; k < (j+i+1); k++) {
			csum += xpow_mults[k];
		//	fprintf(stderr,"%lld,",xpow_mults[k]);
		}// fprintf(stderr,"\n");
		//assert(csum == 1ull, "Chebyshev basis function coefficients fail to sum = 1!");
	}
	// Now sum the weighted expansion coefficients to get the resulting raw-monomial coefficients:
	for(i = 0; i < n; i++) { d[i] = 0.0; }
	for(i = 0; i < n; i++) {	// This loop still over Cheb-basis index!
		j = i*n;	// Ti's coeffs start at array elt i*n
		// The coeffs of the various monomials terms the current (i)th basis function get weighted-added to the respective monomial-coeff accumulators:
		qt = 1.0;	// Need to multiply each power of x by 1/(scale factor) raised to same power
		for(k = j; k < (j+i+1); k++) {
		//	fprintf(stderr,"d[%2d] += %lld * %25.15e:\n",k-j,xpow_mults[k],qfdbl(c[i]));
			d[k-j] = d[k-j] + qt * double(xpow_mults[k]) * c[i];
			qt = qt * qscinv;	// up-multiply inverse coordinate scaling in preparation for next loop pass
		}
	}
	printf("Raw polynomial coefficients, as double and [exact uint64 bitfield]:\n");
	for(j = 0; j < n; j++) {
		char buf[80];
		d[j].write(buf, 72);
		qd_real tmp;
		tmp = d[j] * pow(SCALER, j);
		tmp.write(buf, 72);
		if(d[j] != 0.0) printf("d[%2d] = %s, rel.err: %g\n",
					j, buf, //to_double(d[j]), as_long(to_double(d[j])),
//				       qfdbl(qfdiv(qfsub(d[j],dbl_to_q(qfdbl(d[j]))),d[j])));
//				       qfdbl(qfsub(d[j],dbl_to_q(qfdbl(d[j])))));
				       fabs(to_double((d[j] - to_double(d[j])) / d[j])));
	}

	// Compute maxerr over [0,1], incrementing by 10^6:
	maxerr = 0, sumerr = 0;;
	maxrelerr = 0, sumrelerr = 0;
	for(i = 0; i < 1000001; i++) {
		x = (double) i / 1000000.0;
qfunc = sin(x * qfuncscale);	// f(x) = sin(x*pi/SCALER)
		tf = to_double(d[n-1]);
		x2 = x * x;
		for(j = n - 3; ; j -= 2) {
			tf = fma(x2, tf, to_double(d[j]));
			if (j == 3) break;
		}
		tf *= (x2 * x);
		tf = fma (x, to_double(d[1]), tf);
		derr = fabs(to_double(qfunc - tf)); sumerr += derr;
		if(derr > maxerr) { maxerr = derr; xmaxerr = x; }
		if (i) derr = fabs(to_double(derr / qfunc)); sumrelerr += derr;
		if(derr > maxrelerr) { maxrelerr = derr; xmaxrelerr = x; }
	}
	printf("avgerr = %25.15e, maxerr = %25.15e at x = %20.15f\n",sumerr/1.0e6,maxerr,xmaxerr);
	printf("avgrelerr = %25.15e, maxrelerr = %25.15e at x = %20.15f\n",sumrelerr/1.0e6,maxrelerr,xmaxrelerr);
}
	return 0;
}




//instead of 2.5 or 4.0   pick a number, w, less than 4 that is divisible by 3,5,7,3,11,13
//   and will not loose any bit when we calc k * w / n 
//
// Two strategies: Try to keep x accurate and find really accurate polynomial coefficients OR
// keep x^2 accurate.
// Max n is FFT length / 4
// max k is n / 4
//  k * funkyval / n
//
//  5M FFT:  kmax = 19 bits
//  funkyval/5 can be 1/2 * (53 - 19*2 bits) = 7 bits
//  funkyval = 9 bits?  choose 8 for safety
//  3.999 as 8 bits data / 45 bits zero.  (4*2^6 - something to get divisible by 5)*2^45 =  3.984375
//  Try 515/128 = 4.0234375
//

// Try 22 low bits of zero bits (3 = 2 bits)(29 bit to work with)(22 zero bits)
// Make it a multiple of 5*7*9*11*13 = 45045
// 2147475330 - (i*45045)  * 2^22 / 2^51
