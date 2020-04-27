
// This program is a mess that evolved from my search for sin and cosine routines with the greatest accuracy
// Several implementations were compared during the search.
//
// Compile (on Mac) with:
// g++ sine_compare.cpp -lqd

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "qd/qd_real.h"		// David Bailey's quad-double library (the latest version available on 4/15/2020)

double ksin_fast(double k, double n) {
  double
  S0 = +3.14159265358979322857549585649252479170596805578751964804604910e+00 * 1.0 / (n),
  S1 = -5.16771278004995452583938866754421448491805106851652909471625339e+00 * 1.0 / (n * n * n),
  S2 = +2.55016403987337610028335512515697142485381483455575768286101741e+00 * 1.0 / (n * n * n * n * n),
  S3 = -5.99264528939644926932658650083463907769354154612435064156617923e-01 * 1.0 / (n * n * n * n * n * n * n),
  S4 = +8.21458691800017461862925981661435393176563606567669316878047234e-02 * 1.0 / (n * n * n * n * n * n * n * n * n),
  S5 = -7.37002158690777026975091957439476212577226315422626427540194454e-03 * 1.0 / (n * n * n * n * n * n * n * n * n * n * n),
  S6 = +4.61531855383580995897596361983564763395850652294559984699291272e-04 * 1.0 / (n * n * n * n * n * n * n * n * n * n * n * n * n);
  double x = k;
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S6, z, S5), z, S4), z, S3), z, S2), z, S1) * (z * x);
  return fma(S0, x, r);
}

double ksin_sun(double x, double z) {
  // Coefficients from http://www.netlib.org/fdlibm/k_sin.c
  // Excellent accuracy in [-pi/4, pi/4]
  const double
  S1 = -0x1.5555555555555p-3,  // -1.66666666666666657415e-01 bfc55555'55555555
  S2 = +0x1.1111111110bb3p-7,  // +8.33333333333094970763e-03 3f811111'11110bb3
  S3 = -0x1.a01a019e83e5cp-13, // -1.98412698367611268872e-04 bf2a01a0'19e83e5c
  S4 = +0x1.71de3796cde01p-19, // +2.75573161037288024585e-06 3ec71de3'796cde01
  S5 = -0x1.ae600b42fdfa7p-26, // -2.50511320680216983368e-08 be5ae600'b42fdfa7
  S6 = +0x1.5e0b2f9a43bb8p-33; // +1.59181443044859141215e-10 3de5e0b2'f9a43bb8
  return fma(fma(fma(fma(fma(fma(S6, z, S5), z, S4), z, S3), z, S2), z, S1), z * x, x);
}

double ksin_sun(double x) {
  return ksin_sun(x, x * x);
}

double ksin_ernst(double x, double z, double zx) {
  const double
//S0 = +9.99999999999999996852823196066089573553472194594869813367297147e-01,
  S1 = -1.66666666666666166658012815819114046398869185649590309704946992e-01,
  S2 = +8.33333333332036245671927647514525997763761453303632578016558526e-03,
  S3 = -1.98412698286503000130594117523892524389174787859927861141262169e-04,
  S4 = +2.75573133764001291282060431837359178951753870699465057996709189e-06,
  S5 = -2.50507169741027450278362856839525483869413194490361160645917139e-08,
  S6 = +1.58947366518490952593077204883483730419165920115794630432275505e-10;

  return fma(fma(fma(fma(fma(fma(S6, z, S5), z, S4), z, S3), z, S2), z, S1), zx, x);
}

double ksin_ernst(double x, double z) {
  return ksin_ernst(x, z, z * x);
}

double ksin_ernst(double x) {
  return ksin_ernst(x, x * x);
}

// adapted from https://stackoverflow.com/questions/42792939/
// CAUTION: this function only works for arguments in the range [-0.25; 0.25]!
double ksinpi(int k, int n) {
  double a = k;
  a = a / n;
  double s = a * a;
//  double s = double(k) * double(k);
//  s = s / (double(n) * double(n));			// Again this method of computing s is worse!
  /* Approximate sin(pi*x) for x in [-0.25,0.25] */
  double r =      4.6151442520157035e-4;
  r = fma (r, s, -7.3700183130883555e-3);
  r = fma (r, s,  8.2145868949323936e-2);
  r = fma (r, s, -5.9926452893214921e-1);
  r = fma (r, s,  2.5501640398732688e+0);
  r = fma (r, s, -5.1677127800499516e+0);
  s = s * a;
  r = r * s;
  s = fma (a, M_PI, r);
  return s;
}

double ksinpi_ernst(int k, int n) {			// Using modified Ernst's program k/n < 0.25, x in range [0..1]
  const double
  S0 = +3.14159265358979322857549585649252479170596805578751964804604910e+00,
  S1 = -5.16771278004995452583938866754421448491805106851652909471625339e+00,
  S2 = +2.55016403987337610028335512515697142485381483455575768286101741e+00,
  S3 = -5.99264528939644926932658650083463907769354154612435064156617923e-01,
  S4 = +8.21458691800017461862925981661435393176563606567669316878047234e-02,
  S5 = -7.37002158690777026975091957439476212577226315422626427540194454e-03,
  S6 = +4.61531855383580995897596361983564763395850652294559984699291272e-04;
  double x = double(k) / double(n);
  double z = x * x;
//  double z = (double(k) * double(k)) / (double(n) * double(n));	// This gives worse accuracy though I think it ought to give better accuracy!
  double r = fma(fma(fma(fma(fma(S6, z, S5), z, S4), z, S3), z, S2), z, S1) * (z * x);
  return fma(S0, x, r);
}

double ksinpi_ernst(int k, int n, double S[7]) {		// Using modified Ernst's program k/n < 0.25, x in range [0..1]
  double x = k;
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]) * (z * x);
  return fma(S[0], x, r);
}

double ksinpi_ernst(int k, int n, int multiplier, double S[7]) {	// Using modified Ernst's program k/n < 0.25, x in range [0..1]
  double x = double(k) * (double(multiplier) / double(n));
  double z = x * x;
  double r = fma(fma(fma(fma(fma(S[6], z, S[5]), z, S[4]), z, S[3]), z, S[2]), z, S[1]) * (z * x);
  return fma(S[0], x, r);
}



double kcos_fast(double k, double n) {
  double
//C00 = +9.99999999999999999969670443201471068399669429945726867342133369e-01,
  C0  = -4.93480220054467924730133340069049897627060272547106343170436752e+00 * 1.0 / (n * n),
  C1  = +4.05871212641674734556114676524460087157036330280895752170708528e+00 * 1.0 / (n * n * n * n),
  C2  = -1.33526276885191738260256045657669365723511854417210939209829062e+00 * 1.0 / (n * n * n * n * n * n),
  C3  = +2.35330630190887864529753667631633457993616238952635815846025798e-01 * 1.0 / (n * n * n * n * n * n * n * n),
  C4  = -2.58068856529513046472158981821569266131241613330683601835828971e-02 * 1.0 / (n * n * n * n * n * n * n * n * n * n),
  C5  = +1.92946574408000430568731323183265127732231903063785685866565227e-03 * 1.0 / (n * n * n * n * n * n * n * n * n * n * n * n),
  C6  = -1.03567472551994792518140040785125510857189953475207506740370448e-04 * 1.0 / (n * n * n * n * n * n * n * n * n * n * n * n * n * n);
  double x = k;
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C6, z, C5), z, C4), z, C3), z, C2), z, C1), z, C0), z, 1.0);
}

double kcos_sun(double x, double z) {
  // Coefficients from http://www.netlib.org/fdlibm/k_cos.c
  const double 
  C1  =  4.16666666666666019037e-02, /* 0x3FA55555, 0x5555554C */
  C2  = -1.38888888888741095749e-03, /* 0xBF56C16C, 0x16C15177 */
  C3  =  2.48015872894767294178e-05, /* 0x3EFA01A0, 0x19CB1590 */
  C4  = -2.75573143513906633035e-07, /* 0xBE927E4F, 0x809C52AD */
  C5  =  2.08757232129817482790e-09, /* 0x3E21EE9E, 0xBDB4B1C4 */
  C6  = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */
  return fma(fma(fma(fma(fma(fma(fma(C6, z, C5), z, C4), z, C3), z, C2), z, C1), z, -0.5), z, 1.0);
}

double kcos_sun(double x) {
  return kcos_sun(x, x * x);
}

double kcos_ernst(double x, double z) {
  const double 
//C00 = +9.99999999999999999969670443201471068399669429945726867342133369e-01,
//C0  = -4.99999999999999993706342263080825580843041001984274274704859171e-01,
  C1  = +4.16666666666664523886804383503486252152442239545330965582130625e-02,
  C2  = -1.38888888888610945961029774887229380857669399497858225293821236e-03,
  C3  = +2.48015872838811530035452909953032636542759385223976822151624913e-05,
  C4  = -2.75573130977900862706417627743837036747476258024932866392465620e-07,
  C5  = +2.08755823806639530442925413816102218334564909460318791425328810e-09,
  C6  = -1.13533796382975741261148850741646039533852219102309237307649831e-11;
  return fma(fma(fma(fma(fma(fma(fma(C6, z, C5), z, C4), z, C3), z, C2), z, C1), z, -0.5), z, 1.0);
}

double kcos_ernst(double x) {
  return kcos_ernst(x, x * x);
}

// adapted from https://stackoverflow.com/questions/42792939/
// CAUTION: this function only works for arguments in the range [-0.25; 0.25]!
double kcospi(int k, int n) {
  double a = k;
  a = a / n;
  double s = a * a;
  /* Approximate cos(pi*x) for x in [-0.25,0.25] */
  double r =     -1.0369917389758117e-4;
  r = fma (r, s,  1.9294935641298806e-3);
  r = fma (r, s, -2.5806887942825395e-2);
  r = fma (r, s,  2.3533063028328211e-1);
  r = fma (r, s, -1.3352627688538006e+0);
  r = fma (r, s,  4.0587121264167623e+0);
  r = fma (r, s, -4.9348022005446790e+0);
  return fma (r, s, 1.0);
}

double kcospi_ernst(int k, int n) {
  const double 
//C00 = 9.99999999999999999969670443201471068399669429945726867342133369e-01,
  C0  = -4.93480220054467924730133340069049897627060272547106343170436752e+00,
  C1  = +4.05871212641674734556114676524460087157036330280895752170708528e+00,
  C2  = -1.33526276885191738260256045657669365723511854417210939209829062e+00,
  C3  = +2.35330630190887864529753667631633457993616238952635815846025798e-01,
  C4  = -2.58068856529513046472158981821569266131241613330683601835828971e-02,
  C5  = +1.92946574408000430568731323183265127732231903063785685866565227e-03,
  C6  = -1.03567472551994792518140040785125510857189953475207506740370448e-04;
  double x = double(k) / double(n);
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C6, z, C5), z, C4), z, C3), z, C2), z, C1), z, C0), z, 1.0);
}

double kcospi_ernst(int k, int n, double C[7]) {
  double x = double(k);
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C[6], z, C[5]), z, C[4]), z, C[3]), z, C[2]), z, C[1]), z, C[0]), z, 1.0);
}

double kcospi_ernst(int k, int n, int multiplier, double C[7]) {
  double x = double(k) * (double(multiplier) / double(n));
  double z = x * x;
  return fma(fma(fma(fma(fma(fma(fma(C[6], z, C[5]), z, C[4]), z, C[3]), z, C[2]), z, C[1]), z, C[0]), z, 1.0);
}

int main(int argc, char *argv[])
{
	dd_real  toterr_angle, toterr_angle_sq, toterr_angle_sq2;
	dd_real  totrelerr_angle, totrelerr_angle_sq, totrelerr_angle_sq2;
	dd_real  toterr_perfect, toterr_crt, toterr_fast, toterr_sun, toterr_sun_sq, toterr_ernst, toterr_ernst_sq, toterr_ernst_cube, toterr_pi, toterr_pi_ernst, toterr_pi_ernst2, toterr_pi_ernst3;
	dd_real  totrelerr_perfect, totrelerr_crt, totrelerr_fast, totrelerr_sun, totrelerr_sun_sq, totrelerr_ernst, totrelerr_ernst_sq, totrelerr_ernst_cube, totrelerr_pi, totrelerr_pi_ernst, totrelerr_pi_ernst2, totrelerr_pi_ernst3;
	dd_real  ctoterr_perfect, ctoterr_crt, ctoterr_fast, ctoterr_sun, ctoterr_sun_sq, ctoterr_ernst, ctoterr_ernst_sq, ctoterr_pi, ctoterr_pi_ernst, ctoterr_pi_ernst2, ctoterr_pi_ernst3;
	dd_real  ctotrelerr_perfect, ctotrelerr_crt, ctotrelerr_fast, ctotrelerr_sun, ctotrelerr_sun_sq, ctotrelerr_ernst, ctotrelerr_ernst_sq, ctotrelerr_pi, ctotrelerr_pi_ernst, ctotrelerr_pi_ernst2, ctotrelerr_pi_ernst3;
	double pi_over_n, pi_squared_over_n_squared, pi_cubed_over_n_cubed;
	double C[7],S[7], C2[7],S2[7];
	int multiplier, cmultiplier;

	int CALC_CONSTS_WITH_DOUBLES = 0;

	for (int middle = 7; middle <= 15; middle++) {
		if (middle == 7) multiplier = 7 * 37, cmultiplier = 7 * 149;
		if (middle == 8) multiplier = 1 * 181, cmultiplier = 1 * 237;
		if (middle == 9) multiplier = 9 * 109, cmultiplier = 9 * 31;
		if (middle == 10) multiplier = 5 * 187, cmultiplier = 5 * 125;
		if (middle == 11) multiplier = 11 * 85, cmultiplier = 11 * 123;
		if (middle == 12) multiplier = 3 * 181, cmultiplier = 3 * 79;
		if (middle == 13) multiplier = 13 * 75, cmultiplier = 13 * 71;
		if (middle == 14) continue; //multiplier = 7 * 149;
		if (middle == 15) multiplier = 15 * 65, cmultiplier = 15 * 59;

	int n = 1024 * middle * 256 / 2;		// Emulate slowtrig calls in tailFused
	printf ("\nTrig calls for a %d*512K FFT, multipliers %d, %d\n\n", middle, multiplier, cmultiplier);

	// Calc constants for sinpi_ernst, cospi_ernst
	if (CALC_CONSTS_WITH_DOUBLES) {
		double tmp, ndbl;
		ndbl = double(n);
		tmp =+3.14159265358979322857549585649252479170596805578751964804604910e+00;
		S[0] = tmp / ndbl; 
		tmp =-5.16771278004995452583938866754421448491805106851652909471625339e+00;
		S[1] = tmp / (ndbl * ndbl * ndbl);
		tmp =+2.55016403987337610028335512515697142485381483455575768286101741e+00;
		S[2] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =-5.99264528939644926932658650083463907769354154612435064156617923e-01;
		S[3] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =+8.21458691800017461862925981661435393176563606567669316878047234e-02;
		S[4] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =-7.37002158690777026975091957439476212577226315422626427540194454e-03;
		S[5] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =+4.61531855383580995897596361983564763395850652294559984699291272e-04;
		S[6] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =-4.93480220054467924730133340069049897627060272547106343170436752e+00;
		C[0] = tmp / (ndbl * ndbl);
		tmp =+4.05871212641674734556114676524460087157036330280895752170708528e+00;
		C[1] = tmp / (ndbl * ndbl * ndbl * ndbl);
		tmp =-1.33526276885191738260256045657669365723511854417210939209829062e+00;
		C[2] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =+2.35330630190887864529753667631633457993616238952635815846025798e-01;
		C[3] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =-2.58068856529513046472158981821569266131241613330683601835828971e-02;
		C[4] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =+1.92946574408000430568731323183265127732231903063785685866565227e-03;
		C[5] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
		tmp =-1.03567472551994792518140040785125510857189953475207506740370448e-04;
		C[6] = tmp / (ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl * ndbl);
	} else {
		dd_real tmp, err;

// This code lets us look for multipliers with small relative error rounding S0 and C0 to a double.
#if 0
tmp.read ("+9.99999999999999999969670443201471068399669429945726867342133369e-01", tmp);
S[0] = to_double(tmp); err = (tmp - to_double(tmp)) / tmp;	printf ("Cosine one err: %g\n", to_double(err)); 
for (int j = 1; j <= 15; j += 2) {
printf ("\nj: %d\n\n", j);
for (int i = 1; i < 200; i += 2) {
tmp.read ("+3.14159265358979322857549585649252479170596805578751964804604910e+00", tmp);
tmp = tmp * (dd_real(double(j) * double(i)) ^ -1);
S[0] = to_double(tmp); err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %d  %g\n", i, to_double(err)); err=0.0;
tmp.read ("-4.93480220054467924730133340069049897627060272547106343170436752e+00", tmp);
tmp = tmp * (dd_real(double(j) * double(i)) ^ -2);
S[0] = to_double(tmp); err = (tmp - to_double(tmp)) / tmp;	printf ("Cerr: %g\n", to_double(err)); err=0.0;
}
	}
#endif

		tmp.read ("+3.14159265358979322857549585649252479170596805578751964804604910e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -1);
		S[0] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("-5.16771278004995452583938866754421448491805106851652909471625339e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -3);
		S[1] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("+2.55016403987337610028335512515697142485381483455575768286101741e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -5);
		S[2] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("-5.99264528939644926932658650083463907769354154612435064156617923e-01", tmp);
		tmp = tmp * (dd_real(n) ^ -7);
		S[3] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("+8.21458691800017461862925981661435393176563606567669316878047234e-02", tmp);
		tmp = tmp * (dd_real(n) ^ -9);
		S[4] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("-7.37002158690777026975091957439476212577226315422626427540194454e-03", tmp);
		tmp = tmp * (dd_real(n) ^ -11);
		S[5] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("+4.61531855383580995897596361983564763395850652294559984699291272e-04", tmp);
		tmp = tmp * (dd_real(n) ^ -13);
		S[6] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.read ("-4.93480220054467924730133340069049897627060272547106343170436752e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -2);
		C[0] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("+4.05871212641674734556114676524460087157036330280895752170708528e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -4);
		C[1] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("-1.33526276885191738260256045657669365723511854417210939209829062e+00", tmp);
		tmp = tmp * (dd_real(n) ^ -6);
		C[2] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("+2.35330630190887864529753667631633457993616238952635815846025798e-01", tmp);
		tmp = tmp * (dd_real(n) ^ -8);
		C[3] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("-2.58068856529513046472158981821569266131241613330683601835828971e-02", tmp);
		tmp = tmp * (dd_real(n) ^ -10);
		C[4] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("+1.92946574408000430568731323183265127732231903063785685866565227e-03", tmp);
		tmp = tmp * (dd_real(n) ^ -12);
		C[5] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.read ("-1.03567472551994792518140040785125510857189953475207506740370448e-04", tmp);
		tmp = tmp * (dd_real(n) ^ -14);
		C[6] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;

		char buf[80];
		tmp.read ("+3.14159265358979322857549585649252479170596805578751964804604910e+00", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -1);
		S2[0] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S0 = %s\n", buf);
		tmp.read ("-5.16771278004995452583938866754421448491805106851652909471625339e+00", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -3);
		S2[1] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S1 = %s\n", buf);
		tmp.read ("+2.55016403987337610028335512515697142485381483455575768286101741e+00", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -5);
		S2[2] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S2 = %s\n", buf);
		tmp.read ("-5.99264528939644926932658650083463907769354154612435064156617923e-01", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -7);
		S2[3] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S3 = %s\n", buf);
		tmp.read ("+8.21458691800017461862925981661435393176563606567669316878047234e-02", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -9);
		S2[4] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S4 = %s\n", buf);
		tmp.read ("-7.37002158690777026975091957439476212577226315422626427540194454e-03", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -11);
		S2[5] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S5 = %s\n", buf);
		tmp.read ("+4.61531855383580995897596361983564763395850652294559984699291272e-04", tmp);
		tmp = tmp * (dd_real(multiplier) ^ -13);
		S2[6] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;	printf ("Serr: %g\n", to_double(err)); err=0.0;
		tmp.write (buf, 40);
		printf ("S6 = %s\n", buf);
		tmp.read ("-4.93480220054467924730133340069049897627060272547106343170436752e+00", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -2);
		C2[0] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C0 = %s\n", buf);
		tmp.read ("+4.05871212641674734556114676524460087157036330280895752170708528e+00", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -4);
		C2[1] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C1 = %s\n", buf);
		tmp.read ("-1.33526276885191738260256045657669365723511854417210939209829062e+00", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -6);
		C2[2] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C2 = %s\n", buf);
		tmp.read ("+2.35330630190887864529753667631633457993616238952635815846025798e-01", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -8);
		C2[3] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C3 = %s\n", buf);
		tmp.read ("-2.58068856529513046472158981821569266131241613330683601835828971e-02", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -10);
		C2[4] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C4 = %s\n", buf);
		tmp.read ("+1.92946574408000430568731323183265127732231903063785685866565227e-03", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -12);
		C2[5] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C5 = %s\n", buf);
		tmp.read ("-1.03567472551994792518140040785125510857189953475207506740370448e-04", tmp);
		tmp = tmp * (dd_real(cmultiplier) ^ -14);
		C2[6] = to_double(tmp); //err = (tmp - to_double(tmp)) / tmp;
		tmp.write (buf, 40);
		printf ("C6 = %s\n", buf);

	}

	toterr_angle = toterr_angle_sq = toterr_angle_sq2 = totrelerr_angle = totrelerr_angle_sq = totrelerr_angle_sq2 = 0.0;
	toterr_perfect = toterr_crt = toterr_fast = toterr_sun = toterr_sun_sq = toterr_ernst = toterr_ernst_sq = toterr_ernst_cube = toterr_pi = toterr_pi_ernst = toterr_pi_ernst2 = toterr_pi_ernst3 =
	totrelerr_perfect = totrelerr_crt = totrelerr_fast = totrelerr_sun = totrelerr_sun_sq = totrelerr_ernst = totrelerr_ernst_sq = totrelerr_ernst_cube = totrelerr_pi = totrelerr_pi_ernst = totrelerr_pi_ernst2 = totrelerr_pi_ernst3 = 0.0;
	ctoterr_perfect = ctoterr_crt = ctoterr_fast = ctoterr_sun = ctoterr_sun_sq = ctoterr_ernst = ctoterr_ernst_sq = ctoterr_pi = ctoterr_pi_ernst = ctoterr_pi_ernst2 = ctoterr_pi_ernst3 =
	ctotrelerr_perfect = ctotrelerr_crt = ctotrelerr_fast = ctotrelerr_sun = ctotrelerr_sun_sq = ctotrelerr_ernst = ctotrelerr_ernst_sq = ctotrelerr_pi = ctotrelerr_pi_ernst = ctotrelerr_pi_ernst2 = ctotrelerr_pi_ernst3 = 0.0;

	if (CALC_CONSTS_WITH_DOUBLES) {		// Emulate OpenCL compiler computing PI / n and PI^2 / n^2  (likely uses doubles)
	  printf ("\nCalculating pi/n and (pi/n)^2 using doubles\n\n");
	  pi_over_n = M_PI / double(n);
	  if (pi_over_n != to_double(dd_real::_pi / double(n))) printf ("Pi / n calc needs dd_real precision\n");
	  pi_squared_over_n_squared = to_double(dd_real::_pi * dd_real::_pi) / (double(n) * double(n));
	  if (pi_squared_over_n_squared != to_double(dd_real::_pi * dd_real::_pi / (double(n) * double(n)))) printf ("Pi^2 / n^2 calc needs dd_real precision\n");
	  pi_cubed_over_n_cubed = to_double(dd_real::_pi * dd_real::_pi * dd_real::_pi) / (double(n) * double(n) * double(n));
	  if (pi_cubed_over_n_cubed != to_double(dd_real::_pi * dd_real::_pi * dd_real::_pi / (double(n) * double(n) * double(n)))) printf ("Pi^3 / n^3 calc needs dd_real precision\n");
	} else {		// Emulate  PI / n and PI^2 / n^2 computed as long doubles and passed as #defines to OpenCL compiler
	  printf ("\nCalculating pi/n and (pi/n)^2 using long doubles\n\n");
	  pi_over_n = to_double(dd_real::_pi / double(n));
	  if (pi_over_n != M_PI / double(n)) printf ("Pi / n calc needs dd_real precision\n");
	  pi_squared_over_n_squared = to_double(dd_real::_pi * dd_real::_pi / (double(n) * double(n)));
	  if (pi_squared_over_n_squared != (M_PI * M_PI) / (double(n) * double(n))) printf ("Pi^2 / n^2 calc needs dd_real precision\n");
	  pi_cubed_over_n_cubed = to_double(dd_real::_pi * dd_real::_pi * dd_real::_pi / (double(n) * double(n) * double(n)));
	  if (pi_cubed_over_n_cubed != to_double(dd_real::_pi * dd_real::_pi * dd_real::_pi) / (double(n) * double(n) * double(n))) printf ("Pi^3 / n^3 calc needs dd_real precision\n");
	}

	for (int k = 1; k < n / 4; k++) {
		double angle, angle_squared, angle_cubed, sine_crt, sine_fast, sine_s, sine_s_sq, sine_e, sine_e_sq, sine_e_cube, sine_pi, sine_pi_e, sine_pi_e2, sine_pi_e3;
		double cosine_crt, cosine_fast, cosine_s, cosine_s_sq, cosine_e, cosine_e_sq, cosine_pi, cosine_pi_e, cosine_pi_e2, cosine_pi_e3;
		dd_real angle_dd, sine_dd, cosine_dd, err;

		angle_dd = dd_real(double(k)) / dd_real(double(n)) * dd_real::_pi;
		angle = double(k) * pi_over_n;
		angle_squared = (double(k) * double(k)) * pi_squared_over_n_squared;
		angle_cubed = (double(k) * double(k) * double(k)) * pi_cubed_over_n_cubed;

		//if (angle_squared != angle * angle) printf ("Angle^2 needs dd_real precision\n");

		sine_crt = sin(angle);
		sine_fast = ksin_fast(double(k), double(n));
		sine_s = ksin_sun(angle);
		sine_s_sq = ksin_sun(angle, angle_squared);
		sine_e = ksin_ernst(angle);
		sine_e_sq = ksin_ernst(angle, angle_squared);
		sine_e_cube = ksin_ernst(angle, angle_squared, angle_cubed);
		sine_pi = ksinpi(k, n);
		sine_pi_e = ksinpi_ernst(k, n);
		sine_pi_e2 = ksinpi_ernst(k, n, S);
		sine_pi_e3 = ksinpi_ernst(k, n, multiplier, S2);
		sine_dd = sin(angle_dd);

		err = abs(angle_dd - angle);
		toterr_angle += err;
		totrelerr_angle += err / angle_dd;

		err = abs(angle_dd * angle_dd - angle_squared);
		toterr_angle_sq += err;
		totrelerr_angle_sq += err / (angle_dd * angle_dd);

		err = abs(angle_dd * angle_dd - (angle * angle));
		toterr_angle_sq2 += err;
		totrelerr_angle_sq2 += err / (angle_dd * angle_dd);

		err = abs(sine_dd - to_double(sine_dd));
		toterr_perfect += err;
		totrelerr_perfect += err / sine_dd;

		err = abs(sine_dd - sine_crt);
		toterr_crt += err;
		totrelerr_crt += err / sine_dd;

		err = abs(sine_dd - sine_fast);
		toterr_fast += err;
		totrelerr_fast += err / sine_dd;

		err = abs(sine_dd - sine_s);
		toterr_sun += err;
		totrelerr_sun += err / sine_dd;

		err = abs(sine_dd - sine_s_sq);
		toterr_sun_sq += err;
		totrelerr_sun_sq += err / sine_dd;

		err = abs(sine_dd - sine_e);
		toterr_ernst += err;
		totrelerr_ernst += err / sine_dd;

		err = abs(sine_dd - sine_e_sq);
		toterr_ernst_sq += err;
		totrelerr_ernst_sq += err / sine_dd;

		err = abs(sine_dd - sine_e_cube);
		toterr_ernst_cube += err;
		totrelerr_ernst_cube += err / sine_dd;

		err = abs(sine_dd - sine_pi);
		toterr_pi += err;
		totrelerr_pi += err / sine_dd;

		err = abs(sine_dd - sine_pi_e);
		toterr_pi_ernst += err;
		totrelerr_pi_ernst += err / sine_dd;

		err = abs(sine_dd - sine_pi_e2);
		toterr_pi_ernst2 += err;
		totrelerr_pi_ernst2 += err / sine_dd;

		err = abs(sine_dd - sine_pi_e3);
		toterr_pi_ernst3 += err;
		totrelerr_pi_ernst3 += err / sine_dd;

		cosine_crt = cos(angle);
		cosine_fast = kcos_fast(double(k), double(n));
		cosine_s = kcos_sun(angle);
		cosine_s_sq = kcos_sun(angle, angle_squared);
		cosine_e = kcos_ernst(angle);
		cosine_e_sq = kcos_ernst(angle, angle_squared);
		cosine_pi = kcospi(k, n);
		cosine_pi_e = kcospi_ernst(k, n);
		cosine_pi_e2 = kcospi_ernst(k, n, C);
		cosine_pi_e3 = kcospi_ernst(k, n, cmultiplier, C2);
		cosine_dd = cos(angle_dd);

		err = abs(cosine_dd - to_double(cosine_dd));
		ctoterr_perfect += err;
		ctotrelerr_perfect += err / cosine_dd;

		err = abs(cosine_dd - cosine_crt);
		ctoterr_crt += err;
		ctotrelerr_crt += err / cosine_dd;

		err = abs(cosine_dd - cosine_fast);
		ctoterr_fast += err;
		ctotrelerr_fast += err / cosine_dd;

		err = abs(cosine_dd - cosine_s);
		ctoterr_sun += err;
		ctotrelerr_sun += err / cosine_dd;

		err = abs(cosine_dd - cosine_s_sq);
		ctoterr_sun_sq += err;
		ctotrelerr_sun_sq += err / cosine_dd;

		err = abs(cosine_dd - cosine_e);
		ctoterr_ernst += err;
		ctotrelerr_ernst += err / cosine_dd;

		err = abs(cosine_dd - cosine_e_sq);
		ctoterr_ernst_sq += err;
		ctotrelerr_ernst_sq += err / cosine_dd;

		err = abs(cosine_dd - cosine_pi);
		ctoterr_pi += err;
		ctotrelerr_pi += err / cosine_dd;

		err = abs(cosine_dd - cosine_pi_e);
		ctoterr_pi_ernst += err;
		ctotrelerr_pi_ernst += err / cosine_dd;

		err = abs(cosine_dd - cosine_pi_e2);
		ctoterr_pi_ernst2 += err;
		ctotrelerr_pi_ernst2 += err / cosine_dd;

		err = abs(cosine_dd - cosine_pi_e3);
		ctoterr_pi_ernst3 += err;
		ctotrelerr_pi_ernst3 += err / cosine_dd;

		if (k == n / 4 - 1) {
			printf ("Angle stats\n");
			printf ("k: %u Angle err:         %g relerr: %g\n", k, to_double(toterr_angle), to_double(totrelerr_angle));
			printf ("k: %u Angle_squared err: %g relerr: %g\n", k, to_double(toterr_angle_sq), to_double(totrelerr_angle_sq));
			printf ("k: %u Angle*Angle err:   %g relerr: %g\n", k, to_double(toterr_angle_sq2), to_double(totrelerr_angle_sq2));
			printf ("Sine stats\n");
			printf ("k: %u perfection err: %g   relerr: %g\n", k, to_double(toterr_perfect), to_double(totrelerr_perfect));
			printf ("k: %u CRT lib err:    %g   relerr: %g\n", k, to_double(toterr_crt), to_double(totrelerr_crt));
			printf ("k: %u Fast err:       %g   relerr: %g\n", k, to_double(toterr_fast), to_double(totrelerr_fast));
			printf ("k: %u Sun err:        %g   relerr: %g\n", k, to_double(toterr_sun), to_double(totrelerr_sun));
			printf ("k: %u Sun sq err:     %g   relerr: %g\n", k, to_double(toterr_sun_sq), to_double(totrelerr_sun_sq));
			printf ("k: %u Ernst err:      %g   relerr: %g\n", k, to_double(toterr_ernst), to_double(totrelerr_ernst));
			printf ("k: %u Ernst sq err:   %g   relerr: %g\n", k, to_double(toterr_ernst_sq), to_double(totrelerr_ernst_sq));
			printf ("k: %u Ernst cube err: %g   relerr: %g\n", k, to_double(toterr_ernst_cube), to_double(totrelerr_ernst_cube));
			printf ("k: %u Pi err:         %g   relerr: %g\n", k, to_double(toterr_pi), to_double(totrelerr_pi));
			printf ("k: %u Pi ernst err:   %g   relerr: %g\n", k, to_double(toterr_pi_ernst), to_double(totrelerr_pi_ernst));
			printf ("k: %u Pi ernst2 err:  %g   relerr: %g\n", k, to_double(toterr_pi_ernst2), to_double(totrelerr_pi_ernst2));
			printf ("k: %u Pi ernst3 err:  %g   relerr: %g\n", k, to_double(toterr_pi_ernst3), to_double(totrelerr_pi_ernst3));
			printf ("Cosine stats\n");
			printf ("k: %u perfection err: %g   relerr: %g\n", k, to_double(ctoterr_perfect), to_double(ctotrelerr_perfect));
			printf ("k: %u CRT lib err:    %g   relerr: %g\n", k, to_double(ctoterr_crt), to_double(ctotrelerr_crt));
			printf ("k: %u Fast err:       %g   relerr: %g\n", k, to_double(ctoterr_fast), to_double(ctotrelerr_fast));
			printf ("k: %u Sun err:        %g   relerr: %g\n", k, to_double(ctoterr_sun), to_double(ctotrelerr_sun));
			printf ("k: %u Sun sq err:     %g   relerr: %g\n", k, to_double(ctoterr_sun_sq), to_double(ctotrelerr_sun_sq));
			printf ("k: %u Ernst err:      %g   relerr: %g\n", k, to_double(ctoterr_ernst), to_double(ctotrelerr_ernst));
			printf ("k: %u Ernst sq err:   %g   relerr: %g\n", k, to_double(ctoterr_ernst_sq), to_double(ctotrelerr_ernst_sq));
			printf ("k: %u Pi err:         %g   relerr: %g\n", k, to_double(ctoterr_pi), to_double(ctotrelerr_pi));
			printf ("k: %u Pi ernst err:   %g   relerr: %g\n", k, to_double(ctoterr_pi_ernst), to_double(ctotrelerr_pi_ernst));
			printf ("k: %u Pi ernst2 err:  %g   relerr: %g\n", k, to_double(ctoterr_pi_ernst2), to_double(ctotrelerr_pi_ernst2));
			printf ("k: %u Pi ernst3 err:  %g   relerr: %g\n", k, to_double(ctoterr_pi_ernst3), to_double(ctotrelerr_pi_ernst3));
		}
	}
	}
}
