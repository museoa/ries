/*
    zeta.cpp, an attempt to compute the Riemann Zeta function
    Copyright (C) 2000-2013 Robert P. Munafo

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

    If you got ries.c from the website www.mrob.com, the
    GNU General Public License may be retrieved from the following URL:

    http://www.mrob.com/ries/COPYING.txt

    To compile, I currently use:

      gcc -lstdc++ zeta.cpp -lm -o zeta

    On the Intel Core 2 architecture, a speedup of about 30% is achieved
    using the flags -O3 and -ffast-math.

    If given an argument T, it prints a graph of the Zeta function on the
    critical line (z = 0.5 + i t) from t = T-100 to t = T.

    If not given an argument, it looks for a file called "cdda.wav"
    (which should be 44100-Hz, PCM, 16-bit, stereo) and overwrites it
    with the Zeta function, samples spaced 0.01 apart on the complex
    plane. Thus, each second of audio will cover 441.0 units on the
    imaginary axis.

    This program does not "know" enough about the WAV file format to
    be able to create a WAV file from scratch -- however I suspect that
    would be fairly easy to do.

    I have a description of the AIFF format in "AIFF-1.3.pdf" (in
    another project) but do not currently have a WAV format description.

 */

/* Use the following #define to make it evaluate a pre-defined number
   of samples, for benchmarking purposes.

#define BENCH_NSAMP 10000
*/

/*

REVISION HISTORY

 20000302 Initial version
 20000303 Post query to sci.math
 20000304 Add notes from a reply from sci.math

 20050804 Lots of fixes to make it compatible with official ISO
implementation of __complex__. Make little-endian-ness of WAV format
explicit (so it works on a PowerPC). Run it on a 2-GHz G5, the speed
is truly impressive! Increase the number of terms from 1000 to 10000.

 20101017 Improve output formatting slightly.

 20120102 Add Lanczos Gamma function
 20120109 Add Pugh coefficients for Lanczos
 20120110 Start using f107_o, add test_f107.
 20120111 Convert Lanczos gamma functions to f107.
 20120116 u64/s64 need to be 'long long'
 20131031 Add tests of floor()
 20140206 Most test functions moved to test_f107.cpp
 20140223 Remove the last of the (now unused) gamma and f107 functions.

TO DO

Try to find a better way to compute the Zeta function. Here are some
candidates (compiled on 20100528):

1. The Knopp/Hasse series (formula 4 in "Sondow 1994 Analytic.pdf"):

  zeta(s) = 1/(1-2^(1-s))
      * SIGMAv{n=0..inf} [ 1/(2^(n+1)) SIGMAv{k=0..n} [ -1^k (n k) (k+1)^-s ] ]

2. Algorithm 2 in "Borwein 2000 Efficient.pdf". Select an N, and use:

  zeta(s) = -1/(d(N)*(1-2^(1-s)))
     * SIGMAv{k=0..N-1} [ -1^k (d(k)-d(N))/(k+1)^s ] + error

  d(k) = N * SIGMAv{i=0..k} [ (N+i+1)! * 4^i / ((N-1)!*(2i)!) ]

and note that |error| decreases like 1/(3+sqrt(8))^N for Re(s) >= 1/2.
If many zeta(s) values are being computed with the same N, the d(k)
can be precomputed.

3. Formula 34 in "Borwein 1999 Computational.pdf":

  zeta(s) = limv{N->inf} 1/(2^(N-s+1)-2^N)
      * SIGMAv{k=0..2N-1} [ -1^k/(k+1)^s SIGMAv{m=0..k-N} [(N m)-2^N] ]

4. Algorithm 3 in "Borwein 2000 Efficient.pdf". Select an N, and use:

  zeta(s) = -1/(2^N(1-2^(1-s))) * SIGMAv{j=0..2N-1} [e_j/(j+1)^s] + error

  e_j = -1^j * SIGMAv{k=0..j-N} [(N k) - 2^N]

and note that |error| decreases like 1/8^N for Re(s)>0 (and a somewhat
more complex error bound for Re(s)<=0). If many zeta(s) values are
being computed with the same N, the e_j can be precomputed. Borwein
notes that this method is "even simpler, though not quite as fast" as
the above "Algorithm 2".

LINKS

http://web.viu.ca/pughg/RiemannZeta/RiemannZetaLong.html
  Nice Java applet and some fairly comprehensible explanations

SEE ALSO

There is other info about special functions in:
  .../proj/zeta/ken-takusagawa
  .../proj/ries/function-sources.txt
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef signed char    s8;
typedef unsigned char  u8;
typedef short          s16;
typedef unsigned short u16;
typedef int            s32;
typedef unsigned int   u32;
/* For the 64-bit types we must use 'long long' to work in the GCC compiler
   on 32-bit targets */
typedef long long      s64;
typedef unsigned long long  u64;

// Size of wav file header
#define WAV_START 44

#define LOUDNESS 1000.0

/* FREQUENCY is the number of units on the imaginary axis that will be
   heard each second. Imagine that the Zeta function has been carved
   into a phonograph record and we're playing the record. This constant
   tells how many units on the real axis the needle moves each second. */
#define FREQUENCY 441.0

/* INTERVAL is the spacing between values of the argument we pass to the
   Zeta function to produce the audio samples. Note the constant 44100 --
   we assume that our output sound file is 44100 samples per second. */
#define INTERVAL (FREQUENCY / 44100.0)

/* Function prototypes */
extern "C" {
  void check_sizes(void);

  double logaddexp(double a, double b);
  double cabssqr(__complex__ double z);
  double cabs(__complex__ double z);
  __complex__ double clog(__complex__ double z);
  __complex__ double cexp(__complex__ double z);
  __complex__ double cpow(double b, __complex__ double e);
 
  void cprint(__complex__ double z);

  __complex__ double zetacrit1(double zi, s32 terms);

  int main(int, char **);
}

void check_sizes(void)
{
  int s;

  s = (int) sizeof(u16);
  if (s != 2) {
    printf("sizeof(u16) == %d, expected size was 2\n", s);
    exit(-1);
  }

  s = (int) sizeof(u32);
  if (s != 4) {
    printf("sizeof(u32) == %d, expected size was 4\n", s);
    exit(-1);
  }

  s = (int) sizeof(u64);
  if (s != 8) {
    printf("sizeof(u64) == %d, expected size was 8\n", s);
    exit(-1);
  }
}



double g_pi = 3.1415926535897932;


/* logaddexp returns ln(exp(a) + exp(b)), but works even when both exp's
   would overflow and/or underflow */
double logaddexp(double a, double b)
{
  double l, s; /* l is the larger, s is the smaller */

  /*
  Given a and b, we want to calculate ln(exp(a)+exp(b)). Define eA=exp(a)
  and eB = exp(b). Then we want ln(eA+eB). Note that if eB is really small
  compared to eA, then roundoff reduces this to ln(eA) which is just a.

  ln(eA+eB) = ln(eA) + ln(1 + eB/eA)
            = a + ln(1 + exp(b - a))

  We start by putting the larger (i.e. higher) of a and b into "l" and the
  other into "s". */
  if (a > b) {
    l = a; s = b;
  } else {
    l = b; s = a;
  }

  if (s-l < -50.0) {
    /* exp(s-l) is less than 10^-20, so log(1.0+exp(s-l)) rounds off to 0 */
    return l;
  } else {
    return (l + log(1.0 + exp(s-l)));
  }
}

double cabssqr(__complex__ double z)
{
  return((__real__ z)*(__real__ z) + (__imag__ z)*(__imag__ z));
}

double cabs(__complex__ double z)
{
  return(sqrt(cabssqr(z)));
}

__complex__ double clog(__complex__ double z)
{
  return(log(cabs(z)) + (atan2(__imag__ z, __real__ z) * 1.0fi));
}

/* complex exponential function */
__complex__ double cexp(__complex__ double z)
{
  __complex__ double rv;

  rv = exp(__real__ z) * (cos(__imag__ z) + (1.0fi * sin(__imag__ z)));
  return(rv);
}


/* real base to a complex power */
__complex__ double cpow(double b, __complex__ double e)
{
  __complex__ double rv;

  //printf("base %10.6lg  log(base) %10.6lg\n", b, log(b));
  //printf("  exponent  %10.6lg + %10.6lg i\n", (__real__ e), (__imag__ e));

  rv = cexp(log(b) * e);
  return(rv);
}

void cprint(__complex__ double z)
{
  printf("%14.10lg + %14.10lg i", __real__ z, __imag__ z);
}

/*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
|
|                           COMPLEX ZETA FUNCTION
|
|___________________________________________________________________________*/

/* Here are some formulas to try, but some are probably only valid for
   real values, and most probably won't be useful:

   zeta(z) = 1/(1-2^(-z)) PRODUCT(p prime) 1/(1-p^(-z))
   don't know if "p prime" includes p=2 -- if not, the first term could
   be moved inside the product

   zeta(1-z) = (2/(2 pi)^z) cos(pi z/2) gamma(z) zeta(z)
   zeta(1/2 - it) = (2/(2 pi)^(1/2 + it)) cos(pi (1/2 + it)/2)
                    gamma(1/2 + it) zeta(1/2 + it)

   The "main term" of the Riemann-Seigel Z function:
   Z(t) ~= 2 SIGMA[k=1..v(t)] cos(theta(t) - t ln k) / sqrt(k)
   theta(t) = arg gamma(1/4 + it/2) - (t ln pi)/2
   v(t) = floor(sqrt(t/(2 pi)))
*/

/*
   This is a possibly-futile attempt to compute the Riemann-Siegel z(t)
   function, that is, |zeta(1/2 + i t)|

   For complex s with real(s) > 0, we have:

     Z(s) = 1/(1-2^(1-s)) SIGMA[n=1..inf] -1^(n-1) n^(-s)
          = 1/(1-2^(1-s)) {  SIGMA[odd n>0] n^(-s)
                           - SIGMA[even n>0] n^(-s)  }

   The terms converge at a rate of 1/sqrt(n), which is very slowly.
   They're smaller if you add consecutive terms, but it still converges
   just as slowly.
 */
__complex__ double zetacrit1(double zi, s32 terms)
{
  s32 i;
  __complex__ double z;
  __complex__ double m; /* the first multiplier, 1/(1-2^(1-s)) */
  __complex__ double t; /* one term in the sum */
  double n;
  __complex__ double sum;
  __complex__ double t1;
  __complex__ double rv;

  /* Compute the value of z on the critical line */
  z = 0.5 + (zi * 1.0fi);

  /* compute the first multiplier */
  m = 1.0 / (1.0 - cpow(2.0, (1.0 - z)));

  if (0) {
    printf("m = %10.6lg + %10.6lg i\n",
           (__real__ m), (__imag__ m));
  }

  /* Compute first term */
  n = 1.0;
  t = cpow(n, 0.0 - z);
  sum = t;

  /* Add some terms */
  i = 0;
  while (i < terms) {
    /* a negative term */
    n += 1.0;
    t1 = cpow(n, 0.0 - z);

    /* and a positive term */
    n += 1.0;
    t = cpow(n, 0.0 - z);
    t = t - t1;
    sum += t;
    if(0) {
      printf("n=%7lg  2 terms = %10.6lg + %10.6lg i", n,
             (__real__ t), (__imag__ t));
      printf("    sum = %10.6lg + %10.6lg i\n",
             (__real__ sum), (__imag__ sum));
    }
    i++;
  }

  //  printf("    sum = %10.6lg + %10.6lg i\n", (__real__ sum), (__imag__ sum));
  rv = m * sum;
  //  printf("    rv = %10.6lg + %10.6lg i\n", (__real__ rv), (__imag__ rv));
  return(rv);
}



/*
 From: Raymond Manzoni <raymman@club-internet.fr>
 Date: 03/04/2000
       sci.math
  Hi,
  
   Thank you for the take-off sound, 
   For which country? I don't know!
  
   Here are some of the papers that could interest you:
  
 A nice historical introduction may be found in 
 "Computational Number Theory at CWI in 1970-1994"
 but the paper (Riele-Lune.ps) doesn't seem available online any longer. 

 At CECM you'll find :
 "Computational strategies for the Riemann zeta function" at    
 http://www.cecm.sfu.ca/preprints/1998pp.html#98:118

 consult too Andrew Odlyzko's "Papers on Zeros of the Riemann Zeta
 Function and Related Topics" at
 http://www.research.att.com/~amo/doc/zeta.html (for example "Fast
 algorithms for multiple evaluations of the Riemann zeta function")

 and Andrew's "Tables of zeros of the Riemann zeta function" at
 http://www.research.att.com/~amo/zeta_tables/index.html

 For more papers concerning Riemann zeta look at :
 http://www.archetipo.web66.com/zeta/briefintro.htm


 Or forget all that and use simply the quick pari/gp for evaluation of
 zeta in the critical strip (available with C source at
 http://www.parigp-home.de/ )

 ftp://megrez.math.u-bordeaux.fr/pub/pari/unix/

  
 Hope it helped a little,

                 Raymond Manzoni

   I added too a resume of my earlier attempts using Euler Mac-Laurin : 

     I used the following formula  for that purpose:
 
     zeta(x) ~ sum(1/k^x, k=1..N)
               + 1/((x-1)*N^(x-1))
               -1/(2*N^x)
               +x/(12*N^(x+1))
               -x*(x+1)*(x+2)/(720*N^(x+3))
               +x*(x+1)*(x+2)*(x+3)*(x+4)/(30240*N^(x+5))
               +...

     The greater the N the better the result (but even N=10 is good) 
  
     The accuracy is rather fine when |Im(x)| < 2*PI*N and becomes
     very bad after that !
  
     You may find more terms using the more general Euler Mac-Laurin
     formula : This allows to explore better the negative values of x
     (in fact you will get the exact values for x=-1,-2, until -2
     times the number of Bernoulli terms!). With more terms the
     precision in the imaginary plane is better under 2*PI*N and worse
     over it!
  
    f(x) ~  sum(f(k), k=1..N)
             + int(f(u),u=N..oo)
             -f(N)/2
             -B1/2!*f'(N)
             +B2/4!*f'''(N)
             -...
     where Bn means the nth Bernoulli number

     This formula was very fruitful for me even on my little TI 92.
     (N=100 was enough)

 */


int main(int nargs, char ** argv)
{
  char ** av;
  int   args;
  char * arg;
  double zi;
  s16   got_zi;  /* got z? */

  check_sizes();

  got_zi = 0;

  if (nargs > 1) {
    av = argv;
    args = nargs;

    av++; args--;  /* skip our program name/path */

    while(args) {
      arg = *av; av++; args--;
      
      /* Test for number */
      if (((arg[0] >= '0') && (arg[0] <= '9'))
          || (arg[0] == '-') || (arg[0] == '.')) {
        sscanf(arg, "%lf", &zi);
        got_zi = 1;
      }
    }
  }

  if (0) {
    printf("zetacrit1(0.5 + b i) = %14.10lg\n", cabs(zetacrit1(zi, 100)));
    printf("                1000   %14.10lg\n", cabs(zetacrit1(zi, 1000)));
    printf("               10000   %14.10lg\n", cabs(zetacrit1(zi, 10000)));
    printf("              100000   %14.10lg\n", cabs(zetacrit1(zi, 100000)));
    printf("             1000000   %14.10lg\n", cabs(zetacrit1(zi, 1000000)));
    exit(0);
  }

  if(got_zi) {
#define TRANSFORM(x) ((int) (((x) * 10.0) + 40.0))
    __complex__ double z;
    double t, z1, z2r, z2i;
    s32 terms;

    terms = 500;
    t = zi - 100.0;
    if (t < 0.0) {
      t = 0.0;
    }
    for(; t<zi; t += 0.1) {
      s16 i, i1, i2i, i2r, lim;

      /* Compute Zeta twice, using 500 terms and using 5000 terms. */
      z1 = __real__ zetacrit1(t, terms);
      z = zetacrit1(t, 10 * terms);

      //      printf(" %10.6lg + %10.6lg i\n", (__real__ z), (__imag__ z));
      z2r = __real__ z; z2i = __imag__ z;

      i1 = TRANSFORM(z1);
      i2i = TRANSFORM(z2i);
      i2r = TRANSFORM(z2r);

      printf("%-8.5g", t);
      lim = i1;
      if (i2i > lim) {
        lim = i2i;
      }
      if (i2r > lim) {
        lim = i2r;
      }
      if (lim < 40) {
        lim = 40;
      }
      if(lim > 80) {
        lim = 80;
      }
      for(i=0; i<=lim; i++) {
        if (i == i2i) {
          printf("i");
        } else if (i == i2r) {
          printf("r");
        } else if (i == i1) {
          printf(".");
        } else if (i == 40) {
          printf("|");
        } else {
          printf(" ");
        }
      }
      printf("\n");
    }
  } else {
    __complex__ double z;
    double t, z1r, z1i;
    s32  nsamp, ctr, cs;
    s16  samp[2];
    s8   dbuf[4];
    FILE * wav;

    /* Measure the size of the file, which tells us how many samples we
       will want to compute */
    wav = fopen("cdda.wav", "r+");
    fseek(wav, 0, SEEK_END);
    nsamp = ftell(wav);
    nsamp -= WAV_START;
    nsamp /= 4;
    nsamp--;

#ifdef BENCH_NSAMP
    if (nsamp > BENCH_NSAMP) {
      nsamp = BENCH_NSAMP;
    }
#endif

    printf("WAV file length: %d samples\n", nsamp);
    printf("Evaluating Zeta(1/2 + i t) for t in (0.0 .. %g)\n",
                                                 ((double) nsamp) * INTERVAL);
    printf("(each dot is 441 samples = 0.01 sec.)\n");

    fseek(wav, WAV_START, SEEK_SET);

    printf("t=%10.6lg", 0.0); fflush(stdout);
    t = 0.0; ctr = 0; cs = 0;
    while(nsamp > 0) {
      z = zetacrit1(t, 10000);
      z1r = __real__ z; z1i = __imag__ z;
      samp[0] = ((int) (z1i * LOUDNESS));
      samp[1] = ((int) (z1r * LOUDNESS));
      dbuf[0] = samp[0] & 0xff; dbuf[1] = samp[0] >> 8;
      dbuf[2] = samp[1] & 0xff; dbuf[3] = samp[1] >> 8;
      fwrite((void *) dbuf, 2, 2, wav);
      ctr++;
      if (ctr >= 441) {
        printf("."); fflush(stdout);
        ctr = 0;
        cs++;
        if (cs >= 66) {
          printf("\rt=%10.6lg", t); fflush(stdout);
          cs = 0;
        }
      }
      nsamp--;
      t += INTERVAL;
    }

    fclose(wav);
    printf("\nDone!\n");
  }

  return(0);
}
