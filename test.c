#include <stdio.h>
#include <math.h>
#include <fftw3.h>

//gcc -lfftw3 -lm -o test test.c
//inside the fft library root directory
int main(int argc, char *argv[]) {

  fftw_complex *out;
  double *in;
  fftw_plan p;
  int N = 100;
  int i;


  in = (double*) fftw_malloc(sizeof(double) * N);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);


  for (i = 0; i < N; i++) {
    in[i] = cos(N/(4*3.14)*i);
  }

  fftw_execute(p); /* repeat as needed */
  fftw_destroy_plan(p);

  for (i = 0; i < (N/2+1); i++) {
    printf("%f %f\n", out[i][0], out[i][1]);
  }

  fftw_free(in); fftw_free(out);
}
