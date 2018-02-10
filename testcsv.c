#include <stdio.h>
#include <math.h>
#include <fftw3.h>

//gcc -o test test.c -lfftw3 -lm
//inside the fft library root directory
int main(int argc, char *argv[]) {

  fftw_complex *out;
  double *in;
  fftw_plan p;
  int i;
  int freqs = 120000;
  int freq = 60000;
  int times = 4;
  int count = times*freqs;
  int freqmin = 20000;
  int freqmax = 60000;
  int result = 0;
  int f;

  FILE* fp = fopen("data.csv", "r");
  double* temp;

  in = (double*) fftw_malloc(sizeof(double) * count);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * count);
  p = fftw_plan_dft_r2c_1d(count, in, out, FFTW_ESTIMATE);


  // for (i = 0; i < count; i++) {
  //   in[i] = cos(2*3.1415926*i*freq/freqs);//2pi*freq*time, time = i/freqs, 2pi*freq/freqs*i
  // }
  i = 0;
  while (fscanf(fp, "%lf\n", temp) == 1) {
    in[i] = *temp;
    i++;
    printf("%lf\n", *temp);
  }

  fftw_execute(p); /* repeat as needed */
  fftw_destroy_plan(p);

  for (i = (freqmin*times); i <= (freqmax*times); i++) {
    f = i/times;
    if (abs(out[i][0]) > result) {
      result = f;
    }
    //printf("%d %f %f\n", f, out[i][0], out[i][1]);
  }
  printf("max is %d Hz\n", result);

  fftw_free(in); fftw_free(out);
}
