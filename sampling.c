#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>

#include "pmd.h"
#include "usb-1608G.h"

#define MAX_COUNT		(0xffff)
#define FALSE 0
#define TRUE 1


// compile in makefile
// sampling time(s) sampling_freq(Hz)

int main (int argc, char **argv) {
	libusb_device_handle *udev = NULL;

	double voltage;
	double frequency, duty_cycle;
	float temperature;
	float table_AIN[NGAINS_1608G][2];
	float table_AO[NCHAN_AO_1608GX][2];
	ScanList list[NCHAN_1608G];

	int ch;
	int i, j, m, k, nchan, repeats;
	int nread;
	int nScans = 0;
	uint8_t input;
	int temp, ret;
	uint8_t options;
	char serial[9];
	uint32_t period;
	uint16_t version;
	uint16_t status;
	int usb1608GX_2AO = FALSE;
	int flag;
	int transferred;            // number of bytes transferred
	uint16_t value, data;
	uint16_t *sdataIn;          // holds 16 bit unsigned analog input data
	uint16_t sdataOut[512];     // holds 16 bit unsigned analog output data

	uint8_t mode, gain, channel;

	udev = NULL;

	ret = libusb_init(NULL);

	fftw_complex *out;
	double *in;
	fftw_plan p;
	int freqs = atoi(argv[2]);
	//int freq = 6000;
	int times = atof(argv[1]);
	int count = times*freqs;
	int freqmin = 0;
	int freqmax = 60000;
	int result = 0;
	int f;
	int resultf = 0;

	//FILE* fp;
	//fp = fopen("/home/robot/Desktop/Acoustics/data.csv", "w+");

	if (ret < 0) {
		perror("fail to initialize libsusb");
		exit(1);
	}

	if ((udev = usb_device_find_USB_MCC(USB1608G_V2_PID, NULL))) {
		printf("found USB 1608G\n");
		usbInit_1608G(udev, 2);
	}
	else if ((udev = usb_device_find_USB_MCC(USB1608G_PID, NULL))) {
		printf("found USB 1608G\n");
		usbInit_1608G(udev, 1);
	}
	else {
		printf("fail to find the device\n");
		return 0;
	}

	usbBuildGainTable_USB1608G(udev, table_AIN);

	printf("start sampling\n");
	usbAInScanStop_USB1608G(udev);
	usbAInScanClearFIFO_USB1608G(udev);
	mode = DIFFERENTIAL;
	gain = BP_10V;
	nchan = 3;
	nScans = count;
	frequency = freqs;


	for (channel = 0; channel < nchan; channel++) {
		list[channel].range = gain;
		list[channel].mode = mode;
		list[channel].channel = channel;
	}

	list[nchan-1].mode |= LAST_CHANNEL;

	usbAInConfig_USB1608G(udev, list);

	if ((sdataIn = malloc(2*nchan*nScans)) == NULL) {
		perror("cannot allocate memory for sdataIn");
		return;
	}

	// in = (double*) malloc(sizeof(double) * count);
	// out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * count);
	// p = fftw_plan_dft_r2c_1d(count, in, out, FFTW_ESTIMATE);

// usbAInScanStart_USB1608G(device, nScans, trigger count keep 0, frequency, option keep 0x0);
	usbAInScanStart_USB1608G(udev, nScans, 0, frequency, 0x0);
	// usbAInScanRead_USB1608G(device, nScans, nchan, sdataIn buffer, timeout in millisecond (0 if continuous), option keep 0);
	ret = usbAInScanRead_USB1608G(udev, nScans, nchan, sdataIn, times*1000+1000, 0);
	for (i = 0; i < nScans; i++) {
		//printf("%6d", i);
		for (j = 0; j < nchan; j++) {
			gain = list[j].range;
			k = i*nchan + j;
			data = rint(sdataIn[k]*table_AIN[gain][0] + table_AIN[gain][1]);

			printf("%8.4lf", volts_USB1608G(gain, data));
			printf(",");
			//fprintf(fp, "%8.4lf\n", volts_USB1608G(gain, data));
			// in[i] = volts_USB1608G(gain, data);
		}
		printf("\n");
	}
	free(sdataIn);
	//fclose(fp);

	// fftw_execute(p); /* repeat as needed */
	// fftw_destroy_plan(p);
  //
	// for (i = (freqmin*times); i <= (freqmax*times); i++) {
  //   f = i/times;
  //   if (abs(out[i][0]*out[i][0]+out[i][1]*out[i][1]) > result) {
  //     resultf = f;
  //     result = abs(out[i][0]*out[i][0]+out[i][1]*out[i][1]);
  //   }
  //   printf("%d %f %f\n", f, out[i][0], out[i][1]);
  // }
  // printf("max is %d Hz\n", resultf);
  //
	// free(in); fftw_free(out);

}
