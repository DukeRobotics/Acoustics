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
// sampling_4 time(s) sampling_freq(Hz)

int main (int argc, char **argv) {
	libusb_device_handle *udev = NULL;

	double frequency;
	float table_AIN[NGAINS_1608G][2];
	ScanList list[NCHAN_1608G];

	int i, j, k, nchan;
	int nScans = 0;
	int ret;
	uint16_t data;
	uint16_t *sdataIn;          // holds 16 bit unsigned analog input data

	uint8_t mode, gain, channel;

	udev = NULL;

	ret = libusb_init(NULL);

	int freqs = atoi(argv[2]);
	int times = atof(argv[1]);
	int count = times*freqs;

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
	nchan = 4;
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
		return 0;
	}

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
		}
		printf("\n");
	}
	free(sdataIn);
	return 0;

}
