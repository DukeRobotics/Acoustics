### Acoustic Hydrophone System Repo

## MCCDAQ Linux Driver
- sampling.c for 3 channels
  - sampling.c sampling_time(s) sampling_freq(Hz)
  - sampling speed similar to sampling_4.c
- sampling_4.c for 4 channels
  - sampling_4.c sampling_time(s) sampling_freq(Hz)
  - needs 3.5 second to sample 3 second samples at 125 kS/s
  - needs 1.2 second to sample 1 second samples at 125 kS/s
- modified usb-1608G.c
  - need to replace the original one in mcc-libusb
- both need to be inside /Linux_Drivers/USB/mcc-libusb/ directory and compiled with Makefile inside
  - make
  - sudo make install
- max sampling frequency = 500/#channel kS/s
- differential input mode available
- sampling time can only be integer due to libusb library constraint
- if encounter `Resource temporarily unavailable` error, disconnect usb and reconnect

## Saleae Python Automation Script

## Python Processing Script
