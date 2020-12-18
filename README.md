### Acoustic Hydrophone System Repo

## MCCDAQ Linux Driver
- sampling.c for 3 channels
  - sampling.c sampling_time(s) sampling_freq(Hz)
  - sampling speed similar to sampling_4.c
  - python2
- sampling_4.c for 4 channels
  - sampling_4.c sampling_time(s) sampling_freq(Hz)
  - needs 3.5 second to sample 3 second samples at 125 kS/s
  - needs 1.2 second to sample 1 second samples at 125 kS/s
  - python2
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
- saleae_sampling.py for 4 channels
  - have saleae software running before running the script
  - python3
  - if getting `saleae software down, open saleae software before next script run`, open another terminal to run saleae software
  - need 14 second to sample 3 second samples at 1250 kS/s
  - need 6 second to sample 1 second samples at 1250 kS/s
  - need 10 second to sample 3 second samples at 625 kS/s
  - need 4 second to sample 1 second samples at 625 kS/s
  - need 6 second to sample 3 second samples at 125 kS/s
  - need 3 second to sample 1 second samples at 125 kS/s
  - export path need to be absolute path
  - the export csv data would be slightly longer then the set sampling time, but the sampling rate is accurate
  - sampling_rate: 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s

## Python Processing Script
