# Instruction for Hydrophone Data Access


Current hydrophone database covers data recorded in 2019 summer at Transdec (RoboSub competition facility) and in 2019-2020 academic year at Wilson pool (mainly the diving pool)

Database is stored in Box folder, `hydrophone_data`, and a hard drive at lab. Please contact Muthu or Samuel to request access to the Box folder.

----
## Nomenclature
- Currently all dataset were taken with a square array configuration (under `/square_array` folder). For future datasets taken with different hydrophone array configuration, please create new directories and put datasets under corresponding folders.
  - In the square array configuration, the 1st hydrophone (h0) would be at the origin; the 2nd hydrophone (h1) would be on the negative y-axis; the 3rd hydrophone (h2) would be on the negative x-axis; the 4th hydrophone (h3) would be in the third quadrant with negative x and y coordinates
- `\transdec` directory contains dataset from Transdec pool; `wilson_pool` directory contains dataset from Wilson diving pool (for less reflection off the wall)
- Datasets with file name starting with `1s` or `3s` are timing datasets taken to study the time interval between pings. They are likely to contain no direction information as the mount/robot might be unstable during recording.
  - `1s` and `3s` represent the time length of dataset
  - Sampling rate and target frequency information might be contained in the txt file under the same directory. If not, try 625k for sampling rate and 30k/40k for target frequency.
  - The serial numbers in the file number of timing datasets are system time written for identification.
- Datasets with file name in the format of `xxk_xxk_x_x_x(x)` are directional datasets taken for sound source detection.
  - 1st term represents the sampling rate
  - 2nd term represents the target frequency
  - 3rd, 4th and 5th terms represent the relative direction of sound source to hydrophone array.
    - It follows the format of `x_y_z`, relative to coordinate system origin; h0 is the origin; h1 is on negative y-axis; h2 is on negative x-axis, h3 is in third quadrant.
    - For z-direction, negative axis points down
    - All coordinates information marked in file name were estimated, therefore the accuracy can only be guaranteed to be +-22.5 degree (1/8 pi)
    - Some coordinate information might be associated with units when recorded(inch or meter). However, since there isn't a guaranteed uniform unit, we consider all coordinate information to be unitless, providing only directional information.
  - Last term inside parentheses represents the version number

----
## "Good" Directional Dataset
- Since there are a lot of factors that determine whether a directional dataset is "good", i.e. containing accurate direction information, we are not sure whether some of the directional datasets in this database are "good" enough. They might be either taken when the robot was not stable or the mount was not held horizontal.
- When using a directional dataset to test the direction detection script, one should first use the "good" dataset marked below and compare the output direction with the rough direction indicated by the file name. These datasets were tested and gave correct direction (not accurate since the recorded direction was a rough estimation) after being processed
  - `hydrophone_data/square_array/transdec/july_30/625k_30k_1_1_10(7).csv`
  - `hydrophone_data/square_array/transdec/july_30/625k_30k_1_-1_10(1).csv`
  - `hydrophone_data/square_array/transdec/july_30/625k_30k_1_-1_10(2).csv`
  - `hydrophone_data/square_array/transdec/july_30/625k_30k_1_-1_10(3).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_4_-5_-4(1).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_4_-5_-4(2).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_4_-5_-4(3).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_4_-5_-4(4).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_5_-2_-4(1).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_5_-2_-4(2).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_5_-2_-4(3).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_5_-2_-4(4).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_3_-3_-4(1).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_3_-3_-4(2).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_3_-3_-4(3).csv`
  - `hydrophone_data/square_array/wilson_pool/wilson_range_dive/625k_40k_3_-3_-4(4).csv`
- For directional datasets that are not mentioned above, feel free to process them, as some of them haven't been examined yet.
