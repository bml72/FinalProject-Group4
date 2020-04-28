# -------------------------------------------------------
DATS 6103-10: Group Four Final Project
GUI Code - Benjamin Lee, Kristin Levine, Russell Moncrief
# -------------------------------------------------------

As per final project instructions, the only file that 
must be run to generate our GUI and compute/generate our 
models and visualizations is DATS_6103_G4_FP_GUI.py. 
This file runs linearly from top to bottom and requires
no other .py files to run. However, it must call several
.csv and .png files to run successfully. A summary of 
those files is below. 

The imputed_files folder contains all necessary .csv 
files to run. 

- all-measles-rates.csv

This file contains the raw, unedited data necessary for
running regression. 

- m_tree.csv

This file contains the pre-processed data necessary for 
running decision tree and random forest. 

- measles_imputed.csv

This file contains the imputed values necessary for 
running KNN using the state feature as the target. 
It contains no missing values.

The visualization_data folder contains all necessary
.pdf files to run. All files in here are the respective
visualizations of the mean distribution of the numerical
features by state in the US (enrollment, MMR vaccination 
rate, overall vaccination rate, and percentage of people 
exempt from vaccination for medical, personal and 
religious reasons. These images were generated using 
the original visualization file (visualization_mapping.py)
and cropped for easy display within the GUI file. 
