# ------------------------------------
DATS 6103-10: Group Four Final Project
Individual Code - Benjamin Lee
# ------------------------------------

# ------------------
Explanation of files    
# ------------------

- imputation_of_missing_values.py

This includes my work on imputation of missing values. The first portion 
of this code finds the state-specific mean of the numerical features 
(enroll, mmr, overall, xrel, xmed, xper) for the existing values, and 
then replaces all missing values with the respective mean. This portion 
takes nearly 20 minutes of runtime due to the large number of missing 
values (~146,000). Upon inspecting the newly-created file 
(measles_imputed_step1.csv), there are still a number of missing values.
This is because certain states contain no data for certain features. 
Thus, the second portion of this file imputes missing values based on 
the mean of all existing values for the given feature. The cleaned 
data frame is then exported as measles_imputed.csv.

- visualization_mapping.py

This file contains my work on visualization using GeoPandas. I would like 
to give credit to Geoff Boeing for providing the necessary state data 
to be able to successfully plot and Erik G. for his article explaining how 
to map US states using GeoPandas (credit below). The framework for the 
geography is read in using states.shp in the states folder. The cleaned 
MMR data set is then read in and merged with the GeoPandas data frame (usa)
using a STATE_ABBR column. The maps were displayed using the different 
numerical features as the targets, and states not included in our data set
were greyed out. The feature that allows NA data to be greyed out 
(missing_kwds={} in the plot command) is only available in the version 0.70 
of geopandas; installation of that version is done using the GeoPandas GitHub
link.

- knn_with_state_as_target.py

This file runs KNN using the state feature as the target. I dropped all other
categorical features, split the data into training and testing sets, and 
stadardized the numerical features. Running KNN here yields a 32x32 confusion 
matrix that indicates the results of the predictions made. Additionally, 
after running cross-validation scores and using GridSearchCV I found that the 
best k-value for our data is k = 3. 

- k_val_test.rmd

This file contains additional research into what value of K is best for our 
data set. The graph produced shows a steady decline in accuracy from k = 1 to 
k = 21. These results, combined with what was found in knn_with_state_as_target.py
using GridSearchCV and cross-validation scores, prove that k = 3 is likely the 
best choice of k.

The imputed_files and mapping_files contain files necessary to run the 
aforementioned files. 

- all-measles-rates.py

This file contains the original, raw data set used for this project.

- measles_imputed_step1.py

This is the file created in the first portion of imputation_of_missing_values.py,
where the missing values are imputed using the mean of all state-specific values 
in the respective feature. 

- measles_imputed.py

This is the file created in the second portion of imputation_of_missing_values.py,
where the missing values are imputed using the mean of all values in the 
respective feature.

- states.shp

This file is required to construct the framework for the mapping visualization in 
visualization_mapping.py. 

- other states files

These files are necessary for reading states.shp into visualization_mapping.py.

- .RData

This file is necessary for running the k_val_test.rmd file.

- .RHistory

This file is necessary for running the k_val_test.rmd file and saving the existing
workspace.

# ----
Credit 
# ----

- Geoff Boeing, States Files
https://github.com/gboeing/beer-locations/tree/master/data-analysis/visualization/shapefiles/states_21basic

- Erik G, Article on Mapping in GeoPandas
https://medium.com/@erikgreenj/mapping-us-states-with-geopandas-made-simple-d7b6e66fa20d