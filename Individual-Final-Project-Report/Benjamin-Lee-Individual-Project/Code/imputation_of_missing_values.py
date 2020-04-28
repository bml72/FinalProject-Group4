# ----------------------------------------------------------------------------------------------------------------------
# Installing packages if not already installed
# ----------------------------------------------------------------------------------------------------------------------
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install('pandas')
install('numpy')
# ----------------------------------------------------------------------------------------------------------------------
# Package imports and setup
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
# ----------------------------------------------------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------------------------------------------------
# Reading in the data as 'measles'
measles = pd.read_csv('./imputed_files/all-measles-rates.csv')

# Dropping the irrelevant features and replacing placeholder NaN values (-1's) with NaN
measles = measles.drop(labels=['index', 'year', 'district'], axis=1)
measles = measles.replace(-1, np.NaN)

# Grouping mean of numerical features by state to reference for imputation
measles_means = measles.groupby('state').mean()
measles_means = measles_means.round(decimals=2)
measles_means = measles_means.reset_index()


def imp_enroll():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['enroll']):
                    row['enroll'] = measles_means['enroll'][x]
                    measles.loc[i, 'enroll'] = row['enroll']
    '''
    The purpose of imp_enroll() and the five other imputation functions below is to iterate through each row of the 
    data set (state-by-state) and check whether the value of the respective feature is missing. If the value is 
    missing, it is replaced by the mean value for the respective state. 
    '''


def imp_mmr():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['mmr']):
                    row['mmr'] = measles_means['mmr'][x]
                    measles.loc[i, 'mmr'] = row['mmr']


def imp_overall():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['overall']):
                    row['overall'] = measles_means['overall'][x]
                    measles.loc[i, 'overall'] = row['overall']


def imp_xrel():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['xrel']):
                    row['xrel'] = measles_means['xrel'][x]
                    measles.loc[i, 'xrel'] = row['xrel']


def imp_xmed():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['xmed']):
                    row['xmed'] = measles_means['xmed'][x]
                    measles.loc[i, 'xmed'] = row['xmed']


def imp_xper():
    for x in range(measles_means.shape[0]):
        for i, row in measles.iterrows():
            if row['state'] == measles_means['state'][x]:
                if pd.isnull(row['xper']):
                    row['xper'] = measles_means['xper'][x]
                    measles.loc[i, 'xper'] = row['xper']


imp_enroll()
imp_mmr()
imp_overall()
imp_xmed()
imp_xper()
imp_xrel()

measles = measles.to_csv('./imputed_files/measles_imputed_step1.csv')

measles_imputed_step1 = pd.read_csv('./imputed_files/measles_imputed_step1.csv', index_col=0)

measles_imputed_step1_means = measles_imputed_step1.mean()
measles_imputed_step1_means = measles_imputed_step1_means.round(decimals=2)
measles_imputed_step1_means = measles_imputed_step1_means.reset_index()

# Manually creating a data frame using the values computed in measles_imputed_step1_means
mean_values = [['enroll', 119.07], ['mmr', 94.35], ['overall', 92.06], ['xrel', 4.33], ['xmed', 2.57], ['xper', 7.04]]

measles_imputed_step1_means_df = pd.DataFrame(data=mean_values, columns=['feature', 'value'])

# Here, I am replacing missing values with the mean the entire column for each given feature. As explained in the
# report, there was no easy way to impute missing values state-by-state because certain states had no data for certain
# features.


def imp_val(data, feature, mean_value):
    data[feature] = data[feature].fillna(value=mean_value)


imp_val(measles, 'enroll', measles_imputed_step1_means_df['value'][0])
imp_val(measles, 'mmr', measles_imputed_step1_means_df['value'][1])
imp_val(measles, 'overall', measles_imputed_step1_means_df['value'][2])
imp_val(measles, 'xrel', measles_imputed_step1_means_df['value'][3])
imp_val(measles, 'xmed', measles_imputed_step1_means_df['value'][4])
imp_val(measles, 'xper', measles_imputed_step1_means_df['value'][5])

measles = measles.to_csv('./imputed_files/measles_imputed.csv')
