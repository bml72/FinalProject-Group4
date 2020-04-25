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
measles = pd.read_csv('imputation_fp.csv', index_col=0)

# Mean calculations commented out for the purpose of evading slow runtime; results copied to equivalent data frame

# measles_means = measles.mean()

# measles_means = measles_means.round(decimals=2)

# measles_means = measles_means.drop(labels=['enroll'], axis=1)

# measles_means = measles_means.reset_index()

# print(measles_means)

mean_values = [['enroll', 120.54], ['mmr', 94.35], ['overall', 92.06], ['xrel', 4.33], ['xmed', 2.57], ['xper', 7.04]]

measles_means = pd.DataFrame(data=mean_values, columns=['feature', 'value'])

print(measles_means, '\n', 20*'-')
# ----------------------------------------------------------------------------------------------------------------------
# Imputing based off mean of feature
# ----------------------------------------------------------------------------------------------------------------------


def imp_val(data, feature, mean_value):
    data[feature] = data[feature].fillna(value=mean_value)


imp_val(measles, 'enroll', measles_means['value'][0])
imp_val(measles, 'mmr', measles_means['value'][1])
imp_val(measles, 'overall', measles_means['value'][2])
imp_val(measles, 'xrel', measles_means['value'][3])
imp_val(measles, 'xmed', measles_means['value'][4])
imp_val(measles, 'xper', measles_means['value'][5])

print(measles.isna().sum())
# ----------------------------------------------------------------------------------------------------------------------
# Exporting to CSV file for use in modeling
# ----------------------------------------------------------------------------------------------------------------------
measles = measles.to_csv('impute_final.csv')
