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

measles = pd.read_csv('all-measles-rates.csv')

measles = measles.drop(labels=['index', 'year', 'district'], axis=1)

measles = measles.replace(-1, np.NaN)

measles_means = measles.groupby('state').mean()

measles_means = measles_means.round(decimals=2)

measles_means = measles_means.drop(labels=['enroll'], axis=1)

measles_means = measles_means.reset_index()

print(measles_means)
# ----------------------------------------------------------------------------------------------------------------------
# Imputation of numerical features


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


imp_mmr()
imp_overall()
imp_xmed()
imp_xper()
imp_xrel()


measles = measles.to_csv('imputation_fp.csv')
# ----------------------------------------------------------------------------------------------------------------------
# Alternate methods practiced, with no success

# for x in range(measles_means.shape[0]):
#    measles['mmr'] = measles['mmr'].replace(to_replace=np.NaN, value=measles_means['mmr'][x]).where(
#    measles['state'] == measles_means['state'][x])

# measles['mmr'] = measles['mmr'].replace(np.NaN, measles_means['mmr'][0]).where(
# measles['state'] == measles_means['state'][0])

# measles['overall'] = measles['overall'].apply(lambda x: 10000 if
# pd.isnull(x) and measles['state'] == 'Arizona' else x)

# measles['mmr'] = measles['mmr'].replace(np.NaN, measles_means['mmr'][1]).where(measles['state'] == 'Arkansas')

# measles['mmr'] = measles.loc[measles.state == 'Arizona', 'mmr' == np.NaN] = measles_means['mmr'][0]
