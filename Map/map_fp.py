import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.impute import SimpleImputer
# ---------------------------------------------
# NOTE: must install Geo-pandas in order to run.
# ---------------------------------------------
# File credits:
# https://github.com/gboeing/beer-locations/tree/master/data-analysis/visualization/shapefiles/states_21basic
# https://medium.com/@erikgreenj/mapping-us-states-with-geopandas-made-simple-d7b6e66fa20d

usa = gpd.read_file('./states/states.shp')

# must install descartes package to execute .plot() here.
# usa.plot()
# plt.show()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
world['gdp_per_cap'] = world.gdp_md_est / world.pop_est

# print(world.dtypes)
# print(world.head(10))

# world.plot(column='gdp_per_cap')
# plt.show()

measles = pd.read_csv('all-measles-rates.csv')

measles = measles.drop(labels='district', axis=1)

# print(measles['mmr'].value_counts())
# 18,087 '-1' values for mmr

# ----------------------------------------------------------------
# Imputing for MMR feature and computing the mean MMR values
impute = SimpleImputer(missing_values=-1, strategy='mean')

impute.fit(measles[['mmr']])

SimpleImputer()

measles[['mmr']] = impute.transform(measles[['mmr']])

# print(measles.head(5))

# print(measles.isna().sum())

measles = measles.drop(labels=['index', 'year', 'name', 'type', 'city', 'county', 'enroll', 'overall', 'xrel',
                               'xmed', 'xper'], axis=1)

measles = measles.groupby('state').mean()

# print(measles)
# ----------------------------------------------------------------
# Combining usa and measles

usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO', 'MT', 'NJ',
                         'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI']

measles = measles.rename(columns={'mmr': 'AVG_MMR'})

# measles['STATE_NAME'] = measles['STATE_NAME'].astype(str)
# usa['STATE_NAME'] = usa['STATE_NAME'].astype(str)

merged = usa.merge(measles, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

print(combined)

fig, ax = plt.subplots(1, 1)

combined.plot(column='AVG_MMR', ax=ax, legend=True, cmap='winter')
plt.show()
