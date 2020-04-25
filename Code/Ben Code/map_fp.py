# ----------------------------------------------------------------------------------------------------------------------
# Package imports and setup
# ----------------------------------------------------------------------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------
# NOTE: must install Geo-pandas in order to run.
# ----------------------------------------------
# File credits:
# https://github.com/gboeing/beer-locations/tree/master/data-analysis/visualization/shapefiles/states_21basic
# https://medium.com/@erikgreenj/mapping-us-states-with-geopandas-made-simple-d7b6e66fa20d
# ----------------------------------------------------------------------------------------------------------------------
# Pre-processing and setup

usa = gpd.read_file('./states/states.shp')

usa = usa.drop(labels=None, axis=0, index=[0, 50])

# must install descartes package to execute .plot() here.
# usa.plot()
# plt.show()

measles = pd.read_csv('imputation_fp.csv', index_col=0)

# print(measles.head(5))

measles = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'overall', 'xrel',
                               'xmed', 'xper'], axis=1)

measles = measles.groupby('state').mean()

print(measles)
# ----------------------------------------------------------------------------------------------------------------------
# Combining usa and measles and constructing the map

usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO', 'MT', 'NJ',
                         'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WI']

measles = measles.rename(columns={'mmr': 'AVG_MMR'})

merged = usa.merge(measles, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

print(combined)

fig, ax = plt.subplots(1, 1)

mapped = combined.plot(column='AVG_MMR', ax=ax, legend=True, cmap='winter')

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean MMR Vaccination Rate\n\nin the United States', loc='Center', pad=15)

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
