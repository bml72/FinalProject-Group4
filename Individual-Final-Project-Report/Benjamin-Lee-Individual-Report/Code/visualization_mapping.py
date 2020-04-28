# ----------------------------------------------------------------------------------------------------------------------
# Installing packages if not already installed
# ----------------------------------------------------------------------------------------------------------------------
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install('git+git://github.com/geopandas/geopandas.git')
install('pandas')
install('matplotlib')
# ----------------------------------------------------------------------------------------------------------------------
# Import files and packages
# ----------------------------------------------------------------------------------------------------------------------
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Visualization Construction
# ----------------------------------------------------------------------------------------------------------------------
# Reading in a pre-built file that includes Geopandas geometry all all 50 states in the US. File credits:
# https://github.com/gboeing/beer-locations/tree/master/data-analysis/visualization/shapefiles/states_21basic
# https://medium.com/@erikgreenj/mapping-us-states-with-geopandas-made-simple-d7b6e66fa20d
measles = pd.read_csv('./imputed_files/measles_imputed.csv', index_col=0)

usa = gpd.read_file('./mapping_files/states.shp')

usa = usa.drop(labels=None, axis=0, index=[0, 50])
# ----------------------------------------------------------------------------------------------------------------------
# Enrollment Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for MMR by state, all other features are dropped
measles_means = measles.groupby('state').mean()

print("Mean values by state\n", measles_means, "\n", 58*"-")

measles_enrollment = measles.drop(labels=['name', 'type', 'city', 'county', 'mmr', 'overall', 'xrel',
                               'xmed', 'xper'], axis=1)

measles_map_means = measles_enrollment.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'enroll': 'AVG_ENROLL'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_ENROLL', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean Enrollment\n\nin the United States', loc='Center', pad=15)
plt.show()

print("Measles Enrollment values\n", measles_map_means, "\n", 58*"-")
# ----------------------------------------------------------------------------------------------------------------------
# MMR Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for MMR by state, all other features are dropped
measles_means = measles.groupby('state').mean()

print("Mean values by state\n", measles_means, "\n", 58*"-")

measles_mmr = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'overall', 'xrel',
                               'xmed', 'xper'], axis=1)

measles_map_means = measles_mmr.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'mmr': 'AVG_MMR'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_MMR', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean MMR Vaccination Rate\n\nin the United States', loc='Center', pad=15)
plt.show()

print("Measles MMR values\n", measles_map_means, "\n", 58*"-")
# ----------------------------------------------------------------------------------------------------------------------
# Overall Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for overall by state, all other features are dropped
measles_overall = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'mmr', 'xrel',
                               'xmed', 'xper'], axis=1)

measles_map_means = measles_overall.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'overall': 'AVG_OVERALL'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_OVERALL', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean Overall Vaccination \n\nRate in the United States', loc='Center', pad=15)
plt.show()

print("Measles overall values\n", measles_map_means, "\n", 58*"-")
# ----------------------------------------------------------------------------------------------------------------------
# XREL Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for xrel by state, all other features are dropped
measles_xrel = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'mmr', 'overall',
                                    'xmed', 'xper'], axis=1)

measles_map_means = measles_xrel.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'xrel': 'AVG_XREL'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_XREL', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean xrel Rate\n\nin the United States', loc='Center', pad=15)
plt.show()

print("Measles xrel values\n", measles_map_means, "\n", 58*"-")
# ----------------------------------------------------------------------------------------------------------------------
# XMED Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for xmed by state, all other features are dropped
measles_xmed = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'mmr', 'overall',
                                    'xrel', 'xper'], axis=1)

measles_map_means = measles_xmed.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'xmed': 'AVG_XMED'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_XMED', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean xmed Rate\n\nin the United States', loc='Center', pad=15)
plt.show()

print("Measles xmed values\n", measles_map_means, "\n", 58*"-")
# ----------------------------------------------------------------------------------------------------------------------
# XPER Map
# ----------------------------------------------------------------------------------------------------------------------
# Because this visualization is only for xmed by state, all other features are dropped
measles_xper = measles.drop(labels=['name', 'type', 'city', 'county', 'enroll', 'mmr', 'overall',
                                    'xrel', 'xmed'], axis=1)

measles_map_means = measles_xper.groupby('state').mean()

# Combining usa and measles and constructing the map
usa = usa.sort_values(by='STATE_NAME', ascending=True)
usa = usa.reset_index()
usa = usa.drop(labels='index', axis=1)

measles_map_means['STATE_ABBR'] = ['AZ', 'AR', 'CA', 'CO', 'CT', 'FL', 'ID', 'IL', 'IA', 'ME', 'MA', 'MI', 'MN', 'MO',
                                   'MT', 'NJ', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SD', 'TN', 'TX', 'UT',
                                   'VT', 'VA', 'WA', 'WI']

measles_map_means = measles_map_means.rename(columns={'xper': 'AVG_XPER'})

merged = usa.merge(measles_map_means, on='STATE_ABBR', how='left')

combined = gpd.GeoDataFrame(merged)

mapped = combined.plot(column='AVG_XPER', legend=True, cmap='winter', missing_kwds={'color': 'lightgrey'})

frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

title = plt.title(label='Distribution of Mean xper Rate\n\nin the United States', loc='Center', pad=15)
plt.show()

print("Measles xper values\n", measles_map_means, "\n", 58*"-")
