import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
measles = pd.read_csv('measles.csv')
mumps = pd.read_csv('mumps.csv')
rubella = pd.read_csv('rubella.csv')

import pandas as pd

import numpy as np
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)

print('\n')
print("Head of Complete List")
print(measles.head(5))
print('\n')
print(mumps.head(5))
print('\n')
print(rubella.head(5))
print('\n')

print("Tail of Complete List")
print(measles.tail(5))
print('\n')
print(mumps.tail(5))
print('\n')
print(rubella.tail(5))
print('\n')

print("The columns in this dataframe are:")
print(list(measles))
print(list(mumps))
print(list(rubella))
print('\n')
print("The shape of the dataframe is:", measles.shape)
print("The shape of the dataframe is:", mumps.shape)
print("The shape of the dataframe is:", rubella.shape)
print('\n')
print(measles.dtypes)
print('\n')
print(mumps.dtypes)
print('\n')
print(rubella.dtypes)
print('\n')

max_incidence_measles = measles['incidence_per_capita']
print("The max per capita for measles is:", max_incidence_measles.max())
max_incidence_mumps = mumps['incidence_per_capita']
print("The max per capita for mumps is:", max_incidence_mumps.max())
max_incidence_rubella = rubella['incidence_per_capita']
print("The max per capita for rubella is:", max_incidence_rubella.max())
print('\n')

measles_high = measles.loc[(measles['incidence_per_capita'] > 200)]
print(measles_high)
print('\n')
mumps_high = mumps.loc[(mumps['incidence_per_capita'] > 30)]
print(mumps_high)
print('\n')
rubella_high = rubella.loc[(rubella['incidence_per_capita'] > 50)]
print(rubella_high)

measles_2000 = measles.loc[(measles['week']>200001)]
print('measles_2000')
print(measles_2000)
print('\n')

mumps_2000 = mumps.loc[(mumps['week']>200001)]
print('mumps_2000')
print(mumps_2000)
print('\n')

rubella_2000 = rubella.loc[(rubella['week']>200001)]
print('rubella_2000')
print(rubella_2000)
print('\n')

print('Find the total measles cases per state:')
overall_measles = measles_2000[['state_name', 'cases']]
overall_measles = overall_measles.groupby('state_name').agg({'cases':'sum'}).reset_index()
overall_measles = overall_measles.rename(columns= {'cases': 'sum_cases'})
#overall_measles = overall_measles.round()
print(overall_measles.sort_values(by='sum_cases', ascending=False))
print('\n')

print('Find the total mumps cases per state:')
overall_mumps = mumps_2000[['state_name', 'cases']]
overall_mumps = overall_mumps.groupby('state_name').agg({'cases':'sum'}).reset_index()
overall_mumps = overall_mumps.rename(columns= {'cases': 'sum_cases'})
#overall_mumps = overall_mumps.round()
print(overall_mumps.sort_values(by='sum_cases', ascending=False))
print('\n')

print('Find the total rubella cases per state:')
overall_rubella = rubella_2000[['state_name', 'cases']]
overall_rubella = overall_rubella.groupby('state_name').agg({'cases':'sum'}).reset_index()
overall_rubella = overall_rubella.rename(columns= {'cases': 'sum_cases'})
#overall_rubella = overall_rubella.round()
print(overall_rubella.sort_values(by='sum_cases', ascending=False))
print('\n')