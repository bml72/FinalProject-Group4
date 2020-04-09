import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
m = pd.read_csv('all-measles-rates.csv')

import pandas as pd

import numpy as np
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)

print('\n')
print("Head of Complete List")
print(m.head(5))
print('\n')
print("You can't have an mmr vaccination rate of 100%, but an overall vaccination rate of -1.")
print("I'm thinking -1 is being used as a placeholder when the data wasn't available.")
print("For example, some states reported mmr AND overall rates.  And others just reported one or the other.")
print("I'm thinking we should clean the data, using the mmr rate when available, and the overall rate when mmr is not available.")
print("That appears to be what the WSJ did -- I looked up certain schools on their website.")
print("Also, is enrollment in thousands? Need to figure out.")
print('\n')

print("Tail of Complete List")
print(m.tail(5))
print('\n')
print('This seems to be an example of schools for which they have no data -- see enrollment as NaN.')
print("Do we want to use the techniques learned in class to estimate vaccination rates at these schools?")
print('\n')

print("The columns in this dataframe are:")
print(list(m))
print('\n')
print("The shape of the dataframe is:", m.shape)
print('\n')
print(m.dtypes)
print('\n')

mmr_numbers = m['mmr']
print("The max percentage of mmr vaccinations is:", mmr_numbers.max())
print("The min percentage of mmr vaccinations is:", mmr_numbers.min())
print("This does not make sense.  You can't have a -1% of a school vaccinated.")
print('\n')

print("Notes from Kristin:")
print('For herd immunity -- for highly contagious disease like measles, 90-95% vaccination is required.')
print('For less contagious diseses (like polio) 80-85% vaccinated would be enough')
print('anti-vaccine families tend to cluster together')
print('\n')

print("Sort by mmr rate:")
print("Schools with mmr rate >= 95%:")
mmr_high = m.loc[(m['mmr'] >= 95) & (m['enroll'] >= 1 )]
mmr_high = mmr_high.sort_values(by='mmr', ascending=False)
print("The shape of the dataframe is:", mmr_high.shape)
print('\n')
print("Head >= 95% List")
print(mmr_high.head(5))
print('\n')
print("Tail >= 95% List")
print(mmr_high.tail(5))
print('\n')

print("Schools with lowest mmr rates:")
mmr_low = m.loc[(m['mmr'] <= 50)  & (m['enroll'] >= 1)]
mmr_low = mmr_low.sort_values(by='mmr', ascending=False)
print("The shape of the dataframe is:", mmr_low.shape)
print('\n')
print("Head <= 50% List")
print(mmr_low.head(5))
print('\n')
print("Tail <= 50% List")
print(mmr_low.tail(5))
print('Here you can see the mmr/orverall problem.  The mmr rate for these records is -1, but the overall rate it approx 92%')
print("We'll have to figure out how to handle this.")
print('\n')

print('Find the mean mmr vaccination rate per state:')
#get rid of -1 values
mmr_no_zero = m.loc[(m['mmr'] >= 0)]
mmr_no_zero = mmr_no_zero[['state', 'mmr']]
mmr_mean = mmr_no_zero.groupby('state').agg({'mmr':'mean'}).reset_index()
mmr_mean = mmr_mean.rename(columns= {'mmr': 'mean_mmr_rate'})
mmr_mean = mmr_mean.round()
print(mmr_mean.sort_values(by='mean_mmr_rate', ascending=False))
print('\n')

print('Find the mean overall vaccination rate per state:')
#get rid of -1 values
overall_no_zero = m.loc[(m['overall'] >= 0)]
overall_no_zero = overall_no_zero[['state', 'overall']]
overall_mean = overall_no_zero.groupby('state').agg({'overall':'mean'}).reset_index()
overall_mean = overall_mean.rename(columns= {'overall': 'mean_overall_rate'})
overall_mean = overall_mean.round()
print(overall_mean.sort_values(by='mean_overall_rate', ascending=False))
print('\n')

types_of_schools = m['type']
print(types_of_schools.unique())

m_tree = m[['name', 'enroll', 'state', 'type', 'city', 'mmr', 'overall']]
m_tree = m_tree.loc[(m_tree['enroll'] >=0)]
print(m_tree.head(10))
print(m_tree.tail(10))

m_tree['vac_rate'] = m_tree['mmr']


print(m_tree.head(10))
print(m_tree.tail(10))


        #print('Note added by my daughter: Kara is cool.')