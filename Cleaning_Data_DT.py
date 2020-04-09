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

print("Eliminate schools with no enrollment data")
m = m.loc[(m['enroll'] > 0)]
#what do we do with a school of enrollment 1? Is that valid data??
print('\n')

print('Combine mmr and overall vaccination rates: use mmr if given, overall if no mmr available.')
m['vac_rate'] = m['mmr']
m['vac_rate'] = m['mmr'].where(m['mmr'] > 0, m['overall'] )
print('\n')

print('Eliminate schools with a negative vac_rate')
m = m.loc[(m['vac_rate'] >= 0)]
print('\n')

print(m.head(5))
print(m.tail(5))

m_tree = m[['name', 'state', 'city', 'county', 'type', 'enroll', 'vac_rate']]

print(m_tree.head(5))
print(m_tree.tail(5))


#m_ascending = m.sort_values(by = 'vac_rate')
#print(m_ascending.head(5))
#print(m_ascending.tail(5))

#want to experiement with exemption rates too
#print("Total Exemption Rates")
#m['total_exempt'] = m['xrel'] + m['xmed'] + m['xper']