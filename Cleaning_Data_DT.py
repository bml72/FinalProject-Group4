#m_ascending = m.sort_values(by = 'vac_rate')
#print(m_ascending.head(5))
#print(m_ascending.tail(5))

#want to experiement with exemption rates too
#print("Total Exemption Rates")
#m['total_exempt'] = m['xrel'] + m['xmed'] + m['xper']

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

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

print('Combine mmr and overall vaccination rates: use mmr if given, overall if no mmr available.')
m['vac_rate'] = m['mmr']
m['vac_rate'] = m['mmr'].where(m['mmr'] > 0, m['overall'] )


print('Eliminate schools with a negative vac_rate')
m = m.loc[(m['vac_rate'] >= 0)]

print("Is vaccination rate at least 95%? ")
m['at_least_95'] = (m['vac_rate'] >= 95)
print('\n')

print('Find the mean vac_rate per state:')


print(m.head(5))
print(m.tail(5))

m_tree = m[['state', 'city', 'county', 'type', 'enroll', 'at_least_95']]

print(m_tree.head(5))
print('\n')
print(m_tree.tail(5))
print('\n')


