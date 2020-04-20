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

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

m = pd.read_csv('all-measles-rates.csv')
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)
print('\n')

#Eliminate schools with no enrollment data
m = m.loc[(m['enroll'] > 0)]
#what do we do with a school of enrollment 1? Is that valid data??

#Combine mmr and overall vaccination rates: use mmr if given, overall if no mmr available.
m['vac_rate'] = m['mmr']
m['vac_rate'] = m['mmr'].where(m['mmr'] > 0, m['overall'] )

#Eliminate schoosl with a vegative vac_rate
m = m.loc[(m['vac_rate'] >= 0)]

#Is the vac_rate >= to 95 percent?
m['at_least_95'] = (m['vac_rate'] >= 95)

#find the mean vac_rate per state
state_mean = m[['state', 'vac_rate']]
state_mean = m.groupby('state').agg({'vac_rate':'mean'}).reset_index()
state_mean = state_mean.rename(columns= {'vac_rate': 'state_mean'})
state_mean = state_mean.round(decimals=1)

#add state_mean to original df
# first need to convert state_mean df to a dictionary, then can use replace() function
sm_dict = dict(zip(state_mean['state'], state_mean['state_mean']))
m['state_mean'] = m['state']
m = m.replace({'state_mean': sm_dict})

#find the mean vac_rate per city
city_mean = m[['city', 'vac_rate']]
city_mean = m.groupby('city').agg({'vac_rate':'mean'}).reset_index()
city_mean = city_mean.rename(columns= {'vac_rate': 'city_mean'})
city_mean = city_mean.round(decimals=1)

#add city_mean to original df
cm_dict = dict(zip(city_mean['city'], city_mean['city_mean']))
m['city_mean'] = m['city']
m = m.replace({'city_mean': cm_dict})

#fill city NaN values with zero
m['city_mean'] = m['city_mean'].fillna(0)

#find the mean vac_rate per county
county_mean = m[['county', 'vac_rate']]
county_mean = m.groupby('county').agg({'vac_rate':'mean'}).reset_index()
county_mean = county_mean.rename(columns= {'vac_rate': 'county_mean'})
county_mean = county_mean.round(decimals=1)

#add county_mean to original df
cym_dict = dict(zip(county_mean['county'], county_mean['county_mean']))
m['county_mean'] = m['county']
m = m.replace({'county_mean': cym_dict})

#fill county NaN values with zero
m['county_mean'] = m['county_mean'].fillna(0)

#rename type column
m = m.rename(columns={'type':'type_of_school'})
#enter school type variables as binary
ts_dict = {'Public':1, 'Charter':2, 'Private':3, 'Kindergarten':4}
m = m.replace({'type_of_school':ts_dict})

#fill type_of_school NaN values with zero
m['type_of_school'] = m['type_of_school'].fillna(0)

#fill exemption values with zero
m['xrel'] = m['xrel'].fillna(0)
m['xmed'] = m['xmed'].fillna(0)
m['xper'] = m['xper'].fillna(0)

#add all the different types of exemptions(religious, medical, and personal) together
m['xtotal'] = m['xrel'] + m['xmed'] + m['xper']

#make sure all the additions worked
print(m.head(10))
print('\n')
print(m.tail(10))
print('\n')

#select columsn to use for DT
m_tree = m[['state_mean', 'city_mean', 'county_mean', 'type_of_school', 'enroll', 'xtotal', 'at_least_95']]

print(m_tree.head(5))
print('\n')
print(m_tree.tail(5))
print('\n')

#check to see if df has any NaN values
print(m_tree.isnull().sum())
print(m_tree.dtypes)

#enter variables as binary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
m_tree['state_mean']=le.fit_transform(m_tree['state_mean'])
m_tree['city_mean']=le.fit_transform(m_tree['city_mean'])
m_tree['county_mean']=le.fit_transform(m_tree['county_mean'])
m_tree['type_of_school']=le.fit_transform(m_tree['type_of_school'])
m_tree['enroll']=le.fit_transform(m_tree['enroll'])
m_tree['xtotal']=le.fit_transform(m_tree['xtotal'])
m_tree['at_least_95']=le.fit_transform(m_tree['at_least_95'])

print(m_tree.head(5))
print('\n')
print(m_tree.tail(5))
print('\n')

#KNN Code
# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# Deepak Agarwal------>Email:deepakagarwal@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 June - 13 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% K-Nearest Neighbor  %%%%%%%%%%%%%%%%%%%%%
#%%-----------------------------------------------------------------------
# Importing the required packages

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%%-----------------------------------------------------------------------
#importing Dataset

# define column names
col_names = ['state_mean', 'city_mean', 'county_mean', 'type_of_school', 'enroll', 'xtotal', 'at_least_95']

# read data as panda dataframe
#wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, names=col_names)

# printing the dataset shape
print("Dataset No. of Rows: ", m_tree.shape[0])
print("Dataset No. of Columns: ", m_tree.shape[1])

# printing the dataset observations
print("Dataset first few rows:\n ")
print(m_tree.head(2))

# printing the struture of the dataset
print("Dataset info:\n ")
print(m_tree.info())

# printing the summary statistics of the dataset
print(m_tree.describe(include='all'))
#%%-----------------------------------------------------------------------
# split the dataset
# separate the target variable
X = m_tree.values[:, 0:6]
Y = m_tree.values[:, 6]


#%%-----------------------------------------------------------------------
# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

#%%-----------------------------------------------------------------------
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#%%-----------------------------------------------------------------------
# data preprocessing
# standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%-----------------------------------------------------------------------
# perform training
# creating the classifier object
clf = KNeighborsClassifier(n_neighbors=3)

# performing training
clf.fit(X_train_std, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred = clf.predict(X_test_std)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#%%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = m_tree['at_least_95'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()