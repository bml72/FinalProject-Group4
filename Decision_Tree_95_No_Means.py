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

#rename type column
m = m.rename(columns={'type':'type_of_school'})
#enter school type variables as binary
ts_dict = {'Public':1, 'Charter':2, 'Private':3, 'Kindergarten':4}
m = m.replace({'type_of_school':ts_dict})

#fill type_of_school NaN values with zero
m['type_of_school'] = m['type_of_school'].fillna(0)

#make sure all the additions worked
print(m.head(10))
print('\n')
print(m.tail(10))
print('\n')

#select columsn to use for DT
m_tree = m[['state', 'city', 'county', 'type_of_school', 'enroll', 'at_least_95']]

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
m_tree['state']=le.fit_transform(m_tree['state'])
m_tree['city']=le.fit_transform(m_tree['city'])
m_tree['county']=le.fit_transform(m_tree['county'])
m_tree['type_of_school']=le.fit_transform(m_tree['type_of_school'])
m_tree['enroll']=le.fit_transform(m_tree['enroll'])
m_tree['at_least_95']=le.fit_transform(m_tree['at_least_95'])

print(m_tree.head(5))
print('\n')
print(m_tree.tail(5))
print('\n')

print('Decision Tree Algorithm')
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
from sklearn.tree import DecisionTreeClassifier

# Data Collection
X = m_tree.values[:, 0:5]
Y = m_tree.values[:, 5]

data_feature_names = ['state', 'city', 'county', 'type', 'enroll']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

#graph.write_png('m_tree.png')
#graph.write_svg('m_tree.svg')

# split the dataset
# separate the target variable
X = m_tree.values[:, 0:5]
y = m_tree.values[:, 5]

# encloding the class with sklearn's LabelEncoder
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = m_tree.at_least_95.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = m_tree.at_least_95.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()
