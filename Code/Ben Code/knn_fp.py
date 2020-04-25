# ----------------------------------------------------------------------------------------------------------------------
# Importing necessary packages
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Data Pre-processing
# ----------------------------------------------------------------------------------------------------------------------
# Changing the output settings to read more features.
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

measles = pd.read_csv('impute_final.csv', index_col=0)

measlesKNN = measles.copy(deep=True)

# Dropping the irrelevant categorical features; will initially test KNN using only state
measlesKNN = measlesKNN.drop(labels=['name', 'type', 'city', 'county'], axis=1)

measlesKNN['state'] = measles['state'].astype('category')

print("Data set:\n", measlesKNN, '\n', 20*'-')
# ----------------------------------------------------------------------------------------------------------------------
# K-Nearest Neighbors Algorithm Pre-processing
# ----------------------------------------------------------------------------------------------------------------------
# Setting the feature and target variables. Will change the below to include the one-hot encoded categorical data
le = LabelEncoder()

state_encoded = le.fit_transform(measlesKNN['state'])

X = measlesKNN.values[:, 1:]

Y = state_encoded

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100, stratify=Y)

standardize = StandardScaler()

standardize.fit(X_train)

X_train_standardized = standardize.transform(X_train)
X_test_standardized = standardize.transform(X_test)
# ----------------------------------------------------------------------------------------------------------------------
# Running K-Nearest Neighbors
# ----------------------------------------------------------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train_standardized, Y_train)

Y_pred = model.predict(X_test_standardized)
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Metrics and Constructing Confusion Matrix
# ----------------------------------------------------------------------------------------------------------------------
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
print("\n")

print("Accuracy Score:", accuracy_score(Y_test, Y_pred)*100)
print("\n")

cm = confusion_matrix(Y_test, Y_pred)
class_names = measlesKNN['state'].unique()

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 5}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=5)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=5)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.tight_layout()
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------------
print("Summary\n")
print("Upon running KNN with a variety of different values for K, the following was revealed:")

data = [[1, '88.62%'], [2, '87.33%'], [3, '87.44%'], [4, '86.98%'], [5, '86.74%'], [6, '86.53%'], [7, '86.28%'],
        [8, '86.03%'], [9, '85.73%'], [10, '85.66%']]

accuracy_results = pd.DataFrame(data=data, columns=['K-value', 'Accuracy Score'])

print(accuracy_results)

print("For now, I will stick with K = 3 as the default choice for K.")
# ----------------------------------------------------------------------------------------------------------------------
