import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read data
data = pd.read_csv('all-measles-rates.csv', sep=",", error_bad_lines=False)
data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county', 'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
print('Description of Data')
print(data.describe())
print()

#Check missing values
print('Missing Values:')
print(data.isnull().sum())
print()

#Drop missing values for year
#print('Dropped Missing years')
#data = data[data['year'].notna()]
#print(data.isnull().sum())
#print()

#Fill missing values with means
enroll_mean = data['enroll'].mean(axis=0)
data['enroll'].fillna(enroll_mean, inplace=True)
xrel_mean = data['xrel'].mean(axis=0)
data['xrel'].fillna(xrel_mean, inplace=True)
xmed_mean = data['xmed'].mean(axis=0)
data['xmed'].fillna(xmed_mean, inplace=True)
xper_mean = data['xper'].mean(axis=0)
data['xper'].fillna(xper_mean, inplace=True)
print('Missing Values:')
print(data.isnull().sum())
print()

#Drop remaining missing values
#data = data[data['type'].notna()]
#data = data[data['city'].notna()]
#data = data[data['county'].notna()]
#data = data.drop('district', 1)
#print('Cleaned Dataset')
#print(data.isnull().sum())
#print()
#print('Cleaned Data Description')
#print(data.describe())
#print(data)
#print()

#Graphs
colors = ['red', 'tan', 'lime']
plt.hist(data['enroll'], bins=100, color='green')
plt.title('Distribution of Enrollment')
plt.show()
plt.hist(data['mmr'], bins=10, color='yellow')
plt.title('Distribution of MMR Rate')
plt.show()
plt.hist(data['overall'], bins=10, color='blue')
plt.title('Distribution of Overall Rate')
plt.show()
plt.hist(data['xrel'], bins=10, color='red')
plt.title('Distribution of Xrel Rate')
plt.show()
plt.hist(data['xmed'], bins=10, color='orange')
plt.title('Distribution of Xmed Rate')
plt.show()
plt.hist(data['xper'], bins=10, color='lime')
plt.title('Distribution of Xper Rate')
plt.show()

sns.boxplot(x = "state", y = "mmr" , data = data, palette="PRGn")
plt.xticks(rotation=90)
plt.show()

sns.boxplot(x = "type", y = "mmr" , data = data, palette="PRGn")
plt.xticks(rotation=90)
plt.show()








