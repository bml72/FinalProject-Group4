import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing


def regression(state):
	data = pd.read_csv('all-measles-rates.csv', sep=",", error_bad_lines=False)
	data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county', 
	'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']

	print('Description of Data')
	print(data.describe())
	print()

	print('Description of Categorical Data')
	print(data.describe(include = ['O']))
	print()

	#Check missing values
	print('Missing Values:')
	print(data.isnull().sum())
	print()

	#Impute means into missing values
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

	#Subset data by state
	data = data[data["state"] == state]

	#Scatter plots of variables of interest
	plt.scatter(data['enroll'], data['mmr'])
	plt.title('MMR Vs Enroll')
	plt.xlabel('Enrollment')
	plt.ylabel('MMR Rate')
	plt.show()

	plt.scatter(data['overall'], data['mmr'])
	plt.title('MMR Vs Overall')
	plt.xlabel('Overall Rate')
	plt.ylabel('MMR Rate')
	plt.show()

	plt.scatter(data['xrel'], data['mmr'])
	plt.title('MMR Vs Xrel')
	plt.xlabel('Religious Exemptions')
	plt.ylabel('MMR Rate')
	plt.show()

	plt.scatter(data['xmed'], data['mmr'])
	plt.title('MMR Vs Xmed')
	plt.xlabel('Medical Exemptions')
	plt.ylabel('MMR Rate')
	plt.show()

	plt.scatter(data['xper'], data['mmr'])
	plt.title('MMR Vs Xper')
	plt.xlabel('Personal Exemptions')
	plt.ylabel('MMR Rate')
	plt.show()

	#Define feature and target data
	data.loc[data.mmr == -1, 'mmr'] = 0
	data.loc[data.overall == -1, 'overall'] = 0
	y = data['mmr']
	x = data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]
	

	#Test train split and linear regression
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
	print('Shape of datasets')
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	print()
	mlr = LinearRegression()
	mlr.fit(x_train, y_train)
	y_pred = mlr.predict(x_test)

	#Print coefficients and metrics
	print()
	coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])
	print(coeff_df)
	print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
	print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	print()

	#View actual vs predicted and model score
	df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	print(df)
	print('The score of the model: ', round(mlr.score(x_test, y_test), 3))
	print()
	
	
	#Normalize data 
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	x = pd.DataFrame(x_scaled)
	y = data['mmr'].values
	y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
	y = pd.DataFrame(y_scaled)
	
	#Test train split and linear regression
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
	mlr = LinearRegression()
	mlr.fit(x_train, y_train)
	y_pred = mlr.predict(x_test)
	
	#Print metrics 
	print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
	print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
	print('The score of the model: ', round(mlr.score(x_test, y_test), 3))




if __name__=="__main__":regression("Oregon")




