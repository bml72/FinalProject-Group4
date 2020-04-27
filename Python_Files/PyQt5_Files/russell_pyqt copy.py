
### Created by Ben, Russell, and Kristin
### Project Name : MMR Vaccination Rates
### Date 04/28/2030
### Data Mining
##################################################

import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt


from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

#KNN libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import random
import seaborn as sns



class Regression(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.Title = "Linear Regresion"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Linear Regression ')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each State
        self.state0 = QCheckBox(state_names[0],self)
        self.state1 = QCheckBox(state_names[1],self)
        self.state2 = QCheckBox(state_names[2],self)
        self.state3 = QCheckBox(state_names[3],self)
        self.state4 = QCheckBox(state_names[4],self)
        self.state5 = QCheckBox(state_names[5],self)
        self.state6 = QCheckBox(state_names[6],self)
        self.state7 = QCheckBox(state_names[7],self)
        self.state8 = QCheckBox(state_names[8],self)
        self.state9 = QCheckBox(state_names[9],self)
        self.state10 = QCheckBox(state_names[10],self)
        self.state11 = QCheckBox(state_names[11],self)
        self.state12 = QCheckBox(state_names[12],self)
        self.state13 = QCheckBox(state_names[13],self)
        self.state14 = QCheckBox(state_names[14],self)
        self.state15 = QCheckBox(state_names[15],self)
        self.state16 = QCheckBox(state_names[16],self)
        self.state17 = QCheckBox(state_names[17],self)
        self.state18 = QCheckBox(state_names[18],self)
        self.state19 = QCheckBox(state_names[19],self)
        self.state20 = QCheckBox(state_names[20],self)
        self.state21 = QCheckBox(state_names[21],self)
        self.state22 = QCheckBox(state_names[22],self)
        self.state23 = QCheckBox(state_names[23],self)
        self.state24 = QCheckBox(state_names[24],self)
        self.state25 = QCheckBox(state_names[25],self)
        self.state26 = QCheckBox(state_names[26],self)
        self.state27 = QCheckBox(state_names[27],self)
        self.state28 = QCheckBox(state_names[28],self)
        self.state29 = QCheckBox(state_names[29],self)
        self.state30 = QCheckBox(state_names[30],self)

        self.state0.setChecked(True)
        self.state1.setChecked(True)
        self.state2.setChecked(True)
        self.state3.setChecked(True)
        self.state4.setChecked(True)
        self.state5.setChecked(True)
        self.state6.setChecked(True)
        self.state7.setChecked(True)
        self.state8.setChecked(True)
        self.state9.setChecked(True)
        self.state10.setChecked(True)
        self.state11.setChecked(True)
        self.state12.setChecked(True)
        self.state13.setChecked(True)
        self.state14.setChecked(True)
        self.state15.setChecked(True)
        self.state16.setChecked(True)
        self.state17.setChecked(True)
        self.state18.setChecked(True)
        self.state19.setChecked(True)
        self.state20.setChecked(True)
        self.state21.setChecked(True)
        self.state22.setChecked(True)
        self.state23.setChecked(True)
        self.state24.setChecked(True)
        self.state25.setChecked(True)
        self.state26.setChecked(True)
        self.state27.setChecked(True)
        self.state28.setChecked(True)
        self.state29.setChecked(True)
        self.state30.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute Regression")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG3,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        #::-------------------------------------------------
        #Linear Regression Classifier
        #We populate the dashboard using the parametres chosen by the user
        #::-------------------------------------------------

        # processing the parameters

        self.data = mmr
        if self.state0.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state0]

        self.data = mmr
        if self.state1.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state1]

        self.data = mmr
        if self.state2.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state2]

        self.data = mmr
        if self.state3.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state3]

        self.data = mmr
        if self.state4.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state4]

        self.data = mmr
        if self.state5.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state5]

        self.data = mmr
        if self.state6.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state6]

        self.data = mmr
        if self.state7.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state7]

        self.data = mmr
        if self.state8.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state8]

        self.data = mmr
        if self.state9.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state9]

        self.data = mmr
        if self.state10.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state10]

        self.data = mmr
        if self.state11.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state11]

        self.data = mmr
        if self.state12.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state12]

        self.data = mmr
        if self.state12.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state13]

        self.data = mmr
        if self.state14.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state14]

        self.data = mmr
        if self.state15.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state15]

        self.data = mmr
        if self.state16.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state16]

        self.data = mmr
        if self.state17.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state17]

        self.data = mmr
        if self.state18.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state18]

        self.data = mmr
        if self.state19.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state19]

        self.data = mmr
        if self.state20.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state20]

        self.data = mmr
        if self.state21.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state21]

        self.data = mmr
        if self.state22.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state22]

        self.data = mmr
        if self.state23.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state23]

        self.data = mmr
        if self.state24.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state24]

        self.data = mmr
        if self.state25.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state25]

        self.data = mmr
        if self.state26.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state26]

        self.data = mmr
        if self.state27.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state27]

        self.data = mmr
        if self.state28.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state28]

        self.data = mmr
        if self.state29.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state29]

        self.data = mmr
        if self.state30.isChecked():
            if len(self.data)==0:
                self.data = mmr
            else:
                self.data = data[data["state"] == self.state30]



        #Impute means into missing values

        enroll_mean = data['enroll'].mean(axis=0)
        data['enroll'].fillna(enroll_mean, inplace=True)
        xrel_mean = data['xrel'].mean(axis=0)
        data['xrel'].fillna(xrel_mean, inplace=True)
        xmed_mean = data['xmed'].mean(axis=0)
        data['xmed'].fillna(xmed_mean, inplace=True)
        xper_mean = data['xper'].mean(axis=0)
        data['xper'].fillna(xper_mean, inplace=True)


        # Assign the X and y to run the Linear Regression Classifier

        x = self.data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]
        y = data['mmr']


        # split the dataset into train and test

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

        # fit linear regression classifier

        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred = mlr.predict(x_test)

        #Print coefficients and metrics
        coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])
        coeff_df
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

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





def data_mmr():
    #::--------------------------------------------------
    # Loads the mmr.csv
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global mmr
    global features_list
    global class_names
    global state_names
    mmr = pd.read_csv('m_tree.csv')
    features_list = ["state_mean", "city_mean", "county_mean", "type_of_school",
         "enroll", "xtotal"]
    class_names = ['under_95', 'at_least_95']
    state_names = ['Arizona', 'Arkansas', 'Colorado', 'Connecticut', 'Florida', 'Idaho',
            'Illinois', 'Iowa', 'Maine', 'Massachusetts', 'Michigan', 'Minnesota',
            'Missouri', 'Montana', 'New Jersey', 'New York', 'North Carolina'
            'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania'
            'Rhode Island', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
            'Virginia', 'Washington', 'Wisconsin']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_mmr()
    main()


