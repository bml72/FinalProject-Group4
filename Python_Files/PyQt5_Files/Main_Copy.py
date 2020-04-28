##################################################
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
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class RandomForest95(QMainWindow):
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
        super(RandomForest95, self).__init__()
        self.Title = "Random Forest MMR >= 95"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.btnExecute,5,0)

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

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG3,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        #::-------------------------------------------------
        #Random Forest Classifier
        #We populate the dashboard using the parametres chosen by the user
        #::-------------------------------------------------

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt = self.list_corr_features
        y_dt = mmr['at_least_95']

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = ['','under_95', 'at_least_95']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

class RandomForest90(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier MM >= 90
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest90, self).__init__()
        self.Title = "Random Forest MMR >= 90"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.btnExecute,5,0)

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

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG3,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        #::-------------------------------------------------
        #Random Forest Classifier
        #We populate the dashboard using the parametres chosen by the user
        #::-------------------------------------------------

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt = self.list_corr_features
        y_dt = mmr['at_least_90']

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = ['','under_90', 'at_least_90']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

class DecisionTree95(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree95, self).__init__()

        self.Title ="Decision Tree Classifier MMR >= 95"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Decision Tree Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,5,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)
        self.groupBox1Layout.addWidget(self.btnDTFigure,6,1)

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

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        ## End of elements of the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):

        #::-------------------------------------------------
        #Decision Tree Algorithm
        #We populate the dashboard using the parametres chosen by the user
        #::-------------------------------------------------

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','under_95', 'at_least_95']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_corr_features.columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')

class DecisionTree90(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm >= 90
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree90, self).__init__()

        self.Title ="Decision Tree Classifier MMR >= 90"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Decision Tree Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,5,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)
        self.groupBox1Layout.addWidget(self.btnDTFigure,6,1)

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

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        ## End of elements of the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):

        #::-------------------------------------------------
        #Decision Tree Algorithm
        #We populate the dashboard using the parametres chosen by the user
        #::-------------------------------------------------

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = mmr["at_least_90"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','under_90', 'at_least_90']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_corr_features.columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')


class KNN95(QMainWindow):
    #::----------------------
    # Implementation of KNN Algorithm
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN95, self).__init__()

        self.Title ="KNN Classifier MMR >= 95"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML KNN Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('N Neighbors :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,5,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)

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

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        ## End of elements on the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        #::--------------------------------------------
        #KNN Algorithm
        #We populate the dashboard using the parametres chosen by the user
        #::--------------------------------------------

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # fit and transform the class
        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100, stratify=y_dt)

        # data preprocessing
        # standardize the data
        stdsc = StandardScaler()

        stdsc.fit(X_train)

        X_train_std = stdsc.transform(X_train)
        X_test_std = stdsc.transform(X_test)

        # perform training
        # creating the classifier object
        self.clf = KNeighborsClassifier(n_neighbors=vmax_depth)

        #performing training
        self.clf.fit(X_train_std, y_train)

        # make predictions
        # predicton on test
        y_pred_KNN = self.clf.predict(X_test_std)

        # confusion matrix for KNN model
        conf_matrix = confusion_matrix(y_test, y_pred_KNN)

        # clasification report
        self.ff_class_rep = classification_report(y_test, y_pred_KNN)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_KNN) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','under_95', 'at_least_95']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

class KNN90(QMainWindow):
    #::----------------------
    # Implementation of KNN Algorithm MMR >= 90
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN90, self).__init__()

        self.Title ="KNN Classifier MMR >= 90"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML KNN Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)


        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('N Neighbors :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)

        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,5,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)

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

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        ## End of elements on the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        #::--------------------------------------------
        #KNN Algorithm
        #We populate the dashboard using the parametres chosen by the user
        #::--------------------------------------------

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = mmr[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = mmr[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, mmr[features_list[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = mmr["at_least_90"]

        class_le = LabelEncoder()

        # fit and transform the class
        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100, stratify=y_dt)

        # data preprocessing
        # standardize the data
        stdsc = StandardScaler()

        stdsc.fit(X_train)

        X_train_std = stdsc.transform(X_train)
        X_test_std = stdsc.transform(X_test)

        # perform training
        # creating the classifier object
        self.clf = KNeighborsClassifier(n_neighbors=vmax_depth)

        #performing training
        self.clf.fit(X_train_std, y_train)

        # make predictions
        # predicton on test
        y_pred_KNN = self.clf.predict(X_test_std)

        # confusion matrix for KNN model
        conf_matrix = confusion_matrix(y_test, y_pred_KNN)

        # clasification report
        self.ff_class_rep = classification_report(y_test, y_pred_KNN)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_KNN) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','under_90', 'at_least_90']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

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
        super(Regression, self).__init__()
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


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()
        
        self.btnExecute = QPushButton("Execute Regression")
        self.btnExecute.clicked.connect(self.update)
        
        self.groupBox1Layout.addWidget(self.state0,0,0)
        self.groupBox1Layout.addWidget(self.state1,0,1)
        self.groupBox1Layout.addWidget(self.state2,1,0)
        self.groupBox1Layout.addWidget(self.state3,1,1)
        self.groupBox1Layout.addWidget(self.state4,2,0)
        self.groupBox1Layout.addWidget(self.state5,2,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)
        
        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        
        
        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)

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


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'MMR Vaccination Rates'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        MLModelsMenu = mainMenu.addMenu('ML Models')
        RegMenu = mainMenu.addMenu('Regression')


        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('../../Data-Mining/Demo/PyQt5/Demo/enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        DT95Button =  QAction(QIcon(), 'Decision Tree 95', self)
        DT95Button.setStatusTip('Decision Tree 95')
        DT95Button.triggered.connect(self.MLDT95)

        DT90Button = QAction(QIcon(), 'Decision Tree 90', self)
        DT90Button.setStatusTip('Decision Tree 90')
        DT90Button.triggered.connect(self.MLDT90)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        RF95Button = QAction(QIcon(), 'Random Forest 95', self)
        RF95Button.setStatusTip('Random Forest 95')
        RF95Button.triggered.connect(self.MLRF95)

        RF90Button = QAction(QIcon(), 'Random Forest 90', self)
        RF90Button.setStatusTip('Random Forest 90')
        RF90Button.triggered.connect(self.MLRF90)

        #::------------------------------------------------------
        # KNN Model
        #::------------------------------------------------------

        KNN95Button = QAction(QIcon(), 'KNN 95', self)
        KNN95Button.setStatusTip('KNN 95')
        KNN95Button.triggered.connect(self.MLKNN95)

        KNN90Button = QAction(QIcon(), 'KNN 90', self)
        KNN90Button.setStatusTip('KNN 90')
        KNN90Button.triggered.connect(self.MLKNN90)

        MLModelsMenu.addAction(DT95Button)
        MLModelsMenu.addAction(DT90Button)

        MLModelsMenu.addAction(RF95Button)
        MLModelsMenu.addAction(RF90Button)

        MLModelsMenu.addAction(KNN95Button)
        MLModelsMenu.addAction(KNN90Button)

        self.dialogs = list()

        #::------------------------------------------------------
        # Regression
        #::------------------------------------------------------
        Reg1Button = QAction(QIcon(), 'Regression', self)
        Reg1Button.setStatusTip('Regression')
        Reg1Button.triggered.connect(self.RegressRussell)

        RegMenu.addAction(Reg1Button)


    def MLDT95(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree95()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT90(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree90()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF95(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = RandomForest95()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF90(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = RandomForest90()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNN95(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a KNN Algorithm
        #::-----------------------------------------------------------
        dialog = KNN95()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNN90(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a KNN Algorithm
        #::-----------------------------------------------------------
        dialog = KNN90()
        self.dialogs.append(dialog)
        dialog.show()

    def RegressRussell(self):
        dialog = Regression()
        self.dialogs.append(dialog)
        dialog.show()


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_mmr():
    #::--------------------------------------------------
    # Loads the mmr.csv
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global mmr
    global features_list
    global class_names
    global state_names
    global data

    data = pd.read_csv('all-measles-rates.csv', sep=",", error_bad_lines=False)
    data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county',
                    'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
    mmr = pd.read_csv('m_tree.csv')
    features_list = ["state_mean", "city_mean", "county_mean", "type_of_school",
         "enroll", "xtotal"]
    class_names = ['under_95', 'at_least_95']
    state_names = ['Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Florida', 'Idaho',
            'Illinois', 'Iowa', 'Maine', 'Massachusetts', 'Michigan', 'Minnesota',
            'Missouri', 'Montana', 'New Jersey', 'New York', 'North Carolina',
            'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
            'Rhode Island', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
            'Virginia', 'Washington', 'Wisconsin']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_mmr()
    main()