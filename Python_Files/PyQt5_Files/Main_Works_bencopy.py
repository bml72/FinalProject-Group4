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

from PyQt5.QtGui import QIcon, QPixmap
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


class KnnStateTarget(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KnnStateTarget, self).__init__()

        self.Title = "KNN Using State as the Target"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        # Model Features tab
        self.groupBox1 = QGroupBox('Model Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature_0 = QCheckBox(features[0], self)
        self.feature_1 = QCheckBox(features[1], self)
        self.feature_2 = QCheckBox(features[2], self)
        self.feature_3 = QCheckBox(features[3], self)
        self.feature_4 = QCheckBox(features[4], self)
        self.feature_5 = QCheckBox(features[5], self)

        self.feature_0.setChecked(True)
        self.feature_1.setChecked(True)
        self.feature_2.setChecked(True)
        self.feature_3.setChecked(True)
        self.feature_4.setChecked(True)
        self.feature_5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage of Model Displayed:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel("N Neighbors")
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Run KNN")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature_0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature_1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature_2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature_3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature_4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature_5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)

        # Results From Execution tab
        self.groupBox2 = QGroupBox("Results from Execution")
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

        # Confusion Matrix tab
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox("Confusion Matrix")
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        # End of elements on dashboard
        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        self.list_corr_features = pd.DataFrame([])
        if self.feature_0.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[0]]], axis=1)

        if self.feature_1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[1]]], axis=1)

        if self.feature_2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[2]]], axis=1)

        if self.feature_3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[3]]], axis=1)

        if self.feature_4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[4]]], axis=1)

        if self.feature_5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = measles[features[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, measles[features[5]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        X = self.list_corr_features
        Y = measles['state']

        le = LabelEncoder()

        Y = le.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=vtest_per, random_state=100, stratify=Y)

        standardize = StandardScaler()

        standardize.fit(X_train)

        X_train_standardized = standardize.transform(X_train)
        X_test_standardized = standardize.transform(X_test)

        self.model = KNeighborsClassifier(n_neighbors=vmax_depth)

        self.model.fit(X_train_standardized, Y_train)

        Y_pred = self.model.predict(X_test_standardized)

        # Confusion Matrix
        conf_mat = confusion_matrix(Y_test, Y_pred)

        # Classification Report
        self.class_report = classification_report(Y_test, Y_pred)
        self.txtResults.appendPlainText(self.class_report)

        # Accuracy Score
        self.accur_score = accuracy_score(Y_test, Y_pred)*100
        self.txtAccuracy.setText(str(self.accur_score))

        # Confusion Matrix
        class_names2 = measles['state'].unique()

        self.ax1.matshow(conf_mat, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(classes, fontsize=10)
        self.ax1.set_xticklabels(classes, rotation=90, fontsize=10)
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        for i in range(len(classes)):
            for j in range(len(classes)):
                self.ax1.text(j, i, str(conf_mat[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class Visualization(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Visualization, self).__init__()

        self.Title = "Feature Visualization"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Enroll", "MMR", "Overall", "xRel", "xMed", "xPer"])

        self.dropdown1.currentIndexChanged.connect(self.update)
        self.label = QLabel("Variables")

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Index for plots"))
        self.layout.addWidget(self.dropdown1)

        self.setCentralWidget(self.main_widget)
        self.show()
        self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        cat1 = self.dropdown1.currentText()
        label = QLabel(self)
        if cat1 == "Enroll":
            pixmap = QPixmap("enroll.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()
        elif cat1 == "MMR":
            pixmap = QPixmap("mmr.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()
        elif cat1 == "Overall":
            pixmap = QPixmap("overall.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()
        elif cat1 == "xRel":
            pixmap = QPixmap("xrel.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()
        elif cat1 == "xMed":
            pixmap = QPixmap("xmed.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()
        else:
            pixmap = QPixmap("xper.png")
            label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.show()


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
        self.initui()

    def initui(self):
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
        VisMenu = mainMenu.addMenu('Visualization')
        MLModelsMenu = mainMenu.addMenu('ML Models')


        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        KNN90Button = QAction(QIcon(), 'KNN 90', self)
        KNN90Button.setStatusTip('KNN 90')
        KNN90Button.triggered.connect(self.MLKNN90)

        MLModelsMenu.addAction(KNN90Button)

        KnnStateTargetButton = QAction(QIcon(), 'KNN State Target', self)
        KnnStateTargetButton.setStatusTip('KNN using state as the target variable')
        KnnStateTargetButton.triggered.connect(self.MLKNNState)

        MLModelsMenu.addAction(KnnStateTargetButton)

        VisButton = QAction(QIcon('analysis.png'), 'Feature Visualization', self)
        VisButton.setStatusTip('Visualization of Numerical Features')
        VisButton.triggered.connect(self.Vis)

        VisMenu.addAction(VisButton)

        self.dialogs = list()

    def MLKNN90(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a KNN Algorithm
        #::-----------------------------------------------------------
        dialog = KNN90()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNNState(self):
        dialog = KnnStateTarget()
        self.dialogs.append(dialog)
        dialog.show()

    def Vis(self):
        dialog = Visualization()
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
    mmr = pd.read_csv('m_tree.csv')
    features_list = ["state_mean", "city_mean", "county_mean", "type_of_school",
         "enroll", "xtotal"]
    class_names = ['under_95', 'at_least_95']


def data_mmr2():
    global measles
    global features
    global classes
    measles = pd.read_csv('measles_imputed.csv')
    measles['state'] = measles['state'].astype('category')
    features = ['enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
    classes = measles['state'].unique()


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_mmr()
    data_mmr2()
    main()
