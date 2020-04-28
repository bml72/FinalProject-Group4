# ----------------------------------------------------------------------------------------------------------------------
# DATS 6103-10 Final Project: Group Four
# Benjamin Lee, Kristin Levine, Russell Moncrief
# April 28, 2020
# ----------------------------------------------------------------------------------------------------------------------
# Checking for and installing necessary packages
# ----------------------------------------------------------------------------------------------------------------------
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("pyqt5")
install("pandas")
install("matplotlib")
install("sklearn")
# ----------------------------------------------------------------------------------------------------------------------
# Importing necessary packages
# ----------------------------------------------------------------------------------------------------------------------
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QLabel, QGridLayout, QCheckBox,
                             QGroupBox, QVBoxLayout, QLineEdit, QPlainTextEdit, QSizePolicy, QMessageBox)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin/"

# ----------------------------------------------------------------------------------------------------------------------
# Data pre-processing
# ----------------------------------------------------------------------------------------------------------------------
font_size_window = "font-size:15px"

# NOTE: Pre-processing was done separately by all three group members. Particular to the pre-processing for
# visualizations and running KNN with the state feature as the target (Benjamin Lee), the cleaning and imputation of
# the nearly 146,000 missing values takes nearly 20 minutes of runtime. If you would like to see the pre-processing
# code, please open the individual files on the GitHub repository. For the sake of ease and in the essence of time, the
# pre-processed files are read in using the defined functions at the bottom of this file.
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run decision tree where the MMR value is greater than or equal to 95%
# ----------------------------------------------------------------------------------------------------------------------


class DecisionTree95(QMainWindow):
    # ----------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas based on the parameters chosen by the user
    # view_tree : shows the tree in a pdf form
    # ----------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree95, self).__init__()

        self.Title = "Decision Tree Classifier MMR >= 95"
        self.init_ui()

    def init_ui(self):
        # --------------------------------------------------------------
        # Creating the canvas and all the elements to create a dashboard
        # --------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox("ML Decision Tree Features")
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel("Maximum Depth:")
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # ---------------------------------------
        # Checkboxes are created for each feature
        # ---------------------------------------
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)
        self.groupBox1Layout.addWidget(self.btnDTFigure, 6, 1)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        # --------------------------------
        # End of elements of the dashboard
        # --------------------------------
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # Decision Tree Algorithm
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())
        v_max_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per/100

        # ------------------------------------------------
        # Assigning values to X and y to run the algorithm
        # ------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # ----------------------------------
        # Fitting and transforming the class
        # ----------------------------------
        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=v_max_depth,
                                                  min_samples_leaf=5)

        # -------------------
        # Performing training
        # -------------------
        self.clf_entropy.fit(x_train, y_train)

        # --------------------------------
        # Prediction on test using entropy
        # --------------------------------
        y_pred_entropy = self.clf_entropy.predict(x_test)

        # ----------------------------------
        # Confusion matrix for entropy model
        # ----------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # ---------------------
        # Classification report
        # ---------------------
        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy)*100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        class_names1 = ["", "under_95", "at_least_95"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def view_tree(self):
        # -----------------------------------------------------------
        # Executes Graphviz to create a tree view of the information,
        # then presents the graphic in a PDF format using WebBrowser
        # -----------------------------------------------------------
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_corr_features.columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r"decision_tree_entropy.pdf")
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run decision tree where the MMR value is greater than or equal to 90%
# ----------------------------------------------------------------------------------------------------------------------


class DecisionTree90(QMainWindow):
    # ----------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas based on the parameters chosen by the user
    # view_tree : shows the tree in a pdf form
    # ----------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree90, self).__init__()

        self.Title ="Decision Tree Classifier MMR >= 90"
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------------------
        # Creating the canvas and all elements to create a dashboard
        # ----------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox("ML Decision Tree Features")
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel("Maximum Depth:")
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # ------------------------------------
        # Creating a checkbox for each feature
        # ------------------------------------
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)
        self.groupBox1Layout.addWidget(self.btnDTFigure, 6, 1)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        # --------------------------------
        # End of elements of the dashboard
        # --------------------------------
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # Decision Tree Algorithm
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())
        v_max_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per / 100

        # ------------------------------------------------
        # Assigning values to X and y to run the algorithm
        # ------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_90"]

        class_le = LabelEncoder()

        # ---------------------------
        # Fit and transform the class
        # ---------------------------
        y_dt = class_le.fit_transform(y_dt)

        # -------------------------------------
        # Split the dataset into train and test
        # -------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100)

        # --------------------------------
        # Decision tree with entropy
        # --------------------------------
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=v_max_depth,
                                                  min_samples_leaf=5)

        # -------------------
        # Performing training
        # -------------------
        self.clf_entropy.fit(x_train, y_train)

        # --------------------------------
        # Prediction on test using entropy
        # --------------------------------
        y_pred_entropy = self.clf_entropy.predict(x_test)

        # ----------------------------------
        # Confusion matrix for entropy model
        # ----------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # ---------------------
        # Classification report
        # ---------------------
        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        class_names1 = ["", "under_90", "at_least_90"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def view_tree(self):
        # -----------------------------------------------------------
        # Executes Graphviz to create a tree view of the information,
        # then presents the graphic in PDF format using WebBrowser
        # -----------------------------------------------------------

        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_corr_features.columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r"decision_tree_entropy.pdf")
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run random forest where the MMR value is greater than or equal to 95%
# ----------------------------------------------------------------------------------------------------------------------


class RandomForest95(QMainWindow):
    # ----------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas based on the parameters chosen by the user
    # ----------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest95, self).__init__()
        self.Title = "Random Forest MMR >= 95"
        self.init_ui()

    def init_ui(self):
        # ------------------------------------------------------------
        # Creates the canvas and all the element to create a dashboard
        # ------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # --------------------------------------------
        # Creating a checkbox for each of the features
        # --------------------------------------------
        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        # ---------------------------------
        # Graphic 2: Importance of Features
        #::--------------------------------
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox("Importance of Features")
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG3, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # Random Forest Classifier
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])

        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())

        # ---------------------------------------------------------
        # Clearing the graphs to populate them with new information
        # ---------------------------------------------------------
        self.ax1.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per / 100

        # --------------------------------------------------
        # Assign X and y to run the Random Forest Classifier
        # --------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # ---------------------------
        # Fit and transform the class
        # ---------------------------
        y_dt = class_le.fit_transform(y_dt)

        # -----------------------------------------
        # Splitting the dataset into train and test
        # -----------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100)

        # --------------------------------
        # Specify random forest classifier
        # --------------------------------
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # ----------------
        # Perform training
        # ----------------
        self.clf_rf.fit(x_train, y_train)

        # -------------------------------------
        # Prediction on test using all features
        # -------------------------------------
        y_pred = self.clf_rf.predict(x_test)
        y_pred_score = self.clf_rf.predict_proba(x_test)

        # ---------------------------------
        # Confusion matrix for RandomForest
        # ---------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred)

        # ---------------------
        # Classification report
        # ---------------------
        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred)*100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        class_names1 = ["", "under_95", "at_least_95"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # -----------------------------
        # Graphic 2: Feature Importance
        # -----------------------------
        # Get feature importance
        # -----------------------------
        importance = self.clf_rf.feature_importances_

        # -----------------------------------------------------
        # Convert the importance into one-dimensional 1-d array
        # with corresponding df column names as axis labels
        # -----------------------------------------------------
        f_importance = pd.Series(importance, self.list_corr_features.columns)

        # ----------------------------------------------------
        # Sort the array in descending order of the importance
        # ----------------------------------------------------
        f_importance.sort_values(ascending=False, inplace=True)

        x_features = f_importance.index
        y_importance = list(f_importance)

        self.ax3.barh(x_features, y_importance)
        self.ax3.set_aspect("auto")

        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run random forest where the MMR value is greater than or equal to 90%
# ----------------------------------------------------------------------------------------------------------------------


class RandomForest90(QMainWindow):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest90, self).__init__()
        self.Title = "Random Forest MMR >= 90"
        self.init_ui()

    def init_ui(self):
        # ------------------------------------------------------------
        # Create the canvas and all the elements to create a dashboard
        # ------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout = QGridLayout()  # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 5, 0)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox("Confusion Matrix")
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        # ---------------------------------
        # Graphic 2: Importance of Features
        # ---------------------------------
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

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG3, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # Random Forest Classifier
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])

        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())

        # ------------------------------------------------------
        # Clear the graphs to populate them with new information
        # ------------------------------------------------------
        self.ax1.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per / 100

        # --------------------------------------------------
        # Assign X and y to run the Random Forest Classifier
        # --------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_90"]

        class_le = LabelEncoder()

        # ---------------------------
        # Fit and transform the class
        # ---------------------------
        y_dt = class_le.fit_transform(y_dt)

        # -------------------------------------
        # Split the dataset into train and test
        # -------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100)

        # --------------------------------
        # Specify random forest classifier
        # --------------------------------
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # ----------------
        # Perform training
        # ----------------
        self.clf_rf.fit(x_train, y_train)

        # -------------------------------------
        # Prediction on test using all features
        # -------------------------------------
        y_pred = self.clf_rf.predict(x_test)
        y_pred_score = self.clf_rf.predict_proba(x_test)

        # ---------------------------------
        # Confusion matrix for RandomForest
        # ---------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred)

        # --------------------
        # Classification report
        # --------------------
        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred)*100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        class_names1 = ["", "under_90", "at_least_90"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # -----------------------------
        # Graphic 2: Feature Importance
        # -----------------------------
        # Get feature importance
        # -----------------------------
        importance = self.clf_rf.feature_importances_

        # -----------------------------------------------------
        # Convert the importance into one-dimensional 1-d array
        # with corresponding df column names as axis labels
        # -----------------------------------------------------
        f_importance = pd.Series(importance, self.list_corr_features.columns)

        # ----------------------------------------------------
        # Sort the array in descending order of the importance
        # ----------------------------------------------------
        f_importance.sort_values(ascending=False, inplace=True)

        x_features = f_importance.index
        y_importance = list(f_importance)

        self.ax3.barh(x_features, y_importance)
        self.ax3.set_aspect("auto")

        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run KNN where the MMR value is greater than or equal to 95%
# ----------------------------------------------------------------------------------------------------------------------


class KNN95(QMainWindow):
    # ------------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters as chosen by the user
    # view_tree : shows the tree in a pdf form
    # ------------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN95, self).__init__()

        self.Title = "KNN Classifier MMR >= 95"
        self.init_ui()

    def init_ui(self):
        # ------------------------------------------------------------
        # Create the canvas and all the elements to create a dashboard
        # ------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox("ML KNN Features")
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel("N Neighbors:")
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # KNN Algorithm
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])

        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())
        v_max_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per / 100

        # ------------------------------------------------
        # We assign values to X and y to run the algorithm
        # ------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # ---------------------------
        # Fit and transform the class
        # ---------------------------
        y_dt = class_le.fit_transform(y_dt)

        # -------------------------------------
        # Split the dataset into train and test
        # -------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100,
                                                            stratify=y_dt)

        # -------------------------------------------
        # Data pre-processing: standardizing the data
        # -------------------------------------------
        std_sc = StandardScaler()

        std_sc.fit(x_train)

        x_train_std = std_sc.transform(x_train)
        x_test_std = std_sc.transform(x_test)

        # --------------------------------------------------
        # Performing training and making predictions on test
        # --------------------------------------------------
        self.clf = KNeighborsClassifier(n_neighbors=v_max_depth)
        self.clf.fit(x_train_std, y_train)

        y_pred_KNN = self.clf.predict(x_test_std)

        # ------------------------------
        # Confusion matrix for KNN model
        # ------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred_KNN)

        # ---------------------
        # Classification report
        # ---------------------
        self.ff_class_rep = classification_report(y_test, y_pred_KNN)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_KNN)*100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        class_names1 = ["", "under_95", "at_least_95"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run KNN where the MMR value is greater than or equal to 90%
# ----------------------------------------------------------------------------------------------------------------------


class KNN90(QMainWindow):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN90, self).__init__()

        self.Title = "KNN Classifier MMR >= 90"
        self.init_ui()

    def init_ui(self):
        # -----------------------------------------------------------
        # Create the canvas and all the element to create a dashboard
        # -----------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox("ML KNN Features")
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)

        self.lblPercentTest = QLabel("Percentage for Test:")
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel("N Neighbors:")
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        # ------------------------------------
        # Creating a checkbox for each feature
        # ------------------------------------
        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)

        self.groupBox2 = QGroupBox("Results from the model")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # -----------------------------------------------------------------
        # K-Nearest Neighbors Algorithm
        # We populate the dashboard using the parameters chosen by the user
        # -----------------------------------------------------------------
        # Processing the parameters
        # -----------------------------------------------------------------
        self.list_corr_features = pd.DataFrame([])

        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
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

        v_test_per = float(self.txtPercentTest.text())
        v_max_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per / 100

        # ------------------------------------------------
        # We assign values to X and y to run the algorithm
        # ------------------------------------------------
        x_dt = self.list_corr_features
        y_dt = mmr["at_least_95"]

        class_le = LabelEncoder()

        # ---------------------------
        # Fit and transform the class
        # ---------------------------
        y_dt = class_le.fit_transform(y_dt)

        # -------------------------------------
        # Split the dataset into train and test
        # -------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x_dt, y_dt, test_size=v_test_per, random_state=100,
                                                            stratify=y_dt)

        # -----------------------------------------
        # Data pre-processing: standardize the data
        # -----------------------------------------
        std_sc = StandardScaler()

        std_sc.fit(x_train)

        x_train_std = std_sc.transform(x_train)
        x_test_std = std_sc.transform(x_test)

        # ------------------------------------------------------
        # Performing training and creating the classifier object
        # ------------------------------------------------------
        self.clf = KNeighborsClassifier(n_neighbors=v_max_depth)

        # ------------------------------------------
        # Performing training and making predictions
        # ------------------------------------------
        self.clf.fit(x_train_std, y_train)

        y_pred_KNN = self.clf.predict(x_test_std)

        # ------------------------------
        # Confusion matrix for KNN model
        # ------------------------------
        conf_matrix = confusion_matrix(y_test, y_pred_KNN)

        # ---------------------
        # Classification report
        # ---------------------
        self.ff_class_rep = classification_report(y_test, y_pred_KNN)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # --------------
        # Accuracy score
        # --------------
        self.ff_accuracy_score = accuracy_score(y_test, y_pred_KNN)*100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        class_names1 = ["", "under_95", "at_least_95"]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf.predict_proba(x_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
# ----------------------------------------------------------------------------------------------------------------------
# Creating the class to run KNN using the state feature as the target variable
# ----------------------------------------------------------------------------------------------------------------------


class KnnStateTarget(QMainWindow):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KnnStateTarget, self).__init__()

        self.Title = "KNN Using State as the Target"
        self.init_ui()

    def init_ui(self):
        # ------------------------------------------------------------
        # Create the canvas and all the elements to create a dashboard
        # ------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox("Model Features")
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

        self.groupBox2 = QGroupBox("Results from Execution")
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel("Results:")
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel("Accuracy:")
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
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

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        # ----------------------------------------------------------------
        # KNN Algorithm
        # Populating the dashboard using the parameters chosen by the user
        # ----------------------------------------------------------------
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

        v_test_per = float(self.txtPercentTest.text())
        v_max_depth = int(self.txtMaxDepth.text())

        self.ax1.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        v_test_per = v_test_per/100

        x = self.list_corr_features
        y = measles["state"]

        le = LabelEncoder()

        y = le.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=v_test_per, random_state=100, stratify=y)

        standardize = StandardScaler()

        standardize.fit(x_train)

        x_train_standardized = standardize.transform(x_train)
        x_test_standardized = standardize.transform(x_test)

        self.model = KNeighborsClassifier(n_neighbors=v_max_depth)

        self.model.fit(x_train_standardized, y_train)

        y_pred = self.model.predict(x_test_standardized)

        # ---------------------------
        # Graphic 1: Confusion Matrix
        # ---------------------------
        conf_mat = confusion_matrix(y_test, y_pred)

        # ---------------------
        # Classification Report
        # ---------------------
        self.class_report = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.class_report)

        # --------------
        # Accuracy Score
        # --------------
        self.accur_score = accuracy_score(y_test, y_pred)*100
        self.txtAccuracy.setText(str(self.accur_score))

        # ----------------
        # Confusion Matrix
        # ----------------
        self.ax1.matshow(conf_mat, cmap=plt.cm.get_cmap("Blues", 14))
        self.ax1.set_yticklabels(classes, fontsize=7)
        self.ax1.set_xticklabels(classes, rotation=90, fontsize=7)
        self.ax1.set_xlabel("Predicted label")
        self.ax1.set_ylabel("True label")

        for i in range(len(classes)):
            for j in range(len(classes)):
                self.ax1.text(j, i, str(conf_mat[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean enrollment
# ----------------------------------------------------------------------------------------------------------------------


class VisEnrollment(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean Enrollment in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/enroll.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean mmr
# ----------------------------------------------------------------------------------------------------------------------


class VisMmr(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean MMR in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/mmr.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean overall vaccination rating
# ----------------------------------------------------------------------------------------------------------------------


class VisOverall(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean Overall Vaccination Rate in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/overall.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean xrel
# ----------------------------------------------------------------------------------------------------------------------


class VisXrel(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean xRel in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/xrel.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean xmed
# ----------------------------------------------------------------------------------------------------------------------


class VisXmed(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean xMed in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/xmed.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct visualization for distribution of mean xper
# ----------------------------------------------------------------------------------------------------------------------


class VisXper(QWidget):
    # ---------------------------------------------------------------------------------------
    # The methods in this class are as follows:
    # _init_ : initialize the class
    # init_ui : creates the canvas and all the elements in the canvas
    # update : populates the elements of the canvas base on the parameters chosen by the user
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Distribution of Mean xPer in the US"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        # ----------------------------------------------
        # Create the canvas to display the visualization
        # ----------------------------------------------
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # -------------------
        # Creating the widget
        # -------------------
        label = QLabel(self)
        pix_map = QPixmap("./visualization_data/xper.png")
        label.setPixmap(pix_map)
        self.resize(pix_map.width(), pix_map.height())
        self.show()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a class to construct the main window
# ----------------------------------------------------------------------------------------------------------------------


class App(QMainWindow):
    # --------------------------------------------------
    # This class creates all elements of the application
    # --------------------------------------------------
    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = "MMR Vaccination Rates"
        self.width = 500
        self.height = 300
        self.init_ui()

    def init_ui(self):
        # --------------------------
        # Creates the menu and items
        # --------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # ------------------------------------------
        # Create the menu bar and three items for
        # the menu, File, EDA Analysis and ML Models
        # ------------------------------------------
        main_menu = self.menuBar()
        main_menu.setStyleSheet("background-color: pink")

        file_menu = main_menu.addMenu("File")
        vis_menu = main_menu.addMenu("Visualization")
        ml_models_menu = main_menu.addMenu("ML Models")
        reg_menu = main_menu.addMenu("Regression")

        # --------------------------------------
        # Exit application
        # Creates actions for the file_menu item
        # --------------------------------------
        exit_button = QAction(QIcon("enter.png"), "Exit", self)
        exit_button.setShortcut("Ctrl+Q")
        exit_button.setStatusTip("Exit application")
        exit_button.triggered.connect(self.close)

        file_menu.addAction(exit_button)

        # --------------------
        # Decision Tree Models
        # --------------------
        dt_95_button = QAction(QIcon(), "Decision Tree 95", self)
        dt_95_button.setStatusTip("Decision Tree 95")
        dt_95_button.triggered.connect(self.ml_dt_95)

        dt_90_button = QAction(QIcon(), "Decision Tree 90", self)
        dt_90_button.setStatusTip("Decision Tree 90")
        dt_90_button.triggered.connect(self.ml_dt_90)

        # -------------------------
        # Random Forest Classifiers
        # -------------------------
        rf_95_button = QAction(QIcon(), "Random Forest 95", self)
        rf_95_button.setStatusTip("Random Forest 95")
        rf_95_button.triggered.connect(self.ml_rf_95)

        rf_90_button = QAction(QIcon(), "Random Forest 90", self)
        rf_90_button.setStatusTip("Random Forest 90")
        rf_90_button.triggered.connect(self.ml_rf_90)

        # --------------------------
        # K-Nearest Neighbors Models
        # --------------------------
        knn_95_button = QAction(QIcon(), "KNN 95", self)
        knn_95_button.setStatusTip("KNN 95")
        knn_95_button.triggered.connect(self.ml_knn_95)

        knn_90_button = QAction(QIcon(), "KNN 90", self)
        knn_90_button.setStatusTip("KNN 90")
        knn_90_button.triggered.connect(self.ml_knn_90)

        knn_state_target_button = QAction(QIcon(), "KNN State Target", self)
        knn_state_target_button.setStatusTip("KNN using state as the target variable")
        knn_state_target_button.triggered.connect(self.ml_knn_state)

        # --------------------
        # Visualization Models
        # --------------------
        vis_enroll_button = QAction(QIcon(), "Enrollment Distribution", self)
        vis_enroll_button.setStatusTip("Display a distribution of mean enrollment values in the US")
        vis_enroll_button.triggered.connect(self.vis_enroll)

        vis_mmr_button = QAction(QIcon(), "MMR Distribution", self)
        vis_mmr_button.setStatusTip("Display a distribution of mean MMR values in the US")
        vis_mmr_button.triggered.connect(self.vis_mmr)

        vis_overall_button = QAction(QIcon(), "Overall Distribution", self)
        vis_overall_button.setStatusTip("Display a distribution of mean overall vaccination rate values in the US")
        vis_overall_button.triggered.connect(self.vis_overall)

        vis_x_rel_button = QAction(QIcon(), "xRel Distribution", self)
        vis_x_rel_button.setStatusTip("Display a distribution of mean xRel values in the US")
        vis_x_rel_button.triggered.connect(self.vis_x_rel)

        vis_x_med_button = QAction(QIcon(), "xMed Distribution", self)
        vis_x_med_button.setStatusTip("Display a distribution of mean xMed values in the US")
        vis_x_med_button.triggered.connect(self.vis_x_med)

        vis_x_per_button = QAction(QIcon(), "xPer Distribution", self)
        vis_x_per_button.setStatusTip("Display a distribution of mean xPer values in the US")
        vis_x_per_button.triggered.connect(self.vis_x_per)

        # -----------------
        # Regression Models
        # -----------------
        reg_vermont_button = QAction(QIcon(), "Vermont", self)
        reg_vermont_button.setStatusTip("Results of Regression on Vermont")
        reg_vermont_button.triggered.connect(self.reg_vermont)

        reg_oregon_button = QAction(QIcon(), "Oregon", self)
        reg_oregon_button.setStatusTip("Results of Regression on Oregon")
        reg_oregon_button.triggered.connect(self.reg_oregon)

        # ---------------------------------------------------------
        # Adding the button actions for the machine learning models
        # ---------------------------------------------------------
        ml_models_menu.addAction(dt_95_button)
        ml_models_menu.addAction(dt_90_button)

        ml_models_menu.addAction(rf_95_button)
        ml_models_menu.addAction(rf_90_button)

        ml_models_menu.addAction(knn_95_button)
        ml_models_menu.addAction(knn_90_button)
        ml_models_menu.addAction(knn_state_target_button)

        vis_menu.addAction(vis_enroll_button)
        vis_menu.addAction(vis_mmr_button)
        vis_menu.addAction(vis_overall_button)
        vis_menu.addAction(vis_x_rel_button)
        vis_menu.addAction(vis_x_med_button)
        vis_menu.addAction(vis_x_per_button)

        reg_menu.addAction(reg_vermont_button)
        reg_menu.addAction(reg_oregon_button)

        self.dialogs = list()

    def ml_dt_95(self):
        # --------------------------------------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm using the m_tree dataset
        # --------------------------------------------------------------------------------------
        dialog = DecisionTree95()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_dt_90(self):
        # --------------------------------------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm using the m_tree dataset
        # --------------------------------------------------------------------------------------
        dialog = DecisionTree90()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_rf_95(self):
        # ---------------------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # ---------------------------------------------------------------------------
        dialog = RandomForest95()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_rf_90(self):
        # ---------------------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # ---------------------------------------------------------------------------
        dialog = RandomForest90()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_knn_95(self):
        # -----------------------------------------------------
        # This function creates an instance of the KNN95 class
        # This class presents a dashboard for the KNN Algorithm
        # -----------------------------------------------------
        dialog = KNN95()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_knn_90(self):
        # -----------------------------------------------------
        # This function creates an instance of the KNN90 class
        # This class presents a dashboard for the KNN Algorithm
        # -----------------------------------------------------
        dialog = KNN90()
        self.dialogs.append(dialog)
        dialog.show()

    def ml_knn_state(self):
        dialog = KnnStateTarget()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_enroll(self):
        dialog = VisEnrollment()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_mmr(self):
        dialog = VisMmr()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_overall(self):
        dialog = VisOverall()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_x_rel(self):
        dialog = VisXrel()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_x_med(self):
        dialog = VisXmed()
        self.dialogs.append(dialog)
        dialog.show()

    def vis_x_per(self):
        dialog = VisXper()
        self.dialogs.append(dialog)
        dialog.show()

    def reg_vermont(self):
        # --------------------------------------------------------
        # This function reads in the data, imputes missing values,
        # and runs regression for Vermont directly
        # --------------------------------------------------------
        data = pd.read_csv('./imputed_files/all-measles-rates.csv', sep=",", error_bad_lines=False)
        data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county',
                        'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']

        # ----------------------------
        # Imputation of missing values
        # ----------------------------
        enroll_mean = data['enroll'].mean(axis=0)
        data['enroll'].fillna(enroll_mean, inplace=True)
        xrel_mean = data['xrel'].mean(axis=0)
        data['xrel'].fillna(xrel_mean, inplace=True)
        xmed_mean = data['xmed'].mean(axis=0)
        data['xmed'].fillna(xmed_mean, inplace=True)
        xper_mean = data['xper'].mean(axis=0)
        data['xper'].fillna(xper_mean, inplace=True)

        # --------------------
        # Subset data by state
        # --------------------
        data = data[data["state"] == "Vermont"]

        # ------------------------------
        # Define feature and target data
        # ------------------------------
        data.loc[data.mmr == -1, 'mmr'] = 0
        data.loc[data.overall == -1, 'overall'] = 0
        y = data['mmr']
        x = data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]

        # --------------------------------------
        # Test train split and linear regression
        # --------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred = mlr.predict(x_test)

        # ------------------------------
        # Print coefficients and metrics
        # ------------------------------
        coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])

        # ----------------------------------------
        # View actual vs predicted and model score
        # ----------------------------------------
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        coeff_df = str(coeff_df)
        score = str(round(mlr.score(x_test, y_test), 3))

        QMessageBox.about(self, "Results Example1", coeff_df)
        QMessageBox.about(self, "Results Example2", score)

    def reg_oregon(self):
        # --------------------------------------------------------
        # This function reads in the data, imputes missing values,
        # and runs regression for Vermont directly
        # --------------------------------------------------------
        data = pd.read_csv('./imputed_files/all-measles-rates.csv', sep=",", error_bad_lines=False)

        data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county',
                        'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']

        # ----------------------------
        # Imputation of missing values
        # ----------------------------
        enroll_mean = data['enroll'].mean(axis=0)
        data['enroll'].fillna(enroll_mean, inplace=True)
        xrel_mean = data['xrel'].mean(axis=0)
        data['xrel'].fillna(xrel_mean, inplace=True)
        xmed_mean = data['xmed'].mean(axis=0)
        data['xmed'].fillna(xmed_mean, inplace=True)
        xper_mean = data['xper'].mean(axis=0)
        data['xper'].fillna(xper_mean, inplace=True)

        # --------------------
        # Subset data by state
        # --------------------
        data = data[data["state"] == "Oregon"]

        # ------------------------------
        # Define feature and target data
        # ------------------------------
        data.loc[data.mmr == -1, 'mmr'] = 0
        data.loc[data.overall == -1, 'overall'] = 0
        y = data['mmr']
        x = data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]

        # --------------------------------------
        # Test train split and linear regression
        # --------------------------------------
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred = mlr.predict(x_test)

        # ------------------------------
        # Print coefficients and metrics
        # ------------------------------
        coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])

        # ----------------------------------------
        # View actual vs predicted and model score
        # ----------------------------------------
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

        coeff_df = str(coeff_df)
        score = str(round(mlr.score(x_test, y_test), 3))

        QMessageBox.about(self, "Results Example1", coeff_df)
        QMessageBox.about(self, "Results Example2", score)
# ----------------------------------------------------------------------------------------------------------------------
# Functions called in class and GUI creation
# ----------------------------------------------------------------------------------------------------------------------


def main():
    # -------------------------
    # Initiates the application
    # -------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_mmr():
    # --------------------------------------------------
    # Loads the m_tree.csv and measles_imputed.csv files
    # Sets the feature and class names
    # --------------------------------------------------
    global mmr
    global features_list
    global class_names
    mmr = pd.read_csv('./imputed_files/m_tree.csv')
    features_list = ["state_mean", "city_mean", "county_mean", "type_of_school", "enroll", "xtotal"]
    class_names = ['under_95', 'at_least_95']
    global measles
    global features
    global classes
    measles = pd.read_csv('./imputed_files/measles_imputed.csv')
    measles['state'] = measles['state'].astype('category')
    features = ['enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
    classes = measles['state'].unique()


if __name__ == '__main__':
    # ------------------------------------------
    # Reads the data, then calls the application
    # ------------------------------------------
    data_mmr()
    main()
