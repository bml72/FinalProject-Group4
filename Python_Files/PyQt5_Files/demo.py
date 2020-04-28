#::----------------------------------------------------------------
#:: To create a Menu with options this are the libraries and components that
#:: requiered. For each new option we will be o adding new components
#::----------------------------------------------------------------
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox   # No.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

#::-------------------------------------------------------------
#:: Definition of a Class for the main manu in the application
#::-------------------------------------------------------------
class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        #::-----------------------
        #:: variables use to set the size of the window that contains the menu
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300

        #:: Title for the application

        self.Title = 'Regression'

        #:: The initUi is call to create all the necessary elements for the menu

        self.initUI()

    def initUI(self):

        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        #::-----------------------------
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        #:: Add another option to the Menu Bar

        exampleWin = mainMenu.addMenu ('Examples')   # No. 2

        #::--------------------------------------
        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #:: This line adds the button (item element ) to the menu

        fileMenu.addAction(exitButton)

        #::----------------------------------------------------
        #::Add Example 1 We create the item Menu Example1
        #::This option will present a message box upon request
        #::----------------------------------------------------

        example1Button = QAction("Vermont", self)    # No. 2
        example1Button.setStatusTip("Results")   # No. 2
        example1Button.triggered.connect(self.Vermont)
        
        example2Button = QAction("Oregon", self)    # No. 2
        example2Button.setStatusTip("Results")   # No. 2
        example2Button.triggered.connect(self.Oregon) 
        
           # No. 2

        #:: We addd the example1Button action to the Menu Examples
        exampleWin.addAction(example1Button)
        exampleWin.addAction(example2Button)     # No. 2

        #:: This line shows the windows

        self.show()

    def Vermont(self): 
    	
        data = pd.read_csv('all-measles-rates.csv', sep=",", error_bad_lines=False)
        data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county', 
        'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
        
        
        #Impute means into missing values
        enroll_mean = data['enroll'].mean(axis=0)
        data['enroll'].fillna(enroll_mean, inplace=True)
        xrel_mean = data['xrel'].mean(axis=0)
        data['xrel'].fillna(xrel_mean, inplace=True)
        xmed_mean = data['xmed'].mean(axis=0)
        data['xmed'].fillna(xmed_mean, inplace=True)
        xper_mean = data['xper'].mean(axis=0)
        data['xper'].fillna(xper_mean, inplace=True)
        
        
        
        #Subset data by state
        
        data = data[data["state"] == "Vermont"]
        
        #Define feature and target data
        data.loc[data.mmr == -1, 'mmr'] = 0
        data.loc[data.overall == -1, 'overall'] = 0
        y = data['mmr']
        x = data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]
        #Test train split and linear regression
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    
        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred = mlr.predict(x_test)
        #Print coefficients and metrics
        
        coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])
        #View actual vs predicted and model score
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        coeff_df=str(coeff_df)
        score = str(round(mlr.score(x_test, y_test), 3))
        
        QMessageBox.about(self, "Results Example1", coeff_df)
        QMessageBox.about(self, "Results Example2", score)
        
    
    def Oregon(self):
        
        data = pd.read_csv('all-measles-rates.csv', sep=",", error_bad_lines=False)
        
        data.columns = ['index', 'state', 'year', 'name', 'type', 'city', 'county', 
        'district', 'enroll', 'mmr', 'overall', 'xrel', 'xmed', 'xper']
        
        #Impute means into missing values
        enroll_mean = data['enroll'].mean(axis=0)
        data['enroll'].fillna(enroll_mean, inplace=True)
        xrel_mean = data['xrel'].mean(axis=0)
        data['xrel'].fillna(xrel_mean, inplace=True)
        xmed_mean = data['xmed'].mean(axis=0)
        data['xmed'].fillna(xmed_mean, inplace=True)
        xper_mean = data['xper'].mean(axis=0)
        data['xper'].fillna(xper_mean, inplace=True)
        
        #Subset data by state
        
        data = data[data["state"] == "Oregon"]
        
        #Define feature and target data
        data.loc[data.mmr == -1, 'mmr'] = 0
        data.loc[data.overall == -1, 'overall'] = 0
        y = data['mmr']
        x = data[['enroll', 'overall', 'xrel', 'xmed', 'xper']]
        #Test train split and linear regression
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        
        mlr = LinearRegression()
        mlr.fit(x_train, y_train)
        y_pred = mlr.predict(x_test)
        #Print coefficients and metrics

        coeff_df = pd.DataFrame(mlr.coef_, x.columns, columns=['Coefficient'])
        #View actual vs predicted and model score
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        coeff_df=str(coeff_df)
        score = str(round(mlr.score(x_test, y_test), 3))
        
        QMessageBox.about(self, "Results Example1", coeff_df)
        QMessageBox.about(self, "Results Example2", score)
        
         # No. 2

#::------------------------
#:: Application starts here
#::------------------------

def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

if __name__ == '__main__':
    main()