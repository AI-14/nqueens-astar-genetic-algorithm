# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(310, 390, 61, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextButton.sizePolicy().hasHeightForWidth())
        self.nextButton.setSizePolicy(sizePolicy)
        self.nextButton.setObjectName("nextButton")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(20, 60, 461, 321))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageLabel.sizePolicy().hasHeightForWidth())
        self.imageLabel.setSizePolicy(sizePolicy)
        self.imageLabel.setMaximumSize(QtCore.QSize(2000, 2000))
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setObjectName("imageLabel")
        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevButton.setGeometry(QtCore.QRect(144, 390, 61, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prevButton.sizePolicy().hasHeightForWidth())
        self.prevButton.setSizePolicy(sizePolicy)
        self.prevButton.setObjectName("prevButton")
        self.outputLabel = QtWidgets.QLabel(self.centralwidget)
        self.outputLabel.setGeometry(QtCore.QRect(40, 10, 411, 31))
        self.outputLabel.setTextFormat(QtCore.Qt.RichText)
        self.outputLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.outputLabel.setObjectName("outputLabel")
        self.runAStarButton = QtWidgets.QPushButton(self.centralwidget)
        self.runAStarButton.setGeometry(QtCore.QRect(540, 60, 121, 41))
        self.runAStarButton.setStyleSheet("")
        self.runAStarButton.setObjectName("runAStarButton")
        self.mutprobDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.mutprobDoubleSpinBox.setGeometry(QtCore.QRect(640, 190, 51, 22))
        self.mutprobDoubleSpinBox.setMaximum(1.0)
        self.mutprobDoubleSpinBox.setProperty("value", 0.8)
        self.mutprobDoubleSpinBox.setObjectName("mutprobDoubleSpinBox")
        self.mutProbLabel = QtWidgets.QLabel(self.centralwidget)
        self.mutProbLabel.setGeometry(QtCore.QRect(520, 190, 101, 21))
        self.mutProbLabel.setObjectName("mutProbLabel")
        self.nQueenSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.nQueenSpinBox.setGeometry(QtCore.QRect(641, 10, 51, 22))
        self.nQueenSpinBox.setMinimum(4)
        self.nQueenSpinBox.setMaximum(100)
        self.nQueenSpinBox.setObjectName("nQueenSpinBox")
        self.nQueensLabel = QtWidgets.QLabel(self.centralwidget)
        self.nQueensLabel.setGeometry(QtCore.QRect(520, 10, 101, 21))
        self.nQueensLabel.setObjectName("nQueensLabel")
        self.initpopSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.initpopSpinBox.setGeometry(QtCore.QRect(641, 160, 51, 22))
        self.initpopSpinBox.setMinimum(5)
        self.initpopSpinBox.setMaximum(150)
        self.initpopSpinBox.setObjectName("initpopSpinBox")
        self.initpopLabel = QtWidgets.QLabel(self.centralwidget)
        self.initpopLabel.setGeometry(QtCore.QRect(520, 160, 101, 21))
        self.initpopLabel.setObjectName("initpopLabel")
        self.maxgenLabel = QtWidgets.QLabel(self.centralwidget)
        self.maxgenLabel.setGeometry(QtCore.QRect(520, 220, 101, 21))
        self.maxgenLabel.setObjectName("maxgenLabel")
        self.maxgenSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.maxgenSpinBox.setGeometry(QtCore.QRect(640, 220, 51, 22))
        self.maxgenSpinBox.setMinimum(100)
        self.maxgenSpinBox.setMaximum(1000000)
        self.maxgenSpinBox.setObjectName("maxgenSpinBox")
        self.parentSelectLabel = QtWidgets.QLabel(self.centralwidget)
        self.parentSelectLabel.setGeometry(QtCore.QRect(520, 340, 101, 21))
        self.parentSelectLabel.setObjectName("parentSelectLabel")
        self.runGaButton = QtWidgets.QPushButton(self.centralwidget)
        self.runGaButton.setGeometry(QtCore.QRect(540, 372, 121, 41))
        self.runGaButton.setObjectName("runGaButton")
        self.stopgaButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopgaButton.setGeometry(QtCore.QRect(540, 420, 121, 31))
        self.stopgaButton.setObjectName("stopgaButton")
        self.stopastarButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopastarButton.setGeometry(QtCore.QRect(540, 110, 121, 31))
        self.stopastarButton.setObjectName("stopastarButton")
        self.imagecountLabel = QtWidgets.QLabel(self.centralwidget)
        self.imagecountLabel.setGeometry(QtCore.QRect(210, 390, 91, 20))
        self.imagecountLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imagecountLabel.setObjectName("imagecountLabel")
        self.hLine2 = QtWidgets.QFrame(self.centralwidget)
        self.hLine2.setGeometry(QtCore.QRect(510, 140, 191, 20))
        self.hLine2.setFrameShape(QtWidgets.QFrame.HLine)
        self.hLine2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.hLine2.setObjectName("hLine2")
        self.vLine1 = QtWidgets.QFrame(self.centralwidget)
        self.vLine1.setGeometry(QtCore.QRect(499, 0, 21, 461))
        self.vLine1.setFrameShape(QtWidgets.QFrame.VLine)
        self.vLine1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.vLine1.setObjectName("vLine1")
        self.hLine3 = QtWidgets.QFrame(self.centralwidget)
        self.hLine3.setGeometry(QtCore.QRect(0, 450, 701, 21))
        self.hLine3.setFrameShape(QtWidgets.QFrame.HLine)
        self.hLine3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.hLine3.setObjectName("hLine3")
        self.hLine1 = QtWidgets.QFrame(self.centralwidget)
        self.hLine1.setGeometry(QtCore.QRect(510, 40, 191, 20))
        self.hLine1.setFrameShape(QtWidgets.QFrame.HLine)
        self.hLine1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.hLine1.setObjectName("hLine1")
        self.crossoverTypeComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.crossoverTypeComboBox.setGeometry(QtCore.QRect(620, 310, 71, 22))
        self.crossoverTypeComboBox.setObjectName("crossoverTypeComboBox")
        self.crossoverTypeLabel = QtWidgets.QLabel(self.centralwidget)
        self.crossoverTypeLabel.setGeometry(QtCore.QRect(520, 310, 101, 21))
        self.crossoverTypeLabel.setObjectName("crossoverTypeLabel")
        self.parentSelectComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.parentSelectComboBox.setGeometry(QtCore.QRect(620, 340, 71, 22))
        self.parentSelectComboBox.setObjectName("parentSelectComboBox")
        self.resetButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(420, 392, 75, 51))
        self.resetButton.setObjectName("resetButton")
        self.crossoverRateLabel = QtWidgets.QLabel(self.centralwidget)
        self.crossoverRateLabel.setGeometry(QtCore.QRect(520, 250, 101, 21))
        self.crossoverRateLabel.setObjectName("crossoverRateLabel")
        self.crossoverRateDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.crossoverRateDoubleSpinBox.setGeometry(QtCore.QRect(640, 250, 51, 22))
        self.crossoverRateDoubleSpinBox.setMinimum(0.2)
        self.crossoverRateDoubleSpinBox.setMaximum(1.0)
        self.crossoverRateDoubleSpinBox.setProperty("value", 0.8)
        self.crossoverRateDoubleSpinBox.setObjectName("crossoverRateDoubleSpinBox")
        self.elitismLabel = QtWidgets.QLabel(self.centralwidget)
        self.elitismLabel.setGeometry(QtCore.QRect(520, 280, 101, 21))
        self.elitismLabel.setObjectName("elitismLabel")
        self.elitismComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.elitismComboBox.setGeometry(QtCore.QRect(620, 280, 71, 22))
        self.elitismComboBox.setObjectName("elitismComboBox")
        self.autoTrace = QtWidgets.QPushButton(self.centralwidget)
        self.autoTrace.setGeometry(QtCore.QRect(10, 390, 91, 23))
        self.autoTrace.setObjectName("autoTrace")
        self.stopAutotrace = QtWidgets.QPushButton(self.centralwidget)
        self.stopAutotrace.setGeometry(QtCore.QRect(10, 420, 91, 23))
        self.stopAutotrace.setObjectName("stopAutotrace")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.nextButton.setText(_translate("MainWindow", "Next"))
        self.imageLabel.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" font-size:12pt; font-weight:600; text-decoration: underline;\">Instructions</span><span style=\" font-size:12pt; font-weight:600;\">:</span></p><p align=\"justify\">1. Choose to run A* or Genetic algorithm at a time.</p><p align=\"justify\">2. When output appears on the top label, then click &lt;Next&gt; button </p><p align=\"justify\">to view the images (for visual tracing) or use &lt;Auto Trace&gt;.</p><p align=\"justify\">NOTE: Check if the folder &lt;states_images&gt; is created or not.</p><p align=\"justify\">3. Click on &lt;RESET&gt; button to clear &lt;states_images&gt; folder.</p><p align=\"justify\">Then run any algorithm again.</p><p align=\"justify\"><span style=\" font-weight:600; text-decoration: underline;\">Dictionary:</span></p><p align=\"justify\">1. Crossover type - SP (Single point), TP (Two point)</p><p align=\"justify\">2. Parent selection - RS (Rank selection), RWS (Roulette wheel selection)</p><p align=\"justify\">3. Elitism - Y (Yes), N (No)</p><p align=\"justify\">ENJOY THE APP!</p></body></html>"))
        self.prevButton.setText(_translate("MainWindow", "Prev"))
        self.outputLabel.setText(_translate("MainWindow", "Steps/Time"))
        self.runAStarButton.setText(_translate("MainWindow", "Run A*"))
        self.mutProbLabel.setText(_translate("MainWindow", "Mutation probability"))
        self.nQueensLabel.setText(_translate("MainWindow", "Choose N Queens"))
        self.initpopLabel.setText(_translate("MainWindow", "Initial population"))
        self.maxgenLabel.setText(_translate("MainWindow", "Max generation"))
        self.parentSelectLabel.setText(_translate("MainWindow", "Parent selection "))
        self.runGaButton.setText(_translate("MainWindow", "Run Genetic Algorithm"))
        self.stopgaButton.setText(_translate("MainWindow", "Stop"))
        self.stopastarButton.setText(_translate("MainWindow", "Stop"))
        self.imagecountLabel.setText(_translate("MainWindow", "Image Count"))
        self.crossoverTypeLabel.setText(_translate("MainWindow", "Crossover type"))
        self.resetButton.setText(_translate("MainWindow", "RESET"))
        self.crossoverRateLabel.setText(_translate("MainWindow", "Crossover rate"))
        self.elitismLabel.setText(_translate("MainWindow", "Elitism?"))
        self.autoTrace.setText(_translate("MainWindow", "Auto Trace"))
        self.stopAutotrace.setText(_translate("MainWindow", "Stop AutoTrace"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())