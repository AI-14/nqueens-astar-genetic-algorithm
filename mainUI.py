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
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(100, 420, 75, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextButton.sizePolicy().hasHeightForWidth())
        self.nextButton.setSizePolicy(sizePolicy)
        self.nextButton.setObjectName("nextButton")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(20, 60, 461, 341))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageLabel.sizePolicy().hasHeightForWidth())
        self.imageLabel.setSizePolicy(sizePolicy)
        self.imageLabel.setMaximumSize(QtCore.QSize(2000, 2000))
        self.imageLabel.setText("")
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setObjectName("imageLabel")
        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevButton.setGeometry(QtCore.QRect(10, 420, 75, 23))
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
        self.runAStarButton.setGeometry(QtCore.QRect(540, 80, 111, 23))
        self.runAStarButton.setObjectName("runAStarButton")
        self.mutprobDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.mutprobDoubleSpinBox.setGeometry(QtCore.QRect(640, 270, 51, 22))
        self.mutprobDoubleSpinBox.setMaximum(1.0)
        self.mutprobDoubleSpinBox.setProperty("value", 0.8)
        self.mutprobDoubleSpinBox.setObjectName("mutprobDoubleSpinBox")
        self.mutProbLabel = QtWidgets.QLabel(self.centralwidget)
        self.mutProbLabel.setGeometry(QtCore.QRect(520, 270, 101, 21))
        self.mutProbLabel.setObjectName("mutProbLabel")
        self.nQueenAStarSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.nQueenAStarSpinBox.setGeometry(QtCore.QRect(641, 10, 51, 22))
        self.nQueenAStarSpinBox.setMinimum(4)
        self.nQueenAStarSpinBox.setMaximum(100)
        self.nQueenAStarSpinBox.setObjectName("nQueenAStarSpinBox")
        self.nQueensLabel = QtWidgets.QLabel(self.centralwidget)
        self.nQueensLabel.setGeometry(QtCore.QRect(520, 10, 101, 21))
        self.nQueensLabel.setObjectName("nQueensLabel")
        self.maxlevelAStarSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.maxlevelAStarSpinBox.setGeometry(QtCore.QRect(641, 40, 51, 22))
        self.maxlevelAStarSpinBox.setMinimum(2)
        self.maxlevelAStarSpinBox.setMaximum(1000)
        self.maxlevelAStarSpinBox.setObjectName("maxlevelAStarSpinBox")
        self.maxlevelAStarLabel = QtWidgets.QLabel(self.centralwidget)
        self.maxlevelAStarLabel.setGeometry(QtCore.QRect(520, 40, 101, 21))
        self.maxlevelAStarLabel.setObjectName("maxlevelAStarLabel")
        self.initpopSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.initpopSpinBox.setGeometry(QtCore.QRect(641, 240, 51, 22))
        self.initpopSpinBox.setMinimum(5)
        self.initpopSpinBox.setMaximum(150)
        self.initpopSpinBox.setObjectName("initpopSpinBox")
        self.initpopLabel = QtWidgets.QLabel(self.centralwidget)
        self.initpopLabel.setGeometry(QtCore.QRect(520, 240, 101, 21))
        self.initpopLabel.setObjectName("initpopLabel")
        self.nQueensGaLabel = QtWidgets.QLabel(self.centralwidget)
        self.nQueensGaLabel.setGeometry(QtCore.QRect(520, 210, 101, 21))
        self.nQueensGaLabel.setObjectName("nQueensGaLabel")
        self.nQueensGaSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.nQueensGaSpinBox.setGeometry(QtCore.QRect(641, 210, 51, 22))
        self.nQueensGaSpinBox.setMinimum(4)
        self.nQueensGaSpinBox.setMaximum(100)
        self.nQueensGaSpinBox.setObjectName("nQueensGaSpinBox")
        self.maxgenLabel = QtWidgets.QLabel(self.centralwidget)
        self.maxgenLabel.setGeometry(QtCore.QRect(520, 300, 101, 21))
        self.maxgenLabel.setObjectName("maxgenLabel")
        self.maxgenSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.maxgenSpinBox.setGeometry(QtCore.QRect(641, 300, 51, 22))
        self.maxgenSpinBox.setMinimum(100)
        self.maxgenSpinBox.setMaximum(1000000)
        self.maxgenSpinBox.setObjectName("maxgenSpinBox")
        self.parentSelectComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.parentSelectComboBox.setGeometry(QtCore.QRect(619, 340, 71, 22))
        self.parentSelectComboBox.setObjectName("parentSelectComboBox")
        self.parentSelectLabel = QtWidgets.QLabel(self.centralwidget)
        self.parentSelectLabel.setGeometry(QtCore.QRect(520, 340, 101, 21))
        self.parentSelectLabel.setObjectName("parentSelectLabel")
        self.runGaButton = QtWidgets.QPushButton(self.centralwidget)
        self.runGaButton.setGeometry(QtCore.QRect(540, 380, 121, 23))
        self.runGaButton.setObjectName("runGaButton")
        self.stopgaButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopgaButton.setGeometry(QtCore.QRect(560, 410, 75, 23))
        self.stopgaButton.setObjectName("stopgaButton")
        self.stopastarButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopastarButton.setGeometry(QtCore.QRect(560, 110, 75, 23))
        self.stopastarButton.setObjectName("stopastarButton")
        self.imagecountLabel = QtWidgets.QLabel(self.centralwidget)
        self.imagecountLabel.setGeometry(QtCore.QRect(386, 420, 91, 20))
        self.imagecountLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imagecountLabel.setObjectName("imagecountLabel")
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
        self.prevButton.setText(_translate("MainWindow", "Prev"))
        self.outputLabel.setText(_translate("MainWindow", "Steps/Time"))
        self.runAStarButton.setText(_translate("MainWindow", "Run A*"))
        self.mutProbLabel.setText(_translate("MainWindow", "Mutation Probability"))
        self.nQueensLabel.setText(_translate("MainWindow", "Choose N Queens"))
        self.maxlevelAStarLabel.setText(_translate("MainWindow", "Max levels"))
        self.initpopLabel.setText(_translate("MainWindow", "Initial population"))
        self.nQueensGaLabel.setText(_translate("MainWindow", "Choose N Queens"))
        self.maxgenLabel.setText(_translate("MainWindow", "Max generation"))
        self.parentSelectLabel.setText(_translate("MainWindow", "Parent Selection "))
        self.runGaButton.setText(_translate("MainWindow", "Run Genetic Algorithm"))
        self.stopgaButton.setText(_translate("MainWindow", "Stop"))
        self.stopastarButton.setText(_translate("MainWindow", "Stop"))
        self.imagecountLabel.setText(_translate("MainWindow", "Image Count"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
