from PyQt5 import QtWidgets as qtw, QtCore, QtGui
from mainUI import Ui_MainWindow
from PyQt5.QtCore import QThread
import sys, os, time, PyQt5, shutil
from algorithms.genetic_algorithm import RunGeneticAlgorithm


if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


folder = 'states_images'
ga_output = []
a_star_output = []


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.index = -1
        self.image_names = []
        
        self.ui.parentSelectComboBox.addItem('RS')
        self.ui.parentSelectComboBox.addItem('RWS')

        self.ga_thread = None
        self.a_star_thread = None
        
        self.ui.nextButton.clicked.connect(self.showNextImage)
        self.ui.prevButton.clicked.connect(self.showPrevImage)

        self.ui.runAStarButton.clicked.connect(self.run_a_star)
        self.ui.stopastarButton.clicked.connect(self.stop_a_star_thread)

        self.ui.runGaButton.clicked.connect(self.run_genetic_algorithm)
        self.ui.stopgaButton.clicked.connect(self.stop_ga_thread)


    def run_a_star(self):
        N_queens = self.ui.nQueenAStarSpinBox.value()
        max_level = self.ui.maxlevelAStarSpinBox.value()

        self.a_star_thread = AStarAlgorithmThread(
            N_queens=N_queens,
            max_level=max_level
        )

        self.a_star_thread.start()
        self.a_star_thread.finished.connect(self.show_a_star_results)


    def stop_a_star_thread(self):
        self.ui.outputLabel.setText('Searching Stopped!')
        self.a_star_thread.terminate()

    def show_a_star_results(self):
        global a_star_output
        if len(a_star_output) != 0:
            output = f'Steps = {a_star_output[0]} | Time = {a_star_output[1] :.4} secs'
            self.ui.outputLabel.setText(output)
            self.load_images_directory()
        else:
            print('Output List is Empty!')


    def run_genetic_algorithm(self):
        N_queens = self.ui.nQueensGaSpinBox.value()
        init_pop_size = self.ui.initpopSpinBox.value()
        mutation_prob = self.ui.mutprobDoubleSpinBox.value()
        max_gen = self.ui.maxgenSpinBox.value()
        parent_selection_algo = self.ui.parentSelectComboBox.currentText()

        self.ga_thread = GeneticAlgorithmThread(
                        N_queens=N_queens,
                        init_pop_size=init_pop_size,
                        mutation_prob=mutation_prob,
                        max_gen=max_gen,
                        parent_selection_algo=parent_selection_algo
                        )

        self.ga_thread.start()
        self.ga_thread.finished.connect(self.show_ga_results)
        

    def stop_ga_thread(self):
        self.ui.outputLabel.setText('Searching Stopped!')
        self.ga_thread.terminate()

    def show_ga_results(self):
        global ga_output
        if len(ga_output) != 0:
            output = f'Steps = {ga_output[0]} | Time = {ga_output[1] :.4} secs | Solution Found = {ga_output[2]}'
            self.ui.outputLabel.setText(output)
            self.load_images_directory()
        else:
            print('Output List is Empty!')


    def load_images_directory(self):
        for filename in os.listdir(folder):
            self.image_names.append(filename)
        
        
    def showNextImage(self):
        self.index +=1
        if self.index >= len(self.image_names):
            print("Nothing next")
            self.index = -1
        else:
            img = self.image_names[self.index]
            self.ui.imageLabel.setPixmap(QtGui.QPixmap(folder+'\\'+img))


    def showPrevImage(self):
        self.index -= 1
        if self.index <= -1:
            print('Nothing prev')
            self.index = -1
        else:
            img = self.image_names[self.index]
            self.ui.imageLabel.setPixmap(QtGui.QPixmap(folder+'\\'+img))


class AStarAlgorithmThread(QThread):
    def __init__(self, N_queens, max_level):
        super().__init__()
        self.N_queens = N_queens
        self.max_level = max_level

    def run(self):
        global a_star_output

        """
            TODO: Call A star algorithm. 
        """


class GeneticAlgorithmThread(QThread):
    def __init__(self, N_queens, init_pop_size, mutation_prob, max_gen, parent_selection_algo):
        super().__init__()
        self.N_queens = N_queens
        self.init_pop_size = init_pop_size
        self.mutation_prob = mutation_prob
        self.max_gen = max_gen
        self.parent_selection_algo = parent_selection_algo

    def run(self):
        global ga_output

        ga_output = RunGeneticAlgorithm.run_ga(
                    N_queens=self.N_queens,
                    init_pop_size=self.init_pop_size,
                    mutation_prob=self.mutation_prob,
                    max_gen=self.max_gen,
                    parent_selection_algo=self.parent_selection_algo
                    )


if __name__ == '__main__':
    app = qtw.QApplication([])
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())