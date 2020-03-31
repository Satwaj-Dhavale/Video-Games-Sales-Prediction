import sys

from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QMainWindow, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableView, QStyleFactory, QTableWidgetItem, QScrollArea, QHeaderView

import Graphs
import main

target_train, target_test, prediction, mae, rmse, random_case, col_list = main.get_values()

class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)
        self.setWindowIcon(QIcon('feather.png'))

        self.btn_mae = QPushButton('MAE Comparison', self)
        self.btn_mae.resize(250, 75)
        self.btn_mae.move(50, 120)
        self.btn_mae.clicked.connect(self.disp_mae)

        self.btn_rmse = QPushButton('RMSE Comparison', self)
        self.btn_rmse.resize(250, 75)
        self.btn_rmse.move(500, 120)
        self.btn_rmse.clicked.connect(self.disp_rmse)

        self.ToolsBTN = QPushButton('Close Session', self)
        self.ToolsBTN.move(300, 400)
        self.ToolsBTN.resize(250, 75)

        self.show()

    def disp_mae(self):
        Graphs.model_mae_comparison_graph(mae)

    def disp_rmse(self):
        Graphs.model_rmse_comparison_graph(rmse)


class UIToolTab(QWidget):
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)

        self.setWindowIcon(QIcon('feather.png'))

        self.btn_graph = QPushButton('Graph', self)
        self.btn_graph.resize(250, 75)
        self.btn_graph.move(50, 120)

        self.btn_manual = QPushButton('View a Test Case', self)
        self.btn_manual.resize(250, 75)
        self.btn_manual.move(500, 120)

        self.btn_quit1 = QPushButton("Quit", self)
        self.btn_quit1.clicked.connect(self.quitter)
        self.btn_quit1.resize(250, 75)
        self.btn_quit1.move(300, 400)

    def quitter(self):
        sys.exit()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setGeometry(100, 100, 800, 500)
        self.setWindowTitle('DWM Project')
        self.setWindowIcon(QIcon('feather.png'))

        self.setStyle(QStyleFactory.create('Fusion'))

        self.setStyleSheet("background-color: white;")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        self.startUIToolTab()

    def startUIToolTab(self):
        self.ToolTab = UIToolTab(self)
        self.setCentralWidget(self.ToolTab)

        self.ToolTab.btn_graph.clicked.connect(self.startUIWindow)
        self.ToolTab.btn_manual.clicked.connect(self.startUI1Window)
        self.show()

    def startUIWindow(self):
        self.Window = UIWindow(self)
        self.setWindowTitle("Session Started")
        self.setCentralWidget(self.Window)
        # Main window
        self.Window.ToolsBTN.clicked.connect(self.startUIToolTab)
        self.show()

    def startUI1Window(self):

        self.win = QWidget()
        self.scroll = QScrollArea()
        self.layout = QVBoxLayout()
        self.table = QTableWidget()
        self.scroll.setWidget(self.table)
        self.layout.addWidget(self.table)
        self.win.setLayout(self.layout)

        self.win.setWindowIcon(QIcon('feather.png'))
        self.win.setWindowTitle('Predicted Value')
        self.win.setGeometry(100, 200, 1750, 300)

        df = random_case
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(2)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                self.table.setItem(0, j, QTableWidgetItem(str(col_list[j])))

                self.table.setItem(i + 1, j, QTableWidgetItem(str(df.iloc[i, j])))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)

        self.win.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
