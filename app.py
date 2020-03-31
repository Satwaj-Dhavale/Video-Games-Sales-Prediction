import sys

from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QMainWindow, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableView, QStyleFactory, QTableWidgetItem, QScrollArea, QHeaderView

import Graphs
import main

target_train, target_test, prediction, mae, rmse, random_tuple, col_list = main.get_values()


class MyTableModel(QAbstractTableModel):
    def __init__(self, datain, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)
        self.arraydata = datain

    def rowCount(self, parent):
        return 1

    def columnCount(self, parent):
        return 16

    def data(self, index, role):
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        return QVariant(self.arraydata[index.row()][index.column()])


class UIWindow(QWidget):
    def __init__(self, parent=None):
        # noinspection PyArgumentList
        super(UIWindow, self).__init__(parent)
        self.setWindowIcon(QIcon('feather.png'))
        '''
        Add all buttons for displaying the graph in this init
        '''

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


class TableView(QTableView):
    """
    A simple table to demonstrate the QComboBox delegate.
    """

    def __init__(self, *args, **kwargs):
        QTableView.__init__(self, *args, **kwargs)


class UIWindow1(QWidget):
    def __init__(self, parent=None):
        # noinspection PyArgumentList
        super(UIWindow1, self).__init__(parent)
        self.setWindowIcon(QIcon('feather.png'))
        # self.setGeometry(00, 300, 1900, 1080)
        # self.updateGeometry()

        self.ToolsBTN1 = QPushButton('Close Session', self)
        self.ToolsBTN1.move(300, 400)
        self.ToolsBTN1.resize(250, 75)

        self.label = QLabel(self)
        self.label.setText('The predicted value for the unknown tuple is: {:.04f} M'.format(prediction[0]))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.move(0, 200)

        # self.label1 = QLabel(self)
        # self.label1.setText(random_tuple)
        # self.label1.move(0,100)
        # self.label1.setAlignment((Qt.AlignCenter))

        # self.tablemodel = MyTableModel(random_tuple, self)
        # self.tableview = QTableView()
        # self.tableview.setModel(self.tablemodel)
        # self.tableview.show()

        # self.tableWidget = QTableWidget()
        # self.tableWidget.setRowCount(1)
        # self.tableWidget.setColumnCount(16)
        # self.tableWidget.setItem(0,0, QTableWidgetItem("Cell (1,1)"))

        # self.layout = QVBoxLayout()
        # self.layout.addWidget(self.tableWidget)
        # self.setLayout(self.layout)
        # self.show()

        self.win = QWidget()
        self.scroll = QScrollArea()
        self.layout = QVBoxLayout()
        self.table = QTableWidget()
        self.scroll.setWidget(self.table)
        self.layout.addWidget(self.table)
        self.win.setLayout(self.layout)

        df = random_tuple
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(2)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                self.table.setItem(0, j, QTableWidgetItem(str(col_list[j])))

                self.table.setItem(i + 1, j, QTableWidgetItem(str(df.iloc[i, j])))

        self.win.show()


class UIToolTab(QWidget):
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)

        self.setWindowIcon(QIcon('feather.png'))

        self.btn_graph = QPushButton('Graph', self)
        self.btn_graph.resize(250, 75)
        self.btn_graph.move(50, 120)

        self.btn_manual = QPushButton('Manually Predict', self)
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
        # self.Window1 = UIWindow1(self)
        #
        # self.setWindowTitle("Faculty Menu")
        # self.setCentralWidget(self.Window1)
        # self.Window1.ToolsBTN1.clicked.connect(self.startUIToolTab)
        #
        # self.show()

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

        df = random_tuple
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
