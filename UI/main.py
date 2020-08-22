from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon
from output import Ui_MainWindow
from predict_text import BERT
import sys
import os
from PyQt5.QtGui import QPixmap

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
    
class mywindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.model = BERT()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.ui.label.setText("")
        self.ui.analysis.clicked.connect(self.predict)
        self.show()

    def predict(self):
        text = self.ui.TextInput.toPlainText()
  
        # pixmap1 = QPixmap(resource_path("neg.png"))
        # pixmap2 = QPixmap(resource_path("pos.jpg"))
        result = self.model.predict_text(text)
        
        if result[0][0] == 0.0:
            k = "NEGATIVE"
            # self.ui.label_6.setPixmap(pixmap1)
        else:
            k = "POSITIVE" 
            # self.ui.label_6.setPixmap(pixmap2)
        # self.ui.label.setText(k)
        QMessageBox.information(None, 'Result', k)
        # self.show()
        # print(result)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    application = mywindow()
    application.show()
    sys.exit(app.exec())