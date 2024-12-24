from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
import sys
import cv2
import numpy as np

import Main

global ratio
ratio = 2  # 调节屏幕所占比例

global left
global right
select_img = []  # 全局变量，用于存储选择的图片


class Panorama_Window(QMainWindow):
    def __init__(self):
        super(Panorama_Window, self).__init__()
        self.setupUi(self)

    # class Panorama_Window(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(int(800 * ratio), int(500 * ratio))
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        mainWindow.setFont(font)
        mainWindow.setIconSize(QtCore.QSize(24, 24))
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        ''' 左图 '''
        self.left = QtWidgets.QLabel(self.centralwidget)
        self.left.setGeometry(QtCore.QRect(int(60 * ratio), int(40 * ratio), int(310 * ratio), int(360 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.left.setFont(font)
        self.left.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.left.setAutoFillBackground(False)
        self.left.setAlignment(QtCore.Qt.AlignCenter)
        self.left.setObjectName("左图")
        ''' 右图 '''
        self.right = QtWidgets.QLabel(self.centralwidget)
        self.right.setGeometry(QtCore.QRect(int(430 * ratio), int(40 * ratio), int(310 * ratio), int(360 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.right.setFont(font)
        self.right.setAlignment(QtCore.Qt.AlignCenter)
        self.right.setObjectName("右图")
        ''' Load Left Image '''
        self.LoadImage = QtWidgets.QPushButton(self.centralwidget)
        self.LoadImage.setGeometry(QtCore.QRect(int(125 * ratio), int(410 * ratio), int(180 * ratio), int(30 * ratio)))
        self.LoadImage.clicked.connect(lambda: self.load_left_image())  # 添加点击事件
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.LoadImage.setFont(font)
        self.LoadImage.setIconSize(QtCore.QSize(20, 20))
        self.LoadImage.setCheckable(False)
        self.LoadImage.setObjectName("LoadLeftImage")
        ''' Load Right Image '''
        self.SaveImage = QtWidgets.QPushButton(self.centralwidget)
        self.SaveImage.setGeometry(QtCore.QRect(int(495 * ratio), int(410 * ratio), int(180 * ratio), int(30 * ratio)))
        self.SaveImage.clicked.connect(lambda: self.load_right_image())  # 添加点击事件
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.SaveImage.setFont(font)
        self.SaveImage.setIconSize(QtCore.QSize(20, 20))
        self.SaveImage.setCheckable(False)
        self.SaveImage.setObjectName("LoadRightImage")
        mainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, int(800 * ratio), int(24 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.menubar.setFont(font)
        self.menubar.setNativeMenuBar(True)
        self.menubar.setObjectName("menubar")
        self.actionPanorama = QtWidgets.QAction(mainWindow)
        self.actionPanorama.setObjectName("全景拼接")
        self.actionPanorama.triggered.connect(lambda: self.panoramic_image())
        self.menubar.addAction(self.actionPanorama)
        self.actionSave = QtWidgets.QAction(mainWindow)
        self.actionSave.setObjectName("保存图像")
        self.actionSave.triggered.connect(lambda: self.save())
        self.menubar.addAction(self.actionSave)

        ''' 最后设置将菜单条添加至屏幕 '''
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        # 禁用菜单
        self.menubar.setEnabled(False)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "全景拼接"))
        self.left.setText(_translate("mainWindow", "左图"))
        self.right.setText(_translate("mainWindow", "右图"))
        self.LoadImage.setText(_translate("mainWindow", "加载左图"))
        self.SaveImage.setText(_translate("mainWindow", "加载右图"))
        self.actionPanorama.setText(_translate("mainWindow", "进行全景拼接"))
        self.actionSave.setText(_translate("mainWindow", "保存图像"))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.LoadImage.setFont(font)
        self.SaveImage.setFont(font)

    def load_left_image(self):
        """
        加载图像
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            self.statusbar.showMessage(f"Image loaded from {file_path}")
            global left
            left = cv2.imread(file_path)
            current_image = self.fit_image(left)
            current_bytes = current_image.tobytes()
            photo = QtGui.QImage(current_bytes,
                                 current_image.shape[1],
                                 current_image.shape[0],
                                 current_image.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.left.setPixmap(QtGui.QPixmap.fromImage(photo))
            if 'right' in globals():
                # 若左右图像都已经加载，则启用菜单
                self.menubar.setEnabled(True)
        else :
            QMessageBox.warning(None, "Warning", "请先选定一张左图.")

    def load_right_image(self):
        """
        加载图像
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            self.statusbar.showMessage(f"Image loaded from {file_path}")
            global right
            right = cv2.imread(file_path)
            current_image = self.fit_image(right)
            current_bytes = current_image.tobytes()
            photo = QtGui.QImage(current_bytes,
                                 current_image.shape[1],
                                 current_image.shape[0],
                                 current_image.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.right.setPixmap(QtGui.QPixmap.fromImage(photo))
            if 'left' in globals():
                # 若左右图像都已经加载，则启用菜单
                self.menubar.setEnabled(True)
        else :
            QMessageBox.warning(None, "Warning", "请先选定一张右图.")

    def fit_image(self, image):
        """
        将图片调整到合适大小
        :param image: 原图
        :return 调整后的图像
        """
        size = image.shape
        img_w = size[1]  # 宽度
        img_h = size[0]  # 高度
        if img_w != int(310 * ratio):
            size = (int(310 * ratio), int(310 * ratio * img_h / img_w))  # 二元组(宽,高)
            resize_image = cv2.resize(image, size)
        elif img_h != int(360 * ratio):
            size = (int(360 * ratio * img_w / img_h), int(360 * ratio))
            resize_image = cv2.resize(image, size)
        else:
            resize_image = image
        # print(resize_image.shape[1])
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)  # 注意色彩通道对调问题
        return resize_image

    def save(self):
        """
        保存图像
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            panoramic_image = self.panoramic_image()
            cv2.imwrite(file_path, panoramic_image)
            self.statusbar.showMessage(f"Image saved as {file_path}")

    def panoramic_image(self):
        """
        生成全景图,并进行展示
        """
        # 图像拼接
        global select_img
        select_img.insert(0, left)
        select_img.insert(1, right)
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        # image存放需要拼接的图片数组
        status, pano = stitcher.stitch(select_img)
        # 黑边处理
        if status == cv2.Stitcher_OK:
            # 全景图轮廓提取
            stitched = cv2.copyMakeBorder(pano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # 轮廓最小正矩形
            mask = np.zeros(thresh.shape, dtype = "uint8")
            (x, y, w, h) = cv2.boundingRect(cnts[0])  # 取出list中的轮廓二值图，类型为numpy.ndarray
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            # 腐蚀处理，直到minRect的像素值都为0
            minRect = mask.copy()
            sub = mask.copy()
            while cv2.countNonZero(sub) > 0:
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)
            # 提取minRect轮廓并裁剪
            cnts = cv2.findContours(minRect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            (x, y, w, h) = cv2.boundingRect(cnts[0])
            panoramic_image = stitched[y:y + h, x:x + w]
            cv2.imshow("panoramic_image", panoramic_image)
            return panoramic_image
        else:
            QMessageBox.warning(None, "Warning", "图像匹配的特征点不足.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Panorama_Window()
    ui.setupUi(window)

    window.show()
    sys.exit(app.exec_())
