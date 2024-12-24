from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

import Panorama
import Template

global ratio
ratio = 2  # 调节屏幕所占比例

global image
global change_image
select_img = []  # 全局变量，用于存储选择的图片
global flag


class Main_Window(object):
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
        ''' 原图 '''
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(int(60 * ratio), int(40 * ratio), int(310 * ratio), int(360 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("原图")
        ''' 变换后 '''
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(int(430 * ratio), int(40 * ratio), int(310 * ratio), int(360 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("变换后")
        ''' 加载图片 '''
        self.LoadImage = QtWidgets.QPushButton(self.centralwidget)
        self.LoadImage.setGeometry(QtCore.QRect(int(155 * ratio), int(410 * ratio), int(120 * ratio), int(30 * ratio)))
        self.LoadImage.clicked.connect(lambda: self.load_image())  # 添加点击事件
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.LoadImage.setFont(font)
        self.LoadImage.setIconSize(QtCore.QSize(20, 20))
        self.LoadImage.setCheckable(False)
        self.LoadImage.setObjectName("LoadImage")
        ''' 保存图片 '''
        self.SaveImage = QtWidgets.QPushButton(self.centralwidget)
        self.SaveImage.setGeometry(QtCore.QRect(int(525 * ratio), int(410 * ratio), int(120 * ratio), int(30 * ratio)))
        self.SaveImage.clicked.connect(lambda: self.save_image())  # 添加点击事件
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.SaveImage.setFont(font)
        self.SaveImage.setIconSize(QtCore.QSize(20, 20))
        self.SaveImage.setCheckable(False)
        self.SaveImage.setObjectName("SaveImage")
        mainWindow.setCentralWidget(self.centralwidget)
        ''' 图像基本操作 '''
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, int(800 * ratio), int(24 * ratio)))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.menubar.setFont(font)
        self.menubar.setNativeMenuBar(True)
        self.menubar.setObjectName("menubar")
        self.menu1 = QtWidgets.QMenu()
        self.menu1.setObjectName("图像基本操作")
        self.menubar.addMenu(self.menu1)
        self.menu1_1 = QtWidgets.QMenu()
        self.menu1_1.setObjectName("颜色通道提取")
        self.menu1.addMenu(self.menu1_1)
        self.action1_1_1 = QtWidgets.QAction(mainWindow)
        self.action1_1_1.setObjectName("蓝色")
        self.action1_1_1.triggered.connect(lambda: self.extract_color_channel(0))
        self.action1_1_2 = QtWidgets.QAction(mainWindow)
        self.action1_1_2.setObjectName("绿色")
        self.action1_1_2.triggered.connect(lambda: self.extract_color_channel(1))
        self.action1_1_3 = QtWidgets.QAction(mainWindow)
        self.action1_1_3.setObjectName("红色")
        self.action1_1_3.triggered.connect(lambda: self.extract_color_channel(2))
        self.action1_2 = QtWidgets.QAction(mainWindow)
        self.action1_2.setObjectName("边界填充")
        self.action1_2.triggered.connect(lambda: self.pad_image(50))
        self.action1_3 = QtWidgets.QAction(mainWindow)
        self.action1_3.setObjectName("图像融合")
        self.action1_3.triggered.connect(lambda: self.blend_images(0.4))
        self.action1_4 = QtWidgets.QAction(mainWindow)
        self.action1_4.setObjectName("按比例缩放")
        self.action1_4.triggered.connect(lambda: self.resize_image(1, 2))
        self.menu1_5_1 = QtWidgets.QMenu()
        self.menu1_5_1.setObjectName("亮度")
        self.menu1.addMenu(self.menu1_5_1)
        self.action1_5_1 = QtWidgets.QAction(mainWindow)
        self.action1_5_1.setObjectName("增加亮度")
        self.action1_5_1.triggered.connect(lambda: self.increase_brightness())
        self.action1_5_2 = QtWidgets.QAction(mainWindow)
        self.action1_5_2.setObjectName("降低亮度")
        self.action1_5_2.triggered.connect(lambda: self.reduce_brightness())
        self.menu1_5_2 = QtWidgets.QMenu()
        self.menu1_5_2.setObjectName("对比度")
        self.menu1.addMenu(self.menu1_5_2)
        self.action1_5_3 = QtWidgets.QAction(mainWindow)
        self.action1_5_3.setObjectName("增加对比度")
        self.action1_5_3.triggered.connect(lambda: self.increase_contrast())
        self.action1_5_4 = QtWidgets.QAction(mainWindow)
        self.action1_5_4.setObjectName("减少对比度")
        self.action1_5_4.triggered.connect(lambda: self.reduce_contrast())
        self.action1_6 = QtWidgets.QAction(mainWindow)
        self.action1_6.setObjectName("直方图均衡化")
        self.action1_6.triggered.connect(lambda: self.histogram_equalization())
        self.menu1_1.addAction(self.action1_1_1)
        self.menu1_1.addAction(self.action1_1_2)
        self.menu1_1.addAction(self.action1_1_3)
        self.menu1.addAction(self.action1_2)
        self.menu1.addAction(self.action1_3)
        self.menu1.addAction(self.action1_4)
        self.menu1_5_1.addAction(self.action1_5_1)
        self.menu1_5_1.addAction(self.action1_5_2)
        self.menu1_5_2.addAction(self.action1_5_3)
        self.menu1_5_2.addAction(self.action1_5_4)
        self.menu1.addAction(self.action1_6)
        self.menubar.addAction(self.menu1.menuAction())
        ''' 阈值与平滑处理 '''
        self.menu2 = QtWidgets.QMenu()
        self.menu2.setObjectName("阈值与平滑处理")
        self.menubar.addMenu(self.menu2)
        self.menu2_1 = QtWidgets.QMenu()
        self.menu2_1.setObjectName("阈值处理")
        self.menu2.addMenu(self.menu2_1)
        self.action2_1_1 = QtWidgets.QAction(mainWindow)
        self.action2_1_1.setObjectName("二值化")
        self.action2_1_1.triggered.connect(lambda: self.threshold_processing(1))
        self.action2_1_2 = QtWidgets.QAction(mainWindow)
        self.action2_1_2.setObjectName("反二值化")
        self.action2_1_2.triggered.connect(lambda: self.threshold_processing(2))
        self.action2_1_3 = QtWidgets.QAction(mainWindow)
        self.action2_1_3.setObjectName("截断阈值处理")
        self.action2_1_3.triggered.connect(lambda: self.threshold_processing(3))
        self.action2_1_4 = QtWidgets.QAction(mainWindow)
        self.action2_1_4.setObjectName("低阈值零处理")
        self.action2_1_4.triggered.connect(lambda: self.threshold_processing(4))
        self.action2_1_5 = QtWidgets.QAction(mainWindow)
        self.action2_1_5.setObjectName("超阈值零处理")
        self.action2_1_5.triggered.connect(lambda: self.threshold_processing(5))
        self.menu2_1.addAction(self.action2_1_1)
        self.menu2_1.addAction(self.action2_1_2)
        self.menu2_1.addAction(self.action2_1_3)
        self.menu2_1.addAction(self.action2_1_4)
        self.menu2_1.addAction(self.action2_1_5)
        self.menubar.addAction(self.menu2.menuAction())
        self.menu2_2 = QtWidgets.QMenu()
        self.menu2_2.setObjectName("平滑处理")
        self.menu2.addMenu(self.menu2_2)
        self.action2_2_1 = QtWidgets.QAction(mainWindow)
        self.action2_2_1.setObjectName("均值滤波")
        self.action2_2_1.triggered.connect(lambda: self.Smoothing(1))
        self.action2_2_2 = QtWidgets.QAction(mainWindow)
        self.action2_2_2.setObjectName("方框滤波")
        self.action2_2_2.triggered.connect(lambda: self.Smoothing(2))
        self.action2_2_3 = QtWidgets.QAction(mainWindow)
        self.action2_2_3.setObjectName("高斯滤波")
        self.action2_2_3.triggered.connect(lambda: self.Smoothing(3))
        self.action2_2_4 = QtWidgets.QAction(mainWindow)
        self.action2_2_4.setObjectName("中值滤波")
        self.action2_2_4.triggered.connect(lambda: self.Smoothing(4))
        self.menu2_2.addAction(self.action2_2_1)
        self.menu2_2.addAction(self.action2_2_2)
        self.menu2_2.addAction(self.action2_2_3)
        self.menu2_2.addAction(self.action2_2_4)
        self.menubar.addAction(self.menu2.menuAction())
        """ 图像形态学处理 """
        self.menu3 = QtWidgets.QMenu()
        self.menu3.setObjectName("图像形态学处理")
        self.menubar.addMenu(self.menu3)
        self.action3_1 = QtWidgets.QAction(mainWindow)
        self.action3_1.setObjectName("腐蚀")
        self.action3_1.triggered.connect(lambda: self.image_erosion())
        self.menu3_1 = QtWidgets.QMenu()
        self.menu3_1.setObjectName("膨胀")
        self.menu3.addMenu(self.menu3_1)
        self.action3_1_1 = QtWidgets.QAction(mainWindow)
        self.action3_1_1.setObjectName("迭代1次")
        self.action3_1_1.triggered.connect(lambda: self.image_dilate(1))
        self.action3_1_2 = QtWidgets.QAction(mainWindow)
        self.action3_1_2.setObjectName("迭代2次")
        self.action3_1_2.triggered.connect(lambda: self.image_dilate(2))
        self.action3_1_3 = QtWidgets.QAction(mainWindow)
        self.action3_1_3.setObjectName("迭代3次")
        self.action3_1_3.triggered.connect(lambda: self.image_dilate(3))
        self.action3_3 = QtWidgets.QAction(mainWindow)
        self.action3_3.setObjectName("开运算")
        self.action3_3.triggered.connect(lambda: self.image_opening())
        self.action3_4 = QtWidgets.QAction(mainWindow)
        self.action3_4.setObjectName("闭运算")
        self.action3_4.triggered.connect(lambda: self.image_closing())
        self.action3_5 = QtWidgets.QAction(mainWindow)
        self.action3_5.setObjectName("梯度运算")
        self.action3_5.triggered.connect(lambda: self.gradient())
        self.action3_6 = QtWidgets.QAction(mainWindow)
        self.action3_6.setObjectName("礼帽")
        self.action3_6.triggered.connect(lambda: self.tophat())
        self.action3_7 = QtWidgets.QAction(mainWindow)
        self.action3_7.setObjectName("黑帽")
        self.action3_7.triggered.connect(lambda: self.blackhat())
        self.menu3.addAction(self.action3_1)
        self.menu3_1.addAction(self.action3_1_1)
        self.menu3_1.addAction(self.action3_1_2)
        self.menu3_1.addAction(self.action3_1_3)
        self.menu3.addAction(self.action3_3)
        self.menu3.addAction(self.action3_4)
        self.menu3.addAction(self.action3_5)
        self.menu3.addAction(self.action3_6)
        self.menu3.addAction(self.action3_7)
        self.menubar.addAction(self.menu3.menuAction())
        """ 图像梯度处理 """
        self.menu4 = QtWidgets.QMenu()
        self.menu4.setObjectName("图像梯度处理")
        self.menubar.addMenu(self.menu4)
        self.action4_1 = QtWidgets.QAction(mainWindow)
        self.action4_1.setObjectName("Sobel算子")
        self.action4_1.triggered.connect(lambda: self.Sobel())
        self.action4_2 = QtWidgets.QAction(mainWindow)
        self.action4_2.setObjectName("Canny边缘检测")
        self.action4_2.triggered.connect(lambda: self.Canny())
        self.menu4.addAction(self.action4_1)
        self.menu4.addAction(self.action4_2)
        self.menubar.addAction(self.menu4.menuAction())
        """ 图像金字塔与轮廓检测 """
        self.menu5 = QtWidgets.QMenu()
        self.menu5.setObjectName("图像金字塔与轮廓检测")
        self.menubar.addMenu(self.menu5)
        self.menu5_1 = QtWidgets.QMenu()
        self.menu5_1.setObjectName("高斯金字塔")
        self.menu5.addMenu(self.menu5_1)
        self.action5_1_1 = QtWidgets.QAction(mainWindow)
        self.action5_1_1.setObjectName("向下1次")
        self.action5_1_1.triggered.connect(lambda: self.gaussian(0))
        self.action5_1_2 = QtWidgets.QAction(mainWindow)
        self.action5_1_2.setObjectName("向下2次")
        self.action5_1_2.triggered.connect(lambda: self.gaussian(1))
        self.action5_1_3 = QtWidgets.QAction(mainWindow)
        self.action5_1_3.setObjectName("向下3次")
        self.action5_1_3.triggered.connect(lambda: self.gaussian(2))
        self.menu5_2 = QtWidgets.QMenu()
        self.menu5_2.setObjectName("拉普拉斯金字塔")
        self.menu5.addMenu(self.menu5_2)
        self.action5_2_1 = QtWidgets.QAction(mainWindow)
        self.action5_2_1.setObjectName("向上1次")
        self.action5_2_1.triggered.connect(lambda: self.laplacian(0))
        self.action5_2_2 = QtWidgets.QAction(mainWindow)
        self.action5_2_2.setObjectName("向上2次")
        self.action5_2_2.triggered.connect(lambda: self.laplacian(1))
        self.action5_2_3 = QtWidgets.QAction(mainWindow)
        self.action5_2_3.setObjectName("向上3次")
        self.action5_2_3.triggered.connect(lambda: self.laplacian(2))
        self.action5_3 = QtWidgets.QAction(mainWindow)
        self.action5_3.setObjectName("图像轮廓")
        self.action5_3.triggered.connect(lambda: self.image_contour())
        self.menu5_1.addAction(self.action5_1_1)
        self.menu5_1.addAction(self.action5_1_2)
        self.menu5_1.addAction(self.action5_1_3)
        self.menu5_2.addAction(self.action5_2_1)
        self.menu5_2.addAction(self.action5_2_2)
        self.menu5_2.addAction(self.action5_2_3)
        self.menu5.addAction(self.action5_3)
        self.menubar.addAction(self.menu5.menuAction())
        """ 直方图和傅里叶变换 """
        self.menu6 = QtWidgets.QMenu()
        self.menu6.setObjectName("阈值与平滑处理")
        self.menubar.addMenu(self.menu6)
        self.menu6_1 = QtWidgets.QMenu()
        self.menu6_1.setObjectName("直方图")
        self.menu6.addMenu(self.menu6_1)
        self.action6_1_1 = QtWidgets.QAction(mainWindow)
        self.action6_1_1.setObjectName("普通直方图")
        self.action6_1_1.triggered.connect(lambda: self.histogram())
        self.action6_1_2 = QtWidgets.QAction(mainWindow)
        self.action6_1_2.setObjectName("三通道直方图")
        self.action6_1_2.triggered.connect(lambda: self.three_channel_histogram())
        self.action6_2 = QtWidgets.QAction(mainWindow)
        self.action6_2.setObjectName("傅里叶变换")
        self.action6_2.triggered.connect(lambda: self.fft())
        self.menu6_1.addAction(self.action6_1_1)
        self.menu6_1.addAction(self.action6_1_2)
        self.menu6.addAction(self.action6_2)
        self.menubar.addAction(self.menu6.menuAction())
        """ 图像特征 """
        self.menu7 = QtWidgets.QMenu()
        self.menu7.setObjectName("图像特征")
        self.menubar.addMenu(self.menu7)
        """ harris特征 """
        self.action7_1 = QtWidgets.QAction(mainWindow)
        self.action7_1.setObjectName("harris特征")
        self.action7_1.triggered.connect(lambda: self.harris())
        """ SIFT """
        self.action7_2 = QtWidgets.QAction(mainWindow)
        self.action7_2.setObjectName("SIFT")
        self.action7_2.triggered.connect(lambda: self.SIFT())
        self.menu7.addAction(self.action7_1)
        self.menu7.addAction(self.action7_2)
        self.menubar.addAction(self.menu7.menuAction())
        """ 全景图像拼接 """
        self.action9 = QtWidgets.QAction(mainWindow)
        self.action9.setObjectName("跳转全景拼接")
        self.action9.triggered.connect(lambda: self.new_Panorama())
        self.menubar.addAction(self.action9)
        """ 模板匹配 """
        self.action11 = QtWidgets.QAction(mainWindow)
        self.action11.setObjectName("跳转模版匹配")
        self.action11.triggered.connect(lambda: self.new_Template())
        self.menubar.addAction(self.action11)

        ''' 最后设置将菜单条添加至屏幕 '''
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        # 禁用特定功能
        self.menu1.setEnabled(False)
        self.menu2.setEnabled(False)
        self.menu3.setEnabled(False)
        self.menu4.setEnabled(False)
        self.menu5.setEnabled(False)
        self.menu6.setEnabled(False)
        self.menu7.setEnabled(False)

        # 启用特定功能
        self.action9.setEnabled(True)
        self.action11.setEnabled(True)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "图像处理"))
        self.label.setText(_translate("mainWindow", "原图"))
        self.label_2.setText(_translate("mainWindow", "变换后的图片"))
        self.LoadImage.setText(_translate("mainWindow", "加载图片"))
        self.SaveImage.setText(_translate("mainWindow", "保存图片"))
        self.menu1.setTitle(_translate("mainWindow", "基本操作"))
        self.menu1_1.setTitle(_translate("mainWindow", "颜色通道提取"))
        self.action1_1_1.setText(_translate("mainWindow", "蓝色"))
        self.action1_1_2.setText(_translate("mainWindow", "绿色"))
        self.action1_1_3.setText(_translate("mainWindow", "红色"))
        self.action1_2.setText(_translate("mainWindow", "边界填充"))
        self.action1_3.setText(_translate("mainWindow", "图像融合"))
        self.action1_4.setText(_translate("mainWindow", "按比例缩放"))
        self.menu1_5_1.setTitle(_translate("mainWindow", "亮度"))
        self.action1_5_1.setText(_translate("mainWindow", "增加"))
        self.action1_5_2.setText(_translate("mainWindow", "减低"))
        self.menu1_5_2.setTitle(_translate("mainWindow", "对比度"))
        self.action1_5_3.setText(_translate("mainWindow", "增加"))
        self.action1_5_4.setText(_translate("mainWindow", "减低"))
        self.action1_6.setText(_translate("mainWindow", "直方图均衡化"))
        self.menu2.setTitle(_translate("mainWindow", "阈值与平滑"))
        self.menu2_1.setTitle(_translate("mainWindow", "阈值处理"))
        self.action2_1_1.setText(_translate("mainWindow", "二值化"))
        self.action2_1_2.setText(_translate("mainWindow", "反二值化"))
        self.action2_1_3.setText(_translate("mainWindow", "截断阈值处理"))
        self.action2_1_4.setText(_translate("mainWindow", "低阈值零处理"))
        self.action2_1_5.setText(_translate("mainWindow", "超阈值零处理"))
        self.menu2_2.setTitle(_translate("mainWindow", "图像平滑"))
        self.action2_2_1.setText(_translate("mainWindow", "均值滤波"))
        self.action2_2_2.setText(_translate("mainWindow", "方框滤波"))
        self.action2_2_3.setText(_translate("mainWindow", "高斯滤波"))
        self.action2_2_4.setText(_translate("mainWindow", "中值滤波"))
        self.menu3.setTitle(_translate("mainWindow", "形态学"))
        self.action3_1.setText(_translate("mainWindow", "腐蚀"))
        self.menu3_1.setTitle(_translate("mainWindow", "膨胀"))
        self.action3_1_1.setText(_translate("mainWindow", "迭代1次"))
        self.action3_1_2.setText(_translate("mainWindow", "迭代2次"))
        self.action3_1_3.setText(_translate("mainWindow", "迭代3次"))
        self.action3_3.setText(_translate("mainWindow", "开运算"))
        self.action3_4.setText(_translate("mainWindow", "闭运算"))
        self.action3_5.setText(_translate("mainWindow", "梯度运算"))
        self.action3_6.setText(_translate("mainWindow", "礼帽"))
        self.action3_7.setText(_translate("mainWindow", "黑帽"))
        self.menu4.setTitle(_translate("mainWindow", "梯度"))
        self.action4_1.setText(_translate("mainWindow", "laplacian算子"))
        self.action4_2.setText(_translate("mainWindow", "Canny边缘检测"))
        self.menu5.setTitle(_translate("mainWindow", "金字塔与轮廓"))
        self.menu5_1.setTitle(_translate("mainWindow", "高斯金字塔"))
        self.action5_1_1.setText(_translate("mainWindow", "向下1次"))
        self.action5_1_2.setText(_translate("mainWindow", "向下2次"))
        self.action5_1_3.setText(_translate("mainWindow", "向下3次"))
        self.menu5_2.setTitle(_translate("mainWindow", "拉普拉斯金字塔"))
        self.action5_2_1.setText(_translate("mainWindow", "向上1次"))
        self.action5_2_2.setText(_translate("mainWindow", "向上2次"))
        self.action5_2_3.setText(_translate("mainWindow", "向上3次"))
        self.action5_3.setText(_translate("mainWindow", "图像轮廓"))
        self.menu6.setTitle(_translate("mainWindow", "直方图和傅里叶变换"))
        self.menu6_1.setTitle(_translate("mainWindow", "直方图"))
        self.action6_1_1.setText(_translate("mainWindow", "普通直方图"))
        self.action6_1_2.setText(_translate("mainWindow", "三通道直方图"))
        self.action6_2.setText(_translate("mainWindow", "傅里叶变换"))
        self.menu7.setTitle(_translate("mainWindow", "图像特征"))
        self.action7_1.setText(_translate("mainWindow", "harris特征"))
        self.action7_2.setText(_translate("mainWindow", "SIFT"))
        self.action9.setText(_translate("mainWindow", "全景图像拼接"))
        self.action11.setText(_translate("mainWindow", "模板匹配"))

    def load_image(self):
        """
        加载图像
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Image", "examples\\",
                                                   "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            self.statusbar.showMessage(f"Image loaded from {file_path}")
            global image
            image = cv2.imread(file_path)
            current_image = self.fit_image(image)

            height, width = image.shape[:2]
            resolution_text = f"Resolution: {width}x{height}"
            # 设置字体和文本属性
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # 白色
            font_thickness = 2
            # 获取文本的大小
            text_size, _ = cv2.getTextSize(resolution_text, font, font_scale, font_thickness)
            # 计算文本的位置（右下角）
            height_new, width_new = current_image.shape[:2]
            text_x = width_new - text_size[0] - 10  # 10像素的边距
            text_y = height_new - 10  # 10像素的边距
            # 在图像上绘制文本
            cv2.putText(current_image, resolution_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            current_bytes = current_image.tobytes()
            photo = QtGui.QImage(current_bytes,
                                 current_image.shape[1],
                                 current_image.shape[0],
                                 current_image.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(photo))
            # 启用相关功能
            self.menu1.setEnabled(True)
            self.menu2.setEnabled(True)
            self.menu3.setEnabled(True)
            self.menu4.setEnabled(True)
            self.menu5.setEnabled(True)
            self.menu6.setEnabled(True)
            self.menu7.setEnabled(True)
        else:
            QMessageBox.warning(None, "Warning", "请选择一个图片进行操作.")

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

    def change_image(self, new_image):
        """
        将变换后的图像加载到label_2
        :param new_image: 变换后的图像
        """
        global change_image
        change_image = new_image
        new_image = self.fit_image(new_image)
        new_image = self.write_information(new_image)
        new_bytes = new_image.tobytes()
        photo = QtGui.QImage(new_bytes,
                             new_image.shape[1],
                             new_image.shape[0],
                             new_image.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(photo))

    def write_information(self, new_image):
        """
        在图像右下角添加分辨率信息
        """
        # 获取图像的分辨率
        height, width = change_image.shape[:2]
        resolution_text = f"Resolution: {width}x{height}"
        # 设置字体和文本属性
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # 白色
        font_thickness = 2
        # 获取文本的大小
        text_size, _ = cv2.getTextSize(resolution_text, font, font_scale, font_thickness)
        # 计算文本的位置（右下角）
        height_new, width_new = new_image.shape[:2]
        text_x = width_new - text_size[0] - 10  # 10像素的边距
        text_y = height_new - 10  # 10像素的边距
        # 在图像上绘制文本
        cv2.putText(new_image, resolution_text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        return new_image

    def save_image(self):
        """
        保存图像
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Image", "output\\",
                                                   "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            global change_image
            change_image = cv2.imwrite(file_path, change_image)
            self.statusbar.showMessage(f"Image saved as {file_path}")

    def extract_color_channel(self, channel):
        """
        提取指定颜色通道的图像
        :param channel: 通道索引 (0 - Blue, 1 - Green, 2 - Red)
        """

        zeros = np.zeros(image.shape[:2], dtype="uint8")
        # 蓝色
        if channel == 0:
            channel_image = cv2.merge([image[:, :, 0], zeros, zeros])
        # 绿色
        elif channel == 1:
            channel_image = cv2.merge([zeros, image[:, :, 1], zeros])
        # 红色
        elif channel == 2:
            channel_image = cv2.merge([zeros, zeros, image[:, :, 2]])
        else:
            raise ValueError("Invalid channel index. Expected 0, 1, or 2.")
        self.change_image(channel_image)


    def pad_image(self, pad_size):
        """
        对图像进行边界填充
        :param pad_size: 填充大小
        """
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
        self.change_image(padded_image)

    def blend_images(self, alpha):
        """
        将两张图像混合在一起
        :param image2: 第二张图像
        :param alpha: 混合权重
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            self.statusbar.showMessage(f"Image loaded from {file_path}")
            image2 = cv2.imread(file_path)
            # 将第二张图像的尺寸调整为和第一张图像一样
            image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
            blended_image = cv2.addWeighted(image, alpha, image2, 1 - alpha, 0)
            self.change_image(blended_image)
        else:
            QMessageBox.warning(None, "Warning", "请选择用于融合的图像.")

    def resize_image(self, scale_fx, scale_fy):
        """
        按比例缩放图像
        :param scale_fx: 横向缩放因子
        :param scale_fy: 纵向缩放因子
        """
        resized_image = cv2.resize(image, None, fx=scale_fx, fy=scale_fy, interpolation=cv2.INTER_LINEAR)
        self.change_image(resized_image)

    def increase_brightness(self):
        """
        增加亮度
        """
        # 创建一个与调整后的图像一样大小的空白图像
        blank = np.zeros_like(image)
        blank[:, :] = (50, 50, 50)
        # 将调整后的图像和空白图像相加即可增加亮度
        adjusted_image = cv2.add(image, blank)
        self.change_image(adjusted_image)

    def reduce_brightness(self):
        """
        降低亮度
        """
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(image)
        blank[:, :] = (50, 50, 50)
        # 将原图像和空白图像相减即可减小亮度
        adjusted_image = cv2.subtract(image, blank)
        self.change_image(adjusted_image)

    def increase_contrast(self):
        """
        增加对比度
        """
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(image)
        blank[:, :] = (2, 2, 2)  #
        # # 将原图像和空白图像相乘即可增加对比度
        adjusted_image = cv2.multiply(image, blank)
        self.change_image(adjusted_image)

    def reduce_contrast(self):
        """
        降低对比度
        """
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(image)
        blank[:, :] = (2, 2, 2)  # bgr 分别为2，即为图像对比度比例
        # 将原图像和空白图像相除即可减小对比度
        adjusted_image = cv2.divide(image, blank)
        self.change_image(adjusted_image)

    def histogram_equalization(self):
        """
        对灰度图像进行直方图均衡化
        """
        # 彩色图像直方图均衡化
        (b, g, r) = cv2.split(image)  # 通道分解
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        equalized_image = cv2.merge((bH, gH, rH))  # 通道合成
        # 将彩色图像转换为RGB格式
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
        self.change_image(equalized_image)

    def threshold_processing(self, type):
        """
        阈值处理：剔除图像内像素值高于一定值或者低于一定值的像素点
        :param type: 阈值处理方式 ( 1 - 二值化, 2 - 反二值化, 3 - 截断阈值处理, 4 - 低阈值零处理, 5 - 超阈值零处理 )
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 要二值化图像，必须先将图像转为灰度图
        if type == 1:
            ret, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif type == 2:
            ret, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        elif type == 3:
            ret, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        elif type == 4:
            ret, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
        elif type == 5:
            ret, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
        self.change_image(thresh_image)

    def Smoothing(self, type):
        """
        平滑处理
        :param type: 平滑处理方式 ( 1 - 均值滤波, 2 - 方框滤波, 3 - 高斯滤波, 4 - 中值滤波 )
        """
        if type == 1:
            # 简单的平均卷积操作（卷积核：9个全1的矩阵）
            Smooth_image = cv2.blur(image, (3, 3))
        elif type == 2:
            # 基本和均值一样，可以选择归一化,容易越界
            Smooth_image = cv2.boxFilter(image, -1, (3, 3), normalize=False)
        elif type == 3:
            # 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
            Smooth_image = cv2.GaussianBlur(image, (5, 5), 1)
        elif type == 4:
            # 相当于用中间值代替，平滑(去噪)效果最好
            Smooth_image = cv2.medianBlur(image, 5)
        else:
            raise ValueError("Invalid type index. Expected 1, 2, 3 or 4.")
        self.change_image(Smooth_image)

    def image_erosion(self):
        """
        图像腐蚀
        """
        kernel = np.ones((3, 3), np.uint8)
        erosion_image = cv2.erode(image, kernel, iterations=1)
        self.change_image(erosion_image)

    def image_dilate(self, iterations):
        """
        图像膨胀
        """
        kernel = np.ones((30, 30), np.uint8)
        # iterations表示迭代次数。默认是1，表示进行一次腐蚀，也可以根据需要进行多次迭代，进行多次腐蚀。
        if iterations == 1:
            dilate_image = cv2.dilate(image, kernel, iterations=1)
        elif iterations == 2:
            dilate_image = cv2.dilate(image, kernel, iterations=2)
        elif iterations == 3:
            dilate_image = cv2.dilate(image, kernel, iterations=3)
        else:
            raise ValueError("Invalid type index. Expected 1, 2, or 3.")
        self.change_image(dilate_image)

    def image_opening(self):
        """
        开运算
        """
        # 开运算：先腐蚀，再膨胀（去毛刺，再回复粗度）
        kernel = np.ones((5, 5), np.uint8)
        opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        self.change_image(opening_image)

    def image_closing(self):
        """
        闭运算
        """
        # 闭运算：先膨胀，再腐蚀（不能去毛刺）
        kernel = np.ones((5, 5), np.uint8)
        closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        self.change_image(closing_image)

    def gradient(self):
        """
        梯度运算
        """
        kernel = np.ones((5, 5), np.uint8)
        gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        self.change_image(gradient_image)

    def tophat(self):
        """
        礼帽
        """
        kernel = np.ones((5, 5), np.uint8)
        tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        self.change_image(tophat_image)

    def blackhat(self):
        """
        黑帽
        """
        kernel = np.ones((5, 5), np.uint8)
        blackhat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        self.change_image(blackhat_image)

    def Sobel(self):
        """
        Sobel算子 ( 分别计算x、y方向梯度，再合并 )
        """
        # 计算x方向梯度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        # 计算y方向梯度
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        # 合并
        sobelxy_image = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        self.change_image(sobelxy_image)

    def Canny(self):
        """
        边缘检测
        """
        # 将图像转为灰度图像,图像数据类型转换为uint8，即8位无符号整型图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # 进行Canny边缘检测
        Canny_image = cv2.Canny(gray_image, 50, 180)
        self.change_image(Canny_image)

    def resize_to_power_of_two(self):
        """
        将图像裁剪为2的幂次方
        :return: 裁剪后的图像
        """
        height, width = image.shape[:2]
        new_height = int(2 ** np.ceil(np.log2(height)))
        new_width = int(2 ** np.ceil(np.log2(width)))
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def gaussian(self, times):
        """
        高斯金字塔
        """
        resize_image = self.resize_to_power_of_two()
        pyramid = []
        pyramid.append(resize_image)
        gaussian_image = resize_image
        for _ in range(3):
            gaussian_image = cv2.pyrDown(gaussian_image)
            pyramid.append(gaussian_image)
        if times == 0:
            transformed_image = pyramid[times + 1]
        elif times == 1:
            transformed_image = pyramid[times + 1]
        elif times == 2:
            transformed_image = pyramid[times + 1]
        self.change_image(transformed_image)
        return pyramid

    def laplacian(self, times):
        """
        拉普拉斯金字塔
        """
        gaussian_pyramid = self.gaussian(times)
        pyramid = []
        for i in range(len(gaussian_pyramid) - 1, 0, -1):
            expanded = cv2.pyrUp(gaussian_pyramid[i])
            laplacian = gaussian_pyramid[i - 1] - expanded
            pyramid.append(laplacian)
        if times == 0:
            transformed_image = pyramid[times]
        elif times == 1:
            transformed_image = pyramid[times]
        elif times == 2:
            transformed_image = pyramid[times]
        self.change_image(transformed_image)

    def image_contour(self):
        """
        图像轮廓
        """
        # 将图像转为灰度图像,图像数据类型转换为uint8，即8位无符号整型图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # 进行Canny边缘检测
        Canny_image = cv2.Canny(gray_image, 50, 180)
        # cv2.CHAIN_APPROX_SIMPLE只存储端点
        contours, hierarchy = cv2.findContours(Canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # -1代表绘制所有轮廓
        image1 = image.copy()
        img1 = cv2.drawContours(image1, contours, -1, (0, 0, 255), 2)
        self.change_image(img1)

    def histogram(self):
        """
        直方图
        """
        image1 = image.copy()
        hist = cv2.calcHist([image1], [0], None, [256], [0, 256])  # (256, 1)
        plt.hist(image1.ravel(), 256);
        fig = plt.gcf()  # 获取当前的图像框对象
        fig.canvas.draw()  # 绘制图像框
        image_data = np.array(fig.canvas.renderer.buffer_rgba())  # 将图像框转换为图像数据
        self.change_image(image_data)
        # 展示后清空图像框，避免错误
        plt.clf()

    def three_channel_histogram(self):
        """
        三通道直方图
        """
        color = ('b', 'g', 'r')
        image1 = image.copy()
        for i, col in enumerate(color):
            histr = cv2.calcHist([image1], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        fig = plt.gcf()  # 获取当前的图像框对象
        fig.canvas.draw()  # 绘制图像框
        image_data = np.array(fig.canvas.renderer.buffer_rgba())  # 将图像框转换为图像数据
        self.change_image(image_data)
        # 展示后清空图像框，避免错误
        plt.clf()

    def fft(self):
        """
        傅里叶变换
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        img_float32 = np.float32(gray_image)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # 得到灰度图能表示的形式
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        plt.imshow(magnitude_spectrum, cmap='gray')
        fig = plt.gcf()  # 获取当前的图像框对象
        fig.canvas.draw()  # 绘制图像框
        image_data = np.array(fig.canvas.renderer.buffer_rgba())  # 将图像框转换为图像数据
        self.change_image(image_data)
        # 展示后清空图像框，避免错误
        plt.clf()

    def harris(self):
        """
        harris特征
        """
        image1 = image.copy()
        h, w, c = image1.shape
        # harris dst
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, blockSize=3, ksize=5, k=0.05)
        image_dst = image1[:, :, :]
        image_dst[dst > 0.01 * dst.max()] = [0, 0, 255]
        self.change_image(image_dst)

    def SIFT(self):
        """
        SIFT特征
        """
        # 设置随机种子
        np.random.seed(0)
        image1 = image.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # 实例化的sift函数 得到特征点
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        # 显示特征点
        img = cv2.drawKeypoints(gray, kp, image1)
        self.change_image(img)

    def new_Panorama(self):
        new_Panorama = Panorama.Panorama_Window()
        new_Panorama.show()

    def new_Template(self):
        new_Template = Template.Template_Window()
        new_Template.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    new_Main = QMainWindow()
    ui = Main_Window()
    ui.setupUi(new_Main)

    new_Main.show()
    sys.exit(app.exec_())
