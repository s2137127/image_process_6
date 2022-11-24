import math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from mainui import Ui_MainWindow
import cv2
import numpy as np
import time
import pywt


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.wave1 = None
        self.wave2 = None
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_Clicked)
        self.pushButton_trans.clicked.connect(self.Wavy)
        self.pushButton_wave2.clicked.connect(self.pushbottom_wave2_clicked)
        self.spinBox.valueChanged.connect(self.fusion)
        self.img_list = []
    def pushButton_Clicked(self):
        # 開啟資料夾選則照片
        self.img_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open file",
                                                       "./",
                                                       "Images (*.png *.BMP *.jpg *.JPG)")

        self.img = cv2.imread(self.img_path)  # 讀檔

        height, width, channel = self.img.shape
        qimg = QImage(self.img, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.label.setScaledContents(True)

    def Trapezoidal(self):
        mid = [200, 256]
        short = 80
        long = 256
        high = 200
        src_point = np.array(
            [[0, 0], [self.img.shape[1], 0], [self.img.shape[1], self.img.shape[0]], [0, self.img.shape[0]]])
        target_point = np.array(
            [[mid[1] - long, mid[0] - high], [mid[1] + long, mid[0] - high], [mid[1] + short, mid[0] + high],
             [mid[1] - short, mid[0] + high]])
        h, status = cv2.findHomography(src_point, target_point, method=cv2.RANSAC)
        im_out = cv2.warpPerspective(self.img.copy(), h, (512, 512))
        # im_out.transpose(1,0,2)
        qimg = QImage(im_out.data, im_out.shape[1], im_out.shape[0], 3 * im_out.shape[1],
                      QImage.Format_RGB888).rgbSwapped()
        self.label_geo.setPixmap(QPixmap.fromImage(qimg))
        self.label_geo.setScaledContents(True)

    def Wavy(self):
        img_output = np.zeros(self.img.shape, dtype=self.img.dtype)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < self.img.shape[0] and j + offset_x < self.img.shape[1]:
                    img_output[i, j] = self.img[(i + offset_y) % self.img.shape[0], (j + offset_x) % self.img.shape[1]]
                else:
                    img_output[i, j] = 0
        qimg = QImage(img_output.data, img_output.shape[1], img_output.shape[0], 3 * img_output.shape[1],
                      QImage.Format_RGB888).rgbSwapped()
        self.label_geo.setPixmap(QPixmap.fromImage(qimg))
        self.label_geo.setScaledContents(True)

    def pushbottom_wave2_clicked(self):

        img_path, _ = QFileDialog.getOpenFileNames(self,
                                                  "Open file",
                                                  "./",
                                                  "Images (*.png *.BMP *.jpg *.JPG)")

        img_list = [cv2.imread(i) for i in img_path]
        self.img_list = [i[:, :, 0].copy().squeeze() for i in img_list]
        qimg = QImage( self.img_list[0].data, self.img_list[0].shape[1], self.img_list[0].shape[0], QImage.Format_Grayscale8)
        self.label_wave1.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave1.setScaledContents(True)
        qimg = QImage(self.img_list[1].data, self.img_list[0].shape[1], self.img_list[0].shape[0],
                      QImage.Format_Grayscale8)
        self.label_wave2.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave2.setScaledContents(True)
        if len(self.img_list) >2:
            qimg = QImage(self.img_list[2].data, self.img_list[0].shape[1], self.img_list[0].shape[0],
                          QImage.Format_Grayscale8)
            self.label_wave3.setPixmap(QPixmap.fromImage(qimg))
            self.label_wave3.setScaledContents(True)
        self.fusion()

    def fusion(self):
        if self.img_list is not None and len(self.img_list)>=2:
            scale = self.spinBox.value()
            wavelet = 'db1'
            cooef = [pywt.wavedec2(img.copy(), wavelet,level = scale) for img in self.img_list]

            fusedCooef = []

            for i in range(len(cooef[0]) - 1):
                # The first values in each decomposition is the apprximation values of the top level
                if i == 0:
                    # print("1")
                    tmp = np.array([cooef[j][0] for j in range(len(cooef))]).squeeze().squeeze().tolist()

                    fusedCooef.append(np.max(tmp,axis=0))

                else:

                    tmp = np.array([cooef[j][i] for j in range(len(cooef))]).squeeze()
                    # print(cooef[j][i])
                    # print(t mp)
                    tmp = np.max(tmp,axis=0).squeeze().tolist()

                    fusedCooef.append(tmp)

            fusedImage = pywt.waverec2(fusedCooef, wavelet)
            fusedImage *=255/np.max(fusedImage)
            fusedImage = fusedImage.astype(np.uint8)
        else:
            return 0

        qimg = QImage(fusedImage.data, fusedImage.shape[1], fusedImage.shape[0], QImage.Format_Grayscale8)
        self.label_wave_out.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave_out.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
