import math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from mainui import Ui_MainWindow
import cv2
import numpy as np
import pywt
from numpy.linalg import inv
from math import dist
from os.path import basename
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.best = None
        self.wave1 = None
        self.wave2 = None
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pushButton_Clicked)
        self.pushButton_Wavy.clicked.connect(self.Wavy_clicked)
        self.pushButton_wave2.clicked.connect(self.pushbottom_wave2_clicked)
        self.spinBox.valueChanged.connect(self.fusion)
        self.pushButton_edge.clicked.connect(self.pushbottom_edge_clicked)
        self.pushButton_Trapezoidal.clicked.connect(self.Trapezoidal_clicked)
        self.pushButton_Circular.clicked.connect(self.pushButton_Circular_clicked)
        self.img_list = []

    def pushButton_Clicked(self):
        # 開啟資料夾選則照片
        self.img_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open file",
                                                       "./",
                                                       "Images (*.png *.BMP *.jpg *.JPG)")

        self.img = cv2.imread(self.img_path)  # 讀檔
        img = self.img.copy()
        # self.img = self.img[:,:,0]
        self.img = np.mean(self.img[:, :, :], axis=2).copy().astype(np.uint8)
        height, width, channel = img.shape
        qimg = QImage(img, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.label.setScaledContents(True)

    def Trapezoidal_clicked(self):
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
        qimg = QImage(im_out.data, im_out.shape[1], im_out.shape[0], im_out.shape[1],
                      QImage.Format_Grayscale8)
        self.label_geo.setPixmap(QPixmap.fromImage(qimg))
        self.label_geo.setScaledContents(True)

    def Wavy_clicked(self):
        img_output = np.zeros(self.img.shape, dtype=self.img.dtype)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < self.img.shape[0] and j + offset_x < self.img.shape[1]:
                    img_output[i, j] = self.img[(i + offset_y) % self.img.shape[0], (j + offset_x) % self.img.shape[1]]
                else:
                    img_output[i, j] = 0
        qimg = QImage(img_output.data, img_output.shape[1], img_output.shape[0],  img_output.shape[1],
                      QImage.Format_Grayscale8)
        self.label_geo.setPixmap(QPixmap.fromImage(qimg))
        self.label_geo.setScaledContents(True)

    def _pixel_coordinates_to_unit(self,coordinate, max_value):#影像座標轉成單位長度座標
        return coordinate / max_value * 2 - 1

    def _one_coordinates_to_pixels(self,coordinate, max_value):#單位長度座標轉影像座標
        return (coordinate + 1) / 2 * max_value

    def _elliptical_square_to_disc(self,u, v):#方形座標轉圓形的座標轉化
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2

        if r2 > 1:  # we're outside the disc
            return

        twosqrt2 = 2.0 * math.sqrt(2.0)
        subtermx = 2.0 + u2 - v2
        subtermy = 2.0 - u2 + v2
        termx1 = subtermx + u * twosqrt2
        termx2 = subtermx - u * twosqrt2
        termy1 = subtermy + v * twosqrt2
        termy2 = subtermy - v * twosqrt2
        x = 0.5 * math.sqrt(termx1) - 0.5 * math.sqrt(termx2)
        y = 0.5 * math.sqrt(termy1) - 0.5 * math.sqrt(termy2)
        return x, y
    def _transform(self,inp):#將方形影像map到圓形影像
        result = np.zeros_like(inp)
        for x, row in enumerate(inp):

            unit_x = self._pixel_coordinates_to_unit(x, len(inp))
            for y, _ in enumerate(row):
                # print("1")
                unit_y = self._pixel_coordinates_to_unit(y, len(row))
                try:
                    uv = self._elliptical_square_to_disc(unit_x, unit_y)
                    if uv is None:
                        continue
                    u, v = uv
                    u = self._one_coordinates_to_pixels(u, len(inp))
                    v = self._one_coordinates_to_pixels(v, len(row))

                    result[x][y] = inp[math.floor(u)][math.floor(v)]
                except IndexError:
                    pass

        return result

    def pushButton_Circular_clicked(self):
        # img = np.mean(self.img[:,:,:],axis=2).copy()

        new = self._transform(self.img)

        new = new.astype(np.uint8)
        qimg = QImage(new.data, new.shape[1], new.shape[0], QImage.Format_Grayscale8)
        self.label_geo.setPixmap(QPixmap.fromImage(qimg))
        self.label_geo.setScaledContents(True)
    def pushbottom_wave2_clicked(self):

        img_path, _ = QFileDialog.getOpenFileNames(self,
                                                   "Open file",
                                                   "./",
                                                   "Images (*.png *.BMP *.jpg *.JPG)")

        img_list = [cv2.imread(i) for i in img_path]
        self.img_list = [i[:, :, 0].copy().squeeze() for i in img_list]
        qimg = QImage(self.img_list[0].data, self.img_list[0].shape[1], self.img_list[0].shape[0],
                      QImage.Format_Grayscale8)
        self.label_wave1.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave1.setScaledContents(True)
        qimg = QImage(self.img_list[1].data, self.img_list[0].shape[1], self.img_list[0].shape[0],
                      QImage.Format_Grayscale8)
        self.label_wave2.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave2.setScaledContents(True)
        if len(self.img_list) > 2:
            qimg = QImage(self.img_list[2].data, self.img_list[0].shape[1], self.img_list[0].shape[0],
                          QImage.Format_Grayscale8)
            self.label_wave3.setPixmap(QPixmap.fromImage(qimg))
            self.label_wave3.setScaledContents(True)
        self.fusion()

    def fusion(self):
        if self.img_list is not None and len(self.img_list) >= 2:
            scale = self.spinBox.value()
            wavelet = 'db1'
            cooef = [pywt.wavedec2(img.copy(), wavelet, level=scale) for img in self.img_list]#取各項的係數

            fusedCooef = []

            for i in range(len(cooef[0]) - 1):
                if i == 0:#估計值
                    tmp = np.array([cooef[j][0] for j in range(len(cooef))]).squeeze().squeeze().tolist()

                    fusedCooef.append(np.max(tmp, axis=0))

                else:#取及大值作為新影像的係數
                    tmp = np.array([cooef[j][i] for j in range(len(cooef))]).squeeze()
                    tmp = np.max(tmp, axis=0).squeeze().tolist()
                    fusedCooef.append(tmp)
            # print(fusedCooef)
            fusedImage = pywt.waverec2(fusedCooef, wavelet)#轉為合成影像
            # print(fusedImage)
            fusedImage *= 255 / np.max(fusedImage)
            fusedImage = fusedImage.astype(np.uint8)
        else:
            return 0
        qimg = QImage(fusedImage.data, fusedImage.shape[1], fusedImage.shape[0], QImage.Format_Grayscale8)
        self.label_wave_out.setPixmap(QPixmap.fromImage(qimg))
        self.label_wave_out.setScaledContents(True)
    def get_rec_point(self,edge_image,rec_idx):#取頂底座標
        thetas = np.arange(0, 180, step=1)
        d = np.sqrt(np.square(edge_image.shape[0]) + np.square(edge_image.shape[1]))
        drho = (2 * d) / 180
        rhos = np.arange(-d, d, step=drho)
        edge_height, edge_width = edge_image.shape[:2]
        edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
        point_arr = []
        for i in range(len(rec_idx)):
            idx = rec_idx[i]
            y, x = self.best[idx]
            rho = rhos[y]
            theta = thetas[x]
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))

            x0 = (a * rho) + edge_width_half
            y0 = (b * rho) + edge_height_half
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            m = (y1 - y0) / (x1 - x0)
            # print(x0)
            mat0 = np.array([m * x0 - y0])
            mat1 = np.array([m, -1])
            idx = rec_idx[(i + 1) % 4]
            y, x = self.best[idx]
            rho = rhos[y]
            theta = thetas[x]
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))

            x0 = (a * rho) + edge_width_half
            y0 = (b * rho) + edge_height_half
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            m = (y1 - y0) / (x1 - x0)
            #將兩線的交點求出
            mat0_ = np.array([m * x0 - y0])
            mat1_ = np.array([m, -1])
            mat0 = np.concatenate([mat0, mat0_]).reshape(2, 1)
            mat1 = np.concatenate([mat1, mat1_]).reshape(2, 2)

            p = np.dot(inv(mat1), mat0).squeeze()
            p = p.astype(int)
            p = (p[0], p[1])
            point_arr.append(p)
        return point_arr
    def get_area(self,point_arr):#利用頂點求面積
        x = np.array([point_arr[i][0] for i in range(4)])
        y = np.array([point_arr[i][1] for i in range(4)])
        i = np.arange(len(x))
        Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5) * 0.25
        return Area*0.25
    def get_area_perimeter(self,image):#求周長&面積
        thetas = np.arange(0, 180, step=1)
        angle = []
        for line in self.best:
            y, x = line
            theta = thetas[x]
            a = np.tan(np.deg2rad(theta))
            angle.append(a)
        rec1 = []
        rec1_idx = []
        #若兩線的夾角接近90度則將兩線視為同個方形的兩個邊，並將其中一個方形的四邊index找出
        for j in range(len(angle)):
            if angle[0] * angle[j] < -0.6 and angle[0] * angle[j] > -1.1:
                rec1.append(angle[0])
                rec1_idx.append(0)
                rec1.append(angle[j])
                rec1_idx.append(j)
                break
        for i in range(1,3):
            for j in range(len(angle)):
                if j in rec1_idx:
                    continue
                # print("22")
                if rec1[i] * angle[j]<-0.7 and rec1[i] * angle[j] >-1.1:
                    rec1.append(angle[j])
                    rec1_idx.append(j)
        #求出第二個方形的四邊的index
        rec2_idx = [i for i in range(len(self.best)) if i not in rec1_idx]
        point1_arr = self.get_rec_point(image,rec1_idx)
        point2_arr = self.get_rec_point(image, rec2_idx)
        #get perimeter
        perimeter1 = 0
        perimeter2 = 0
        print("1")
        for i in range(4):
            perimeter1 += dist(point1_arr[i],point1_arr[(i+1)%4])
        perimeter1 *= 0.5
        for i in range(4):
            perimeter2 += dist(point2_arr[i],point2_arr[(i+1)%4])
        perimeter2 *= 0.5
        #get area
        Area1 = self.get_area(point1_arr)
        Area2 = self.get_area(point2_arr)
        org = np.mean(point1_arr,axis = 0).astype(int)
        org[0] -= 100
        org[1] -= 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        #put rec informations in image
        cv2.putText(image,"Area: %.2f" % Area1,org,font,fontScale,color,thickness,cv2.LINE_AA)
        org[1] += 50
        cv2.putText(image, "perimeter: %.2f" % perimeter1, org, font, fontScale, color, thickness,cv2.LINE_AA)
        org = np.mean(point2_arr, axis=0).astype(int)
        org[0] -= 100
        cv2.putText(image, "Area: %.2f" % Area2, org, font, fontScale, color, thickness, cv2.LINE_AA)
        org[1] += 50
        cv2.putText(image, "perimeter: %.2f" % perimeter2, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def line_detection_vectorized(self,image, edge_image, num_rhos=180, num_thetas=180, t_count=500):

        thres = self.horizontalSlider_best_thres.value()
        edge_height, edge_width = edge_image.shape[:2]
        edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
        #
        d = np.sqrt(np.square(edge_height) + np.square(edge_width))
        dtheta = 180 / num_thetas
        drho = (2 * d) / num_rhos
        #
        thetas = np.arange(0, 180, step=dtheta)
        rhos = np.arange(-d, d, step=drho)
        #
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        edge_points = np.argwhere(edge_image != 0)
        edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
        #
        rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
        #找到所有線
        accumulator, theta_vals, rho_vals = np.histogram2d(
            np.tile(thetas, rho_values.shape[0]),
            rho_values.ravel(),
            bins=[thetas, rhos]
        )
        accumulator = np.transpose(accumulator)
        # 取出所有滿足條件的線
        lines = np.argwhere(accumulator > t_count)

        self.best = []
        #過濾太過相近的線，將同一個邊緣的其他霍夫直線過濾掉
        for line in lines:
            if len(self.best)>0:
                app = 0
                for i in self.best:
                    if np.mean(np.abs(line[:]-i[:])) < thres:
                        break
                    else:
                         app += 1
                    if app == len(self.best):
                        self.best.append(line)
            else:
                self.best.append(line)
        #畫線道影像中
        for line in self.best:
            y, x = line
            rho = rhos[y]
            theta = thetas[x]
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))

            x0 = (a * rho) + edge_width_half
            y0 = (b * rho) + edge_height_half
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # print(x0,y0)
            cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

        return accumulator, rhos, thetas,image
    def pushbottom_edge_clicked(self):#霍夫轉換
        img_path, _ = QFileDialog.getOpenFileName(self,
                                                  "Open file",
                                                  "./",
                                                  "Images (*.png *.bmp *.jpg *.JPG)")
        img = cv2.imread(img_path)  # 讀檔

        t_count = self.horizontalSlider_tcount.value()
        img1 = img[:, :, 0].copy()
        s = 2
        origin = np.zeros((img1.shape[0] + 2 * s, img.shape[1] + 2 * s), np.uint8)
        origin[s:-s, s:-s] = img1.copy()
        new = cv2.GaussianBlur(img1, (3, 3), 1)
        new = cv2.Canny(new, 100, 200)
        accumulator, thetas, rhos,img = self.line_detection_vectorized(img,new,t_count = t_count)
        #只對特定圖片進行面積及周長計算
        if basename(img_path) == 'rects.bmp':
            self.get_area_perimeter(img)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3,QImage.Format_RGB888)
        self.label_edge.setPixmap(QPixmap.fromImage(qimg))
        self.label_edge.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
