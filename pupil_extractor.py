import cv2
import os
import  PIL.Image as Image
from PIL import Image, ImageOps
import numpy as np
import sys
import cv2
import numpy
import imutils
from matplotlib import pyplot
from scipy.stats import gaussian_kde


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)
    height_max = width_max
    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

def extract_puil(img_path):
      img = cv2.imread(img_path)
      # 使用縮圖以減少計算量
      if  os.path.isfile(img_path):
        print(os.path.isfile(img_path))
        img_small = imutils.resize(img, width=640)

        # 在圖片周圍加上白邊，讓處理過程不會受到邊界影響
        padding = int(img.shape[1]/25)
        img = cv2.copyMakeBorder(img, padding, padding, padding, padding,
              cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 轉換至 HSV 色彩空間
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        # 取出飽和度
        saturation = hsv[:,:,1]
        saturation_small = hsv_small[:,:,1]

        # 取出明度
        value = hsv[:,:,2]
        value_small = hsv_small[:,:,2]

        # 綜合飽和度與明度
        sv_ratio = 0.8
        sv_value = cv2.addWeighted(saturation, sv_ratio, value, 1-sv_ratio, 0)
        sv_value_small = cv2.addWeighted(saturation_small, sv_ratio, value_small, 1-sv_ratio, 0)

        # 除錯用的圖形
        # pyplot.subplot(131).set_title("Saturation"), pyplot.imshow(saturation), pyplot.colorbar()
        # pyplot.subplot(132).set_title("Value"), pyplot.imshow(value), pyplot.colorbar()
        # pyplot.subplot(133).set_title("SV-value"), pyplot.imshow(sv_value), pyplot.colorbar()
        # pyplot.show()


        # 使用 Kernel Density Estimator 計算出分佈函數
        density = gaussian_kde(sv_value_small.ravel(), bw_method=0.15)

        # 找出 PDF 中第一個區域最小值（Local Minimum）作為門檻值
        step = 0.5
        xs = numpy.arange(0, 256, step)
        ys = density(xs)
        cum = 0
        for i in range(1, 250):
          cum += ys[i-1] * step
          if (cum > 0.02) and (ys[i] < ys[i+1]) and (ys[i] < ys[i-1]):
            threshold_value = xs[i]
            break

        # 除錯用的圖形
        # pyplot.hist(sv_value_small.ravel(), 256, [0, 256], True, alpha=0.5)
        # pyplot.plot(xs, ys, linewidth = 2)
        # pyplot.axvline(x=threshold_value, color='r', linestyle='--', linewidth = 2)
        # pyplot.xlim([0, max(threshold_value*2, 80)])
        # pyplot.show()

        # 以指定的門檻值篩選區域
        _, threshold = cv2.threshold(sv_value, threshold_value, 255.0, cv2.THRESH_BINARY)

        # 去除微小的雜訊
        kernel_radius = int(img.shape[1]/100)
        kernel = numpy.ones((kernel_radius, kernel_radius), numpy.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # 除錯用的圖形
        # pyplot.imshow(threshold, "gray")
        # pyplot.show()


        # 產生等高線
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 建立除錯用影像
        img_debug = img.copy()

        # 線條寬度
        line_width = int(img.shape[1]/100)



        # 以藍色線條畫出所有的等高線
        cv2.drawContours(img_debug, contours, -1, (255, 0, 0), line_width)

        # print(len(contours))
        # 找出面積最大的等高線區域
        c = sorted(contours, key = cv2.contourArea)
        # print(c)
        # if len(c) ==2 :
        #   c = c[0]
        # elif len(c) ==3:
        #### need to be modify
        # print(len(c))
        c = c[-2]

        # 找出可以包住面積最大等高線區域的方框，並以綠色線條畫出來
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_debug,(x, y), (x + w, y + h), (0, 255, 0), line_width)

        # 嘗試在各種角度，以最小的方框包住面積最大的等高線區域，以紅色線條標示
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(img_debug, [box], 0, (0, 0, 255), line_width)

        # 除錯用的圖形
        # pyplot.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        # pyplot.show()


        # 取得紅色方框的旋轉角度
        angle = rect[2]
        if angle < -45:
          angle = 90 + angle

        # 以影像中心為旋轉軸心
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        # 計算旋轉矩陣
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋轉圖片
        rotated = cv2.warpAffine(img_debug, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        img_final = cv2.warpAffine(img, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        # 除錯用的圖形
        # pyplot.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        # pyplot.show()

        # 旋轉紅色方框座標
        pts = numpy.int0(cv2.transform(numpy.array([box]), M))[0]

        # 計算旋轉後的紅色方框範圍
        y_min = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
        y_max = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
        x_min = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1])
        x_max = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1])

        # 裁切影像
        #need to be modify the range
        img_crop = rotated[x_min:x_max, y_min:y_max]
        img_final = img_final[x_min-150:x_max+150, y_min-150:y_max+150]
        # img_final = cv2.resize(img_final, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img_final)
        return img_final
        # cv2.imwrite('test', img)
        # 除錯用的圖形
        # pyplot.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        # pyplot.show()

        # 完成圖
        # pyplot.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        # pyplot.show()


        ## image padding

for sub in os.listdir(r"C:\Users\admin\Downloads\1243"):
    sub_path = os.path.join(r"C:\Users\admin\Downloads\1243",sub)
    imges=[]
    imges_name=[]
    for f in os.listdir(sub_path):
        if '.jpg' in f :
            print(f)

            img_path = os.path.join(sub_path, f)
            img_final = extract_puil(img_path)
            ###
            # img_final = cv2.imread(img_path)
            ###
            imges.append(img_final)
            imges_name.append(img_path)
    pad_imges = pad_images_to_same_size(imges)
    for i,name in zip(pad_imges,imges_name):
          ret = cv2.imwrite(name, i)
          print(ret)
