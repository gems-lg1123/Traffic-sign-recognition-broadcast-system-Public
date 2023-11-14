import cv2
import math
import numpy as np
import os
import xlwt
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2lab


def hisEqulColor2(test):
    B, G, R = cv2.split(test)  # get single 8-bits channel
    EB = cv2.equalizeHist(B)
    EG = cv2.equalizeHist(G)
    ER = cv2.equalizeHist(R)
    h_img = cv2.merge((EB, EG, ER))  # merge it back
    return h_img


def brightness(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    # np.full 构造一个数组，用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m
    return k


def gamma(img):
    h, w = img.shape[:2]
    all = np.zeros((h, w), dtype=np.float64)
    out_img = img.copy()

    B = np.array(img[:, :, 0], dtype=np.float64)
    G = np.array(img[:, :, 1], dtype=np.float64)
    R = np.array(img[:, :, 2], dtype=np.float64)

    print(B.dtype)
    for i in range(0, h):
        for j in range(0, w):
            all[i, j] = ((B[i, j] + G[i, j] + R[i, j])) / 3.0

    revert_all = np.full((h, w), 255.0, dtype=np.float64)
    mast = revert_all - all
    mast_blur = cv2.GaussianBlur(mast, (41, 41), 0)
    for k in range(0, 3):
        for i in range(0, h):
            for j in range(0, w):
                exp = 2 ** ((128 - mast_blur[i, j]) / 128.0)
                value = np.uint8(255 * ((img.item(i, j, k) / 255.0) ** exp))
                out_img.itemset((i, j, k), value)
    return out_img


def deviation(img):
    """计算偏色值"""
    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    d_a, d_b, M_a, M_b = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            d_a = d_a + a[i][j]
            d_b = d_b + b[i][j]
    d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
    D = np.sqrt((np.square(d_a) + np.square(d_b)))

    for i in range(m):
        for j in range(n):
            M_a = np.abs(a[i][j] - d_a - 128) + M_a
            M_b = np.abs(b[i][j] - d_b - 128) + M_b

    M_a, M_b = M_a / (m * n), M_b / (m * n)
    M = np.sqrt((np.square(M_a) + np.square(M_b)))
    k = D / M
    # print('偏色值:%f' % k)
    return k


def color_correction(img):
    """
    基于图像分析的偏色检测及颜色校正方法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    """
    b, g, r = cv2.split(img)
    I_r_2 = (r.astype(np.float32) ** 2).astype(np.float32)
    I_b_2 = (b.astype(np.float32) ** 2).astype(np.float32)
    sum_I_r_2 = I_r_2.sum()
    sum_I_b_2 = I_b_2.sum()
    sum_I_g = g.sum()
    sum_I_r = r.sum()
    sum_I_b = b.sum()
    max_I_r = r.max()
    max_I_g = g.max()
    max_I_b = b.max()
    max_I_r_2 = I_r_2.max()
    max_I_b_2 = I_b_2.max()
    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    b_point = u_b * (b.astype(np.float32) ** 2) + v_b * b.astype(np.float32)
    r_point = u_r * (r.astype(np.float32) ** 2) + v_r * r.astype(np.float32)
    b_point[b_point > 255] = 255
    b_point[b_point < 0] = 0
    b = b_point.astype(np.uint8)
    r_point[r_point > 255] = 255
    r_point[r_point < 0] = 0
    r = r_point.astype(np.uint8)
    return cv2.merge([b, g, r])


def laplace(img):
    # Apply Laplacian correction to the color image
    a = -1
    b = 0
    kernel = np.array([[b, a, b], [a, 5, a], [b, a, b]])
    enhanced_image = cv2.filter2D(img, -1, kernel)
    return enhanced_image


def clearness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian


if __name__ == '__main__':

    # file address
    data_base_dir = r'image'  # 输入文件夹的路径
    outfile_dir = r'processed'  # 输出文件夹的路径
    boutfile_dir = r'bprocessed'  # 输出文件夹的路径
    coutfile_dir = r'cprocessed'  # 输出文件夹的路径
    cloutfile_dir = r'clprocessed'  # 输出文件夹的路径

    list = os.listdir(data_base_dir)
    list.sort()
    list2 = os.listdir(outfile_dir)
    list2.sort()

    # make a table
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('image value', cell_overwrite_ok=True)
    col = ('image', 'original brightness', 'corrected brightness', 'brightness difference',
           'original deviation', 'corrected deviation', 'deviation difference',
           'original clearness', 'corrected clearness', 'clearness difference')
    for i in range(0, 10):  # 列
        sheet.write(0, i, col[i]) #行 列 数
    j = 0  # 行
    btotal = 0
    dtotal = 0
    ctotal = 0

    # 遍历目标文件夹图片
    for file in list:
        read_img_name = data_base_dir + '/' + file.strip()  # 取图片完整路径
        img = cv2.imread(read_img_name)  # 读入图片
        imgh = cv2.imread(read_img_name,-1)
        # Calculate the original values
        bvalue = brightness(img)
        dvalue = deviation(img)
        cvalue = clearness(img)
        print("亮度:{}  色偏值：{}  清晰度:{}".format(float(bvalue), float(dvalue), float(cvalue)))

        # correction
        # # only Histogram Equalization
        # hh_img = hisEqulColor2(img)
        # later Histogram Equalization
        if bvalue > 0.1:
              n_img = gamma(img)  # gimage
        else:
              n_img = img
        nc_img = color_correction(n_img)  # gcimage
        if cvalue < 2000:  # gclimage
            ncl_img = laplace(nc_img)
        else:
            ncl_img = nc_img
        h_img = hisEqulColor2(ncl_img)
        hbvalue = brightness(h_img)
        hdvalue = deviation(h_img)
        hcvalue = clearness(h_img)

        plt.subplot(231)
        plt.imshow(img)
        plt.title("original")

        plt.subplot(232)
        plt.imshow(n_img)
        plt.title("brightness")#灰度值检测亮度；gamma

        plt.subplot(233)
        plt.imshow(nc_img)
        plt.title("color")#等效圆； grey world & perfect reflection

        plt.subplot(234)
        plt.imshow(ncl_img)
        plt.title("clearness")#laplace二阶算子

        plt.subplot(235)
        plt.imshow(h_img)
        plt.title("contrast")#直方图均衡化

        plt.show()



        print("后直方: 亮度:{}  色偏值：{}  清晰度:{}".format(float(hbvalue), float(hdvalue), float(hcvalue)))
        bdifference = hbvalue - bvalue
        ddifference = hdvalue - dvalue
        cdifference = hcvalue - cvalue
        data = [file.strip(), float(bvalue), float(hbvalue), float(bdifference),
                              float(dvalue), float(hdvalue), float(ddifference),
                              float(cvalue), float(hcvalue), float(cdifference)]
        for i in range(0, 10):
            sheet.write(j + 1, i, data[i])
            j += 1

        out_img_name = outfile_dir + '/' + file.strip()
        bout_img_name = boutfile_dir + '/' + file.strip()
        cout_img_name = coutfile_dir + '/' + file.strip()
        clout_img_name = cloutfile_dir + '/' + file.strip()

        cv2.imwrite(out_img_name, h_img)
        cv2.imwrite(bout_img_name, n_img)
        cv2.imwrite(cout_img_name, nc_img)
        cv2.imwrite(clout_img_name, ncl_img)
        print("The photo which is processed is {}".format(file))


        #cv2.waitKey(0)  # 等待键盘触发事件，释放窗口

    baverage = btotal / j
    daverage = dtotal / j
    caverage = ctotal / j


    print("平均亮度修正值", baverage)
    print("平均色偏修正值", daverage)
    print("平均清晰修正值", caverage)
    savepath = 'C:/Users/15192/Desktop/image value.xls'
    book.save(savepath)
