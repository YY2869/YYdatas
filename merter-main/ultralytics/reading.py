
import cv2
import imutils
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    # 等待
    cv2.waitKey(0)  # 0表示任意键终止
    # 释放内存
    cv2.destroyAllWindows()


def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst



class Basic_processing:   #运行函数时注意顺序，反了会报错
    def __init__(self, path1, path2):
        self.image1 = path1
        self.image2 = path2
        self.sigma_list = [15, 80, 200]
        self.SIFTPerspectiveTransformation()
        self.Bilateral()
        self.clahe()


    def SIFTPerspectiveTransformation(self):  # 基于SIFT特征点检测的透视变换算法
        # 灰度图转换
        img1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # 提取特征点
        sift = cv2.SIFT_create(800)  # SIFT特征检测器
        keyPoint1, desc1 = sift.detectAndCompute(img1, None)
        keyPoint2, desc2 = sift.detectAndCompute(img2, None)

        # 特征点匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # 应用比率测试以获得良好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 提取匹配点的坐标
        imagePoints1 = np.float32([keyPoint1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        imagePoints2 = np.float32([keyPoint2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算透视变换矩阵
        homo, _ = cv2.findHomography(imagePoints1, imagePoints2, cv2.RANSAC)

        # 执行透视变换
        self.imageTransform1 = cv2.warpPerspective(self.image1, homo, (self.image2.shape[1], self.image2.shape[0]))
        # cv_show('SIFTPerspectiveTransformation', self.imageTransform1)
        # cv2.imwrite('C:/Users/yang123/Desktop/video/rain/SIFT.jpg', self.imageTransform1)

    def Bilateral(self):   #函数必须运行，才能使用里面的参数
        #...定义滤波器参数
        #diameter  滤波器直径，控制像素邻域大小
        #sigma_color 色彩空间标准差，控制颜色相似性权重
        #sigma_space  空间空间标准差，控制空间相似性权重
        self.bilateral_filtered_image = cv2.bilateralFilter(self.imageTransform1, 25, 30, 100)  #双边滤波
        # self.img3 = np.float64(bilateral_filtered_image) + 1.0
        # cv2.imwrite('C:/Users/yang123/Desktop/lvbo.jpg', self.bilateral_filtered_image)
        # cv_show('lvbo', self.bilateral_filtered_image)

    def clahe(self):
        b, g, r = cv2.split(self.bilateral_filtered_image)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        image_clahe = cv2.merge([b, g, r])
        image_clahe = cv2.bilateralFilter(image_clahe, 7, 17, 100)  # 双边滤波
        # image_clahe = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2GRAY)
        # cv_show('lvbo', image_clahe)
        # cv2.imwrite('C:/Users/yang123/Desktop/zengqiang.jpg',image_clahe)
        return image_clahe


# if __name__ == "__main__":
#     # 读取图像
#     image01 = cv2.imread('C:/Users/yang123/Desktop/video/258.jpg')  # 测试图片
#     image01 = cv2.resize(image01, (480, 640))
#     image02 = cv2.imread('C:/Users/yang123/Desktop/video/258.jpg')  # 模板图片
#     image02 = cv2.resize(image02, (480, 640))
#
#     A = Basic_processing(image01, image02)


