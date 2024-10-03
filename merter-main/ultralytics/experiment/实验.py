# import cv2
# import numpy as np
# from math import sqrt
# from reading import Basic_processing
# import random
# import imutils
# import onnx
# import onnxruntime as ort
#
#
# def cv_show(name, img):
#     cv2.imshow(name, img)
#     # 等待
#     cv2.waitKey(0)  # 0表示任意键终止
#     # 释放内存
#     cv2.destroyAllWindows()
#
#
# CLASSES = ['yuanxin']  # coco80类别
#
# onnx_path = 'C:/Users/yang123/Desktop/video/best.onnx'
#
#
# def nms(dets, thresh):
#     # dets:x1 y1 x2 y2 score class
#     # x[:,n]就是取所有集合的第n个数据
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     # -------------------------------------------------------
#     #   计算框的面积
#     #	置信度从大到小排序
#     # -------------------------------------------------------
#     areas = (y2 - y1 + 1) * (x2 - x1 + 1)
#     scores = dets[:, 4]
#     # print(scores)
#     keep = []
#     index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
#     # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序
#
#     while index.size > 0:
#         i = index[0]
#         keep.append(i)
#         # -------------------------------------------------------
#         #   计算相交面积
#         #	1.相交
#         #	2.不相交
#         # -------------------------------------------------------
#         x11 = np.maximum(x1[i], x1[index[1:]])
#         y11 = np.maximum(y1[i], y1[index[1:]])
#         x22 = np.minimum(x2[i], x2[index[1:]])
#         y22 = np.minimum(y2[i], y2[index[1:]])
#
#         w = np.maximum(0, x22 - x11 + 1)
#         h = np.maximum(0, y22 - y11 + 1)
#
#         overlaps = w * h
#         # -------------------------------------------------------
#         #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
#         #	IOU小于thresh的框保留下来
#         # -------------------------------------------------------
#         ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
#         idx = np.where(ious <= thresh)[0]
#         index = index[idx + 1]
#     return keep
#
#
# def xywh2xyxy(x):
#     # [x, y, w, h] to [x1, y1, x2, y2]
#     y = np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2
#     y[:, 1] = x[:, 1] - x[:, 3] / 2
#     y[:, 2] = x[:, 0] + x[:, 2] / 2
#     y[:, 3] = x[:, 1] + x[:, 3] / 2
#     return y
#
#
# def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
#     # -------------------------------------------------------
#     #   删除为1的维度
#     #	删除置信度小于conf_thres的BOX
#     # -------------------------------------------------------
#     org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
#     # (25200, 9)
#     # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
#     conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
#     box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
#     # print('box:符合要求的框')
#     # print(box.shape)
#
#     # -------------------------------------------------------
#     #   通过argmax获取置信度最大的类别
#     # -------------------------------------------------------
#     cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
#     cls = []
#     for i in range(len(cls_cinf)):
#         cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
#     all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
#     # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
#     # -------------------------------------------------------
#     #   分别对每个类别进行过滤
#     #   1.将第6列元素替换为类别下标
#     #	2.xywh2xyxy 坐标转换
#     #	3.经过非极大抑制后输出的BOX下标
#     #	4.利用下标取出非极大抑制后的BOX
#     # -------------------------------------------------------
#     output = []
#     for i in range(len(all_cls)):
#         curr_cls = all_cls[i]
#         curr_cls_box = []
#         curr_out_box = []
#
#         for j in range(len(cls)):
#             if cls[j] == curr_cls:
#                 box[j][5] = curr_cls
#                 curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5
#
#         curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
#         # curr_cls_box_old = np.copy(curr_cls_box)
#         curr_cls_box = xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
#         curr_out_box = nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标
#
#         for k in curr_out_box:
#             output.append(curr_cls_box[k])
#     output = np.array(output)
#     return output
#
#
# class Functions:
#     @staticmethod
#     def GetClockAngle(v1, v2):
#         # 2个向量模的乘积 ,返回夹角
#         TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
#         # 叉乘
#         rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
#         # 点乘
#         theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
#         if rho > 0:
#             return 360 - theta
#         else:
#             return theta
#
#     @staticmethod
#     def Disttances(a, b):
#         # 返回两点间距离
#         x1, y1 = a
#         x2, y2 = b
#         Disttances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
#         return Disttances
#
#     @staticmethod
#     def couputeMean(deg):
#         # 对数据进行处理，提取均值
#         """
#         :funtion :
#         :param b:
#         :param c:
#         :return:
#         """
#         if (True):
#             # new_nums = list(set(deg)) #剔除重复元素
#             mean = np.mean(deg)
#             var = np.var(deg)
#             # print("原始数据共", len(deg), "个\n", deg)
#             '''
#             for i in range(len(deg)):
#                 print(deg[i],'→',(deg[i] - mean)/var)
#                 #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
#             '''
#             # print("中位数:",np.median(deg))
#             percentile = np.percentile(deg, (25, 50, 75), method='midpoint')
#             # print("分位数：", percentile)
#             # 以下为箱线图的五个特征值
#             Q1 = percentile[0]  # 上四分位数
#             Q3 = percentile[2]  # 下四分位数
#             IQR = Q3 - Q1  # 四分位距
#             ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
#             llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
#
#             new_deg = []
#             uplim = []
#             for i in range(len(deg)):
#                 if (llim < deg[i] and deg[i] < ulim):
#                     new_deg.append(deg[i])
#             # print("清洗后数据共", len(new_deg), "个\n", new_deg)
#         new_deg = np.mean(new_deg)
#
#         return new_deg
#
# class Yolov5ONNX(object):
#     def __init__(self, onnx_path):
#         """检查onnx模型并初始化onnx"""
#         onnx_model = onnx.load(onnx_path)
#         try:
#             onnx.checker.check_model(onnx_model)
#         except Exception:
#             print("Model incorrect")
#         else:
#             print("Model correct")
#
#         options = ort.SessionOptions()
#         options.enable_profiling = True
#         # self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options,
#         #                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#         self.onnx_session = ort.InferenceSession(onnx_path)
#         self.input_name = self.get_input_name()  # ['images']
#         self.output_name = self.get_output_name()  # ['output0']
#
#     def get_input_name(self):
#         """获取输入节点名称"""
#         input_name = []
#         for node in self.onnx_session.get_inputs():
#             input_name.append(node.name)
#
#         return input_name
#
#     def get_output_name(self):
#         """获取输出节点名称"""
#         output_name = []
#         for node in self.onnx_session.get_outputs():
#             output_name.append(node.name)
#
#         return output_name
#
#     def get_input_feed(self, image_numpy):
#         """获取输入numpy"""
#         input_feed = {}
#         for name in self.input_name:
#             input_feed[name] = image_numpy
#
#         return input_feed
#
#     def inference(self, img):
#         """ 1.cv2读取图像并resize
#         2.图像转BGR2RGB和HWC2CHW(因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640)
#         3.图像归一化
#         4.图像增加维度
#         5.onnx_session 推理 """
#         # img = cv2.imread(img_path)
#         or_img = cv2.resize(img, (480, 640))  # resize后的原图 (640, 640, 3)
#         img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
#         img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
#         img /= 255.0
#         img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
#         # img尺寸(1, 3, 640, 640)
#         input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
#         pred = self.onnx_session.run(None, input_feed)[0]  # <class 'numpy.ndarray'>(1, 25200, 9)
#
#         return pred, or_img
#
#
#
# def draw(image, box_data):
#     # -------------------------------------------------------
#     #	取整，方便画框
#     # -------------------------------------------------------
#     center_dnn = []
#     boxes = box_data[..., :4].astype(np.int32)  # x1 x2 y1 y2
#     scores = box_data[..., 4]
#     classes = box_data[..., 5].astype(np.int32)
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = box
#         # 求中心
#         center_x = int((top + right) / 2)
#         center_y = int((left + bottom) / 2)
#         center_dnn = [center_x, center_y]
#         # print('class: {}, score: {}'.format(CLASSES[cl], score))
#         # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
#
#         cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)  # 在中心点画一个红色的圆
#         cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score), (top, left), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                     (0, 0, 255), 2)
#
#     return image, center_dnn
#
#
#
#
# class MeterDetection:
#     def __init__(self, img_loading):
#         self.image = img_loading
#         self.divisionValue = 1.6 / 270  # 分度值
#         self.cirleData = None  # 过滤圆数据
#         self.bitwiseOr = None  # 圆盘图像
#         self.poniterMask = None  # 指针图像
#         self.numLineMask = None  # 刻度图像
#
#     # ..........................................截取表盘区域，滤除背景...............................
#     def Dial_area(self):
#         img1 = self.image
#         cimage = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
#         circles = np.uint16(np.around(circles))  # 把类型换成整数
#         r_1 = circles[0, 0, 2]  # 圆心半径
#         c_x = circles[0, 0, 0]
#         c_y = circles[0, 0, 1]
#         circle = np.ones(img1.shape, dtype="uint8")
#         circle = circle * 255
#         cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)
#         self.bitwiseOr = cv2.bitwise_or(img1, circle)
#         # cv2.imwrite('C:/Users/yang123/Desktop/video/Dial_area.jpg', self.bitwiseOr)
#         # self.centerPoint_dnn = opencv_dnn.main()  # dnn求得中心点
#         self.cirleData = [r_1, c_x, c_y]
#         # cv_show('Circular dial', self.bitwiseOr)  # 表盘图像
#
#     # .......................................dnn........................................
#     def opencv_dnn(self):
#         try:
#             model = Yolov5ONNX(onnx_path)
#             output, or_img = model.inference(self.bitwiseOr)
#             # print('pred: 位置[0, 10000, :]的数组')
#             # print(output.shape)
#             # print(output[0, 10000, :])
#             outbox = filter_box(output, 0.7, 0.5)  # 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
#             # print('outbox( x1 y1 x2 y2 score class):')
#             # print(outbox)
#             # if len(outbox) == 0:
#             #     print('没有发现物体')
#             #     sys.exit(0)
#             or_img, self.center_dnn = draw(or_img, outbox)
#             cv2.imwrite('C:/Users/yang123/Desktop/video/123.jpg', or_img)
#             return self.center_dnn
#         except Exception as e:
#             print("程序错误：", e)
#
#     # ......................................清楚表盘文字信息....................................
#     def Clear_information(self):
#         # 初始化卷积核
#         rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 核的大小3x9
#         sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 核的大小5x5
#
#         # 读取输入图像，预处理
#         image = self.bitwiseOr
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # cv_show('gray', gray)
#
#         # 礼帽操作，突出更明亮的区域
#         tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#         # cv_show('tophat', tophat)
#         # sobel算法
#         gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1相当于用3*3的
#
#         gradX = np.absolute(gradX)
#         (minVal, maxVal) = (np.min(gradX), np.max(gradX))
#         gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
#         gradX = gradX.astype("uint8")
#
#         # print(np.array(gradX).shape)
#         # cv_show('gradX', gradX)
#
#         # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
#         gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
#         # cv_show('gradX', gradX)
#         # THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
#         thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#         # cv_show('thresh', thresh)
#         # 再来一个闭操作,填缝隙
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # 再来一个闭操作
#         # cv_show('thresh',thresh)
#
#         # 以上操作全是为了找轮廓参数，再原图上画
#         # 计算轮廓
#         threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cnts = threshCnts
#         cur_img = image.copy()
#         cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)  # 经过一系列步骤算出的轮廓参数，然后画在原始图像当中
#         # cv_show('cur_img', cur_img)
#         mask = np.full_like(gray.copy(), 255)
#
#         # 遍历轮廓，根据实际需要只保留有数字的轮廓
#         for (i, c) in enumerate(cnts):
#             # 计算矩形
#             (x, y, w, h) = cv2.boundingRect(c)  # 外接矩形
#             ar = w / float(h)  # 算比例
#
#             # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
#             if ar > 3:
#                 if w > 25 and h > 10:
#                     self.mask = cv2.rectangle(mask, (x - 5, y - 5), (x + w + 10, y + h + 10), (0, 0, 0), -1)
#         # cv_show('img', self.mask)
#
#     # ............................................找指针和刻度线.......................................
#     def Scalelines_Pointer(self):
#         img2 = self.bitwiseOr.copy()
#         gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#         binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,
#                                        -15)  # 二值化,很大影响
#         kernel1 = np.ones((2, 2), np.uint8)
#         kernel2 = np.ones((7, 7), np.uint8)
#         # 开运算：先腐蚀，在膨胀
#         binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
#         # cv_show('Binarization', binary)
#
#         # cv2.imwrite('C:/Users/yang123/Desktop/999.jpg', binary)
#         # # 使用按位与运算将原始图像和掩码相结合
#         result = cv2.bitwise_and(binary, self.mask)
#         result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel2)
#         # cv_show('3', result)
#         contours, hier = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓查找
#         r_1, c_x, c_y = self.cirleData
#
#         cntset = []  # 刻度线轮廓集合
#         cntareas = []  # 刻度线面积集合
#         needlecnt = []  # 指针轮廓集合
#         needleareas = []  # 指针面积集合
#         radiusLength = [r_1 * 0.6, r_1 * 1]  # 半径范围
#         localtion = []
#
#         for cnt in contours:
#             rect = cv2.minAreaRect(cnt)
#             a, (w, h), c = rect
#             w = int(w)
#             h = int(h)
#             ''' 满足条件:“长宽比例”，“面积”'''
#             if h == 0 or w == 0:
#                 pass
#             else:
#                 dis = Functions.Disttances((c_x, c_y), a)
#                 if (radiusLength[0] < dis and radiusLength[1] > dis):
#                     # 矩形筛选
#                     if h / w > 4 or w / h > 4:
#                         localtion.append(dis)
#                         cntset.append(cnt)
#                         cntareas.append(w * h)
#                 else:
#                     if w > r_1 / 2 or h > r_1 / 2:
#                         needlecnt.append(cnt)
#                         needleareas.append(w * h)
#
#         cntareas = np.array(cntareas)
#         areasMean = Functions.couputeMean(cntareas)  # 中位数，上限区
#         new_cntset = []
#         # 面积
#         for i, cnt in enumerate(cntset):
#             if (cntareas[i] <= areasMean * 1.6 and cntareas[i] >= areasMean * 0.7):  # 改变刻度数量
#                 new_cntset.append(cnt)
#
#         self.r = np.mean(localtion)
#         mask = np.zeros(img2.shape[0:2], np.uint8)
#         self.poniterMask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜
#         mask = np.zeros(img2.shape[0:2], np.uint8)
#         self.numLineMask = cv2.drawContours(mask, new_cntset, -1, (255, 255, 255), -1)  # 生成掩膜
#         self.new_cntset = new_cntset
#         self.binary = binary
#
#         # cv_show('Pointer', self.poniterMask)
#         # cv_show('Scale lines',self.numLineMask)
#
#     # .............................................刻度线拟合........................................
#     def Scaleline_fitting(self):
#         lineSet = []  # 拟合线集合
#         img3 = self.image.copy()
#         for cnt in self.new_cntset:
#             rect = cv2.minAreaRect(cnt)
#             # 获取矩形四个顶点，浮点型
#             box = cv2.boxPoints(rect)
#             box = np.intp(box)
#             cv2.polylines(img3, [box], True, (0, 255, 0), 1)  # pic
#             output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
#             k = output[1] / output[0]
#             k = round(k[0], 2)
#             b = output[3] - k * output[2]
#             b = round(b[0], 2)
#             x1 = 1
#             x2 = img3.shape[0]
#             y1 = int(k * x1 + b)
#             y2 = int(k * x2 + b)
#             cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 1)
#             # lineSet:刻度线拟合直线数组，k斜率 b
#             lineSet.append([k, b])  # 求中心点的点集[k,b]
#             self.lineSet = lineSet
#         # cv_show('Scale line fitting', img3)
#
#     # ....................................................求圆心.......................................
#     def Center_of_Circle(self):
#         w, h, c = self.image.shape
#         xlist = []
#         ylist = []
#         if len(self.lineSet) > 2:
#             # print(len(lineSet))
#             np.random.shuffle(self.lineSet)
#             lkb = int(len(self.lineSet) / 2)
#             kb1 = self.lineSet[0:lkb]
#             kb2 = self.lineSet[lkb:(2 * lkb)]
#             # print('len', len(kb1), len(kb2))
#             kb1sample = random.sample(kb1, int(len(kb1) / 2))
#             kb2sample = random.sample(kb2, int(len(kb2) / 2))
#
#         else:
#             kb1sample = self.lineSet[0]
#             kb2sample = self.lineSet[1]
#         for i, wx in enumerate(kb1sample):
#             # for wy in kb2:
#             for wy in kb2sample:
#                 k1, b1 = wx
#                 k2, b2 = wy
#                 # print('kkkbbbb',k1[0],b1[0],k2[0],b2[0])
#                 # k1-->[123]
#                 try:
#                     if (b2 - b1) == 0:
#                         b2 = b2 - 0.1
#                     if (k1 - k2) == 0:
#                         k1 = k1 - 0.1
#                     x = (b2 - b1) / (k1 - k2)
#                     y = k1 * x + b1
#                     x = int(round(x))
#                     y = int(round(y))
#                 except:
#                     x = (b2 - b1 - 0.01) / (k1 - k2 + 0.01)
#                     y = k1 * x + b1
#                     x = int(round(x))
#                     y = int(round(y))
#                 # x,y=solve_point(k1, b1, k2, b2)
#                 if x < 0 or y < 0 or x > w or y > h:
#                     break
#                 # point_list.append([x, y])
#                 xlist.append(x)
#                 ylist.append(y)
#                 # cv2.circle(img, (x, y), 2, (122, 22, 0), 2)
#         # print('point_list',point_list)
#         cx = int(np.mean(xlist))
#         cy = int(np.mean(ylist))
#
#         self.centerPoint = [cx, cy]
#         cv2.circle(self.image, (self.center_dnn[0], self.center_dnn[1]), 2, (0, 0, 255), 2)
#
#         # cv_show('Center of Circle', self.image)
#
#     def Pointer_top(self):
#         # ....................................找指针顶点........................................
#         img4 = self.poniterMask
#         lines = cv2.HoughLinesP(img4, 1, np.pi / 180, 100, minLineLength=int(self.r / 2), maxLineGap=2)
#         dmax = 0
#         pointerLine = []
#         # 最长的线段为指针
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             d1 = Functions.Disttances((x1, y1), (x2, y2))
#             if (d1 > dmax):
#                 dmax = d1
#                 pointerLine = line[0]
#         x1, y1, x2, y2 = pointerLine
#         d1 = Functions.Disttances((x1, y1), (self.center_dnn[0], self.center_dnn[1]))
#         d2 = Functions.Disttances((x2, y2), (self.center_dnn[0], self.center_dnn[1]))
#         if d1 > d2:
#             self.farPoint = [x1, y1]
#         else:
#             self.farPoint = [x2, y2]
#
#         cv2.line(self.image, (self.center_dnn[0], self.center_dnn[1]), (self.farPoint[0], self.farPoint[1]),
#                  (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.circle(self.image, (self.farPoint[0], self.farPoint[1]), 2, (0, 0, 255), 2)
#
#         # cv_show('Pointer vertex', self.image)
#
#     # ..............................找零刻度点.............................
#     def Zero_scale(self):
#         # 读取模板图片
#         template = cv2.imread('C:/Users/yang123/Desktop/video/666.jpg')
#         # 转换为灰度图片
#         template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#         # 执行边缘检测
#         template = cv2.Canny(template, 50, 200)
#         (tH, tW) = template.shape[:2]
#         # 显示模板
#         # cv2.imshow("Template", template)
#
#         # 读取测试图片并将其转化为灰度图片
#         image = self.bitwiseOr.copy()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         found = None
#
#         # 循环遍历不同的尺度
#         for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#             # 根据尺度大小对输入图片进行裁剪
#             resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
#             r = gray.shape[1] / float(resized.shape[1])
#
#             # 如果裁剪之后的图片小于模板的大小直接退出
#             if resized.shape[0] < tH or resized.shape[1] < tW:
#                 break
#
#             # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
#             edged = cv2.Canny(resized, 50, 200)
#             result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#             (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
#
#             # 绘制矩形框并显示结果
#             clone = np.dstack([edged, edged, edged])
#             cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#             # cv_show('clone', clone)
#
#             # 如果发现一个新的关联值则进行更新
#             if found is None or maxVal > found[0]:
#                 found = (maxVal, maxLoc, r)
#
#         # 计算测试图片中模板所在的具体位置，即左上角和右下角的坐标值，并乘上对应的裁剪因子
#         (_, maxLoc, r) = found
#         (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
#         (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
#
#         # 绘制并显示结果
#         cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#         # cv_show('image', image)
#         # 求出矩形框的中心点
#         center_x = (startX + endX) / 2
#         center_y = (startY + endY) / 2
#         self.zeroPoint = [int(center_x), int(center_y)]  # 找原点
#         cv2.circle(self.image, (self.zeroPoint[0], self.zeroPoint[1]), 3, (0, 0, 255), -1)  # 在中心点画一个红色的圆
#         cv2.line(self.image, (self.zeroPoint[0], self.zeroPoint[1]), (self.center_dnn[0], self.center_dnn[1]),
#                  (0, 0, 255), 2, cv2.LINE_AA)
#         # cv_show('image2', self.image)
#
#     def reading(self):
#
#         try:
#             self.Dial_area()
#             self.opencv_dnn()
#             self.Clear_information()
#             self.Scalelines_Pointer()
#             self.Scaleline_fitting()
#             self.Center_of_Circle()
#             self.Pointer_top()
#             self.Zero_scale()
#             divisionValue = 1.6 / 270
#             v1 = [self.zeroPoint[0] - self.center_dnn[0], self.center_dnn[1] - self.zeroPoint[1]]
#             v2 = [self.farPoint[0] - self.center_dnn[0], self.center_dnn[1] - self.farPoint[1]]
#             theta = Functions.GetClockAngle(v1, v2)
#             readValue = divisionValue * theta + 0.03
#             return readValue
#         except Exception as e:
#             print("程序错误：", e)
#
#
# # if __name__ =="__main__":
# #
# #     try:
# #         image01 = cv2.imread('C:/Users/yang123/Desktop/video/999.jpg')  # 测试图片
# #         image01 = cv2.resize(image01, (480, 640))
# #         image02 = cv2.imread('C:/Users/yang123/Desktop/video/1.jpg')  # 模板图片
# #         image02 = cv2.resize(image02, (480, 640))
# #         read = Basic_processing(image01, image02)
# #         img_enhance = read.clahe()
# #
# #         # cv_show('s', img_enhance)
# #
# #         img_deal = MeterDetection(img_enhance)
# #         value= img_deal.reading()
# #         print(value)
# #     except Exception as e:
# #         print("程序错误：", e)
#
# # yolo export model=ultralytics/runs/detect/train102/weights/best.pt format=onnx opset=12
#
import onnxruntime as rt
import numpy as np
import cv2
import matplotlib.pyplot as plt



CLASSES = ['yuanxin']  # coco80类别

def nms(pred, conf_thres, iou_thres):
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def draw(image, box_data):
    # -------------------------------------------------------
    #	取整，方便画框
    # -------------------------------------------------------
    center_dnn = []
    boxes = box_data[..., :4].astype(np.int32)  # x1 x2 y1 y2
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        #求中心
        center_x = int((top + right) / 2)
        center_y = int((left + bottom) / 2)
        center_dnn = [center_x, center_y]
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)  # 在中心点画一个红色的圆
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),(top, left),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)

    return image, center_dnn


if __name__ == '__main__':
    height, width = 640, 640
    img0 = cv2.imread('C:/Users/yang123/Desktop/video/888.jpg')
    x_scale = img0.shape[1] / width
    y_scale = img0.shape[0] / height
    img = img0 / 255.
    img = cv2.resize(img, (width, height))
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    sess = rt.InferenceSession('C:/Users/yang123/Desktop/video/best_meter.onnx')
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    result = nms(pred, 0.3, 0.45)
    output = np.array(result)
    print(output)
    or_img, center_dnn = draw(img0, output)
    cv2.imwrite('C:/Users/yang123/Desktop/video/123.jpg', or_img)





