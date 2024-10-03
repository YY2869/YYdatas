# import os
# import random
# import shutil
#
# # 原数据集目录
# root_dir = 'D:/work-app/yolov8/data2'
# # 划分比例
# train_ratio = 0.7
# valid_ratio = 0.2
# test_ratio = 0.1
#
# # 设置随机种子
# random.seed(42)
#
# # 拆分后数据集目录
# split_dir = 'datasets/meter'
# os.makedirs(os.path.join(split_dir, 'train/images'), exist_ok=True)
# os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
# os.makedirs(os.path.join(split_dir, 'valid/images'), exist_ok=True)
# os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
# os.makedirs(os.path.join(split_dir, 'test/images'), exist_ok=True)
# os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)
#
# # 获取图片文件列表
# image_files = os.listdir(os.path.join(root_dir, 'images'))
# label_files = os.listdir(os.path.join(root_dir, 'labels'))
#
# # 随机打乱文件列表
# combined_files = list(zip(image_files, label_files))
# random.shuffle(combined_files)
# image_files_shuffled, label_files_shuffled = zip(*combined_files)
#
# # 根据比例计算划分的边界索引
# train_bound = int(train_ratio * len(image_files_shuffled))
# valid_bound = int((train_ratio + valid_ratio) * len(image_files_shuffled))
#
# # 将图片和标签文件移动到相应的目录
# for i, (image_file, label_file) in enumerate(zip(image_files_shuffled, label_files_shuffled)):
#     if i < train_bound:
#         shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'train/images', image_file))
#         shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'train/labels', label_file))
#     elif i < valid_bound:
#         shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'valid/images', image_file))
#         shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'valid/labels', label_file))
#     else:
#         shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'test/images', image_file))
#         shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'test/labels', label_file))


import os
import random
import shutil

# 定义源文件夹和目标文件夹路径
source_folder = "D:/work-app/yolov8/data2/images"
train_folder = "D:/work-app/yolov8/ultralytics-main/ultralytics/datasets/meter/train/images"
validation_folder = "D:/work-app/yolov8/ultralytics-main/ultralytics/datasets/meter/valid/images"
test_folder = "D:/work-app/yolov8/ultralytics-main/ultralytics/datasets/meter/test/images"

# 创建目标文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 获取源文件夹中所有图片文件的列表
image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg") or f.endswith(".png")]

# 随机打乱图片文件列表
random.shuffle(image_files)

# 计算训练集、验证集和测试集的数量
total_images = len(image_files)
train_count = int(total_images * 0.7)
validation_count = int(total_images * 0.2)
test_count = total_images - train_count - validation_count

# 分配图片到不同的文件夹
for i, image_file in enumerate(image_files):
    if i < train_count:
        shutil.copy(os.path.join(source_folder, image_file), train_folder)
    elif i < train_count + validation_count:
        shutil.copy(os.path.join(source_folder, image_file), validation_folder)
    else:
        shutil.copy(os.path.join(source_folder, image_file), test_folder)

