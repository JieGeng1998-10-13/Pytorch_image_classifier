import os
import shutil
import random

# 设置根目录和训练集比例
root_dir = 'Resnet分类任务'
train_ratio = 0.7

# 获取所有类别的文件夹名称
categories = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

# 排除已经存在的 'train' 和 'test' 文件夹
categories = [cat for cat in categories if cat not in ['train', 'test']]

for category in categories:
    category_path = os.path.join(root_dir, category)
    images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]

    # 随机打乱图像列表
    random.shuffle(images)

    # 计算训练集的图像数量
    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    test_images = images[num_train:]

    # 创建 'train' 和 'test' 子目录
    train_category_path = os.path.join(root_dir, 'train', category)
    test_category_path = os.path.join(root_dir, 'test', category)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)

    # 将图像复制到训练集文件夹
    for image in train_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(train_category_path, image)
        shutil.copy2(src, dst)

    # 将图像复制到测试集文件夹
    for image in test_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(test_category_path, image)
        shutil.copy2(src, dst)

print('数据集划分完成。')