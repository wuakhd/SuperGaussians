# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms


def gaussian_filter_high_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    # 创建 x 和 y 的网格坐标
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    # 计算变换矩阵
    template = np.exp(- dis_square / (2 * D ** 2))

    return template * fshift


def gaussian_filter_low_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = 1 - np.exp(- dis_square / (2 * D ** 2))  # 高斯过滤器

    return template * fshift


def circle_filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * fshift.shape[0] / 2)
    if len(fshift.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def circle_filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * fshift.shape[0] / 2)
    if len(fshift.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg


def get_low_high_f(img, radius_ratio, D):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = circle_filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = circle_filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
    hight_parts_fshift = gaussian_filter_low_f(fshift.copy(), D=D)
    low_parts_fshift = gaussian_filter_high_f(fshift.copy(), D=D)

    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
            np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


def getFreTensor(img):
    # 假设输入的是tensor类型表示的RGB的图片
    r, g, b = img[0], img[1], img[2]
    gray_tensor = 0.299 * r + 0.587 * g + 0.114 * b
    # 将结果转换为 NumPy ndarray
    gray_image = gray_tensor.cpu().numpy()
    low_freq_part_img, high_freq_part_img = get_low_high_f(gray_image, radius_ratio=0.5,
                                                           D=50)  # multi channel or single
    transform_to = transforms.ToTensor()
    return transform_to(high_freq_part_img)

#扩张操作，将3*3范围内的每个值都取其中最大，相当于扩大了视为高频像素点的个数
#一个像素点高频，则周围一片区域也为高频
def get_local_maxima(tensor, kernel_size=3, padding_value=0):
    # 对张量进行边界填充，填充大小为 (kernel_size - 1) // 2
    padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
    padded_tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
    # 使用 unfold 提取每个 3x3 窗口
    unfolded = padded_tensor.unfold(0, kernel_size, 1).unfold(1, kernel_size, 1)
    # unfolded 的维度为 [height, width, kernel_size, kernel_size]
    # unfolded 张量形状是 [行数, 列数, kernel_size, kernel_size]
    # 计算每个局部窗口的最大值
    max_values, _ = unfolded.max(dim=-1)  # 对每个窗口的每一列取最大值
    max_values, _ = max_values.max(dim=-1)  # 对每个窗口的每一行取最大值
    return max_values

def grayToWeight(gray, scale = 1.0):
    # 计算灰度图的权重，并将其映射到 [1, 2] 范围内
    #normalized_weight = (gray - gray.min()) / (gray.max() - gray.min())  # 归一化到 [0, 1]
    #weight = normalized_weight * scale  # 缩放到 [1, 2] 范围
    # 扩展权重，使其与 RGB 图像的大小相匹配
    # 这里假设灰度图像是单通道的，我们需要将权重扩展为 (C, H, W) 形状
    #weight_expanded = weight.unsqueeze(0).expand_as(rgb_target)  # 将权重扩展到 (3, H, W)

    # 应用 sigmoid 函数
    tensor_sigmoid = torch.sigmoid(gray)
    # 将 sigmoid 输出映射到 [1, 2]
    tensor_scaled = 4.33 * tensor_sigmoid - 0.165

    #扩张操作（一个高频点对其周围的像素也视为高频点）
    tensor_scaled = get_local_maxima(tensor_scaled)

    return tensor_scaled

    #return weight#weight_expanded

def weight_l1_loss(network_output, gt, weight):
    return torch.abs((network_output - gt)*weight).mean()