import cv2
import numpy as np
from PIL import Image


def super_resolution(img, model_path='ESPCN_x2.pb'):
    """基于深度学习的超分辨率重建"""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 2)
    return sr.upsample(img)


def adaptive_denoise(img):
    """改进的自适应降噪流程"""
    # 非局部均值降噪与双边滤波复合
    denoised = cv2.fastNlMeansDenoising(img, h=15, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


def dynamic_contrast_enhancement(img):
    """动态对比度增强模块"""
    # CLAHE与自适应伽马校正复合
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 动态计算clip limit
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = max(1.0, np.std(l) / 25)
    clahe.setClipLimit(cl)
    l_clahe = clahe.apply(l)

    # 自适应伽马校正
    gamma = 1.5 - (np.mean(l) / 255) * 0.5
    l_gamma = np.power(l_clahe / 255.0, gamma) * 255.0
    lab = cv2.merge((l_gamma.astype(np.uint8), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def hybrid_sharpen(img):
    """复合锐化算法"""
    # 拉普拉斯锐化与非锐化掩模复合
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    laplacian = cv2.filter2D(img, -1, kernel)

    # 边缘自适应权重
    edges = cv2.Canny(img, 50, 150) / 255.0
    weight_map = cv2.GaussianBlur(edges, (21, 21), 3)
    weight_map = np.clip(weight_map * 3.0, 0, 1)

    # 将权重矩阵转换为平均权重系数
    avg_weight = np.mean(weight_map)
    sharpened = cv2.addWeighted(img, 1.0, laplacian, 0.6 * avg_weight, 0)
    return sharpened


def text_edge_optimization(img):
    """文字边缘专项处理"""
    # 边缘定向增强
    edges = cv2.Canny(img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 边缘区域局部对比度增强
    mask = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 3) / 255.0
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 仅增强边缘区域的亮度分量
    y = np.clip(y + (y * mask * 0.3).astype(np.uint8), 0, 255)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)


def super_resolution_pipeline(input_path, output_path):
    # 初始化处理流程
    orig_image = cv2.imread(input_path)

    # 预处理阶段
    denoised = adaptive_denoise(orig_image)

    # 超分辨率处理
    sr_result = super_resolution(denoised)

    # 后处理流程
    enhanced = dynamic_contrast_enhancement(sr_result)
    sharpened = hybrid_sharpen(enhanced)
    final = text_edge_optimization(sharpened)
    Image.fromarray(final.astype('uint8')).save(output_path)


if __name__ == '__main__':
    input_path = r'outputs/b_output.jpg'
    output_path = 'outputs/b_output2.jpg'
    super_resolution_pipeline(input_path, output_path)