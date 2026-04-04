import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
            num_classes=9,
            color_map=None  # 彩色标签→类别索引的映射表（必须包含0~num_classes-1）
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.color_map = color_map if color_map is not None else {}

        # 校验：确保color_map包含所有类别（0~num_classes-1）
        required_classes = set(range(num_classes))
        provided_classes = set(self.color_map.keys())
        assert required_classes == provided_classes, \
            f"color_map缺失类别！需包含{required_classes}，实际包含{provided_classes}"

        # 1. 读取训练/验证样本列表
        if self.split == "train":
            file_path = os.path.join(self._base_dir, train_file_dir)
        elif self.split == "val":
            file_path = os.path.join(self._base_dir, val_file_dir)
        else:
            raise ValueError(f"无效的split: {self.split}，仅支持 'train' 或 'val'")

        with open(file_path, "r", encoding="utf-8") as f:
            # 过滤空行，确保样本名有效
            self.sample_list = [line.strip() for line in f if line.strip()]
        assert len(self.sample_list) > 0, f"{split}样本列表{file_path}为空！"

        # 2. 校验数据目录存在性
        self.image_dir = os.path.join(self._base_dir, "images")
        self.mask_dir = os.path.join(self._base_dir, "masks")
        assert os.path.exists(self.image_dir), f"图像目录不存在: {self.image_dir}"
        assert os.path.exists(self.mask_dir), f"标签目录不存在: {self.mask_dir}"

        # 3. 打印数据集基本信息（仅保留关键统计，无冗余）
        print(f"[{split}] 加载完成：{len(self.sample_list)}个样本 | 类别数：{num_classes}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 获取当前样本名称（假设txt中仅存文件名，不含后缀）
        case_name = self.sample_list[idx]

        # -------------------------- 1. 读取图像（RGB格式）--------------------------
        image_path = os.path.join(self.image_dir, f"{case_name}.jpg")
        image = cv2.imread(image_path)
        assert image is not None, f"样本{case_name}：无法读取图像 {image_path}"
        # OpenCV默认BGR，转换为模型常用的RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------- 2. 读取彩色标签并转类别索引 --------------------------
        mask_path = os.path.join(self.mask_dir, f"{case_name}.png")
        color_mask = cv2.imread(mask_path)  # 读取彩色标签（OpenCV默认BGR格式）
        assert color_mask is not None, f"样本{case_name}：无法读取标签 {mask_path}"
        # 获取原始标签尺寸（后续数据增强会修改，暂存备用）
        orig_h, orig_w = color_mask.shape[:2]

        # 初始化类别索引标签（单通道，值为0~num_classes-1）
        label_idx = np.zeros((orig_h, orig_w), dtype=np.int64)
        for class_idx, (b, g, r) in self.color_map.items():
            # 生成当前颜色的掩码（匹配BGR值）
            color_mask_bool = np.all(color_mask == [b, g, r], axis=2)
            # 为掩码区域赋值对应的类别索引
            label_idx[color_mask_bool] = class_idx

        # 校验标签合法性（避免无效类别索引）
        assert np.max(label_idx) < self.num_classes, \
            f"样本{case_name}：标签含无效类别索引 {np.max(label_idx)}（最大允许 {self.num_classes - 1}）"
        assert np.min(label_idx) >= 0, \
            f"样本{case_name}：标签含负索引 {np.min(label_idx)}"

        # -------------------------- 3. 数据增强（图像+标签同步变换）--------------------------
        if self.transform is not None:
            # 使用albumentations的Compose，确保image和mask同步增强
            augmented = self.transform(image=image, mask=label_idx)
            image = augmented["image"]  # 增强后的图像
            label_idx = augmented["mask"]  # 增强后的类别索引标签
            # 关键：获取增强后的尺寸（替代原始尺寸，避免后续独热编码尺寸不匹配）
            aug_h, aug_w = label_idx.shape
        else:
            # 无增强时，使用原始尺寸
            aug_h, aug_w = orig_h, orig_w

        # -------------------------- 4. 图像预处理（适配模型输入）--------------------------
        # 归一化：将像素值从0~255缩放到0~1（模型训练更稳定）
        image = image.astype(np.float32) / 255.0
        # 维度转换：从[H, W, C]（图像常规格式）转为[C, H, W]（PyTorch输入格式）
        image = np.transpose(image, (2, 0, 1))

        # -------------------------- 5. 标签预处理（独热编码，适配多类别损失）--------------------------
        # 初始化独热编码标签（形状：[num_classes, H, W]）
        label_one_hot = np.zeros((self.num_classes, aug_h, aug_w), dtype=np.float32)
        for c in range(self.num_classes):
            # 为每个类别生成二进制掩码（1表示该类别，0表示其他）
            label_one_hot[c] = (label_idx == c).astype(np.float32)

        # -------------------------- 6. 返回样本（含关键元数据，便于调试）--------------------------
        sample = {
            "image": image,               # 预处理后的图像 [C, H, W]
            "label": label_one_hot,       # 独热编码标签 [num_classes, H, W]
            "label_idx": label_idx,       # 类别索引标签 [H, W]（备用，如可视化）
            "case_name": case_name,       # 样本名称（调试时定位问题）
            "idx": idx                    # 样本在数据集中的索引
        }
        return sample