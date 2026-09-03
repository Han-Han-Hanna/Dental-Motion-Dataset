import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class ToothSegmentationDataset(Dataset):
    def __init__(self, data_root, list_path, image_size=256, training=False):
        self.data_root = Path(data_root)
        self.names = Path(list_path).read_text().split()
        self.image_size = image_size
        self.training = training
        self.color_to_class = {
            (0, 0, 0): 0,
            (0, 0, 128): 1,
            (64, 0, 0): 2,
            (128, 128, 128): 3,
            (0, 128, 128): 4,
            (128, 128, 0): 5,
            (128, 0, 128): 6,
            (0, 128, 0): 7,
            (128, 0, 0): 8,
        }
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(self.data_root / 'JPG Images' / f'{name}.jpg').convert('RGB')
        mask_image = Image.open(self.data_root / 'PNG Segmentation Masks' / f'{name}.png')
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_array = np.array(mask_image)
        if mask_array.ndim == 2:
            mask = mask_array.astype(np.int64)
        else:
            mask = np.zeros(mask_array.shape[:2], dtype=np.int64)
            for color, class_id in self.color_to_class.items():
                mask[np.all(mask_array[:, :, :3] == np.array(color), axis=2)] = class_id
        if mask.min() < 0 or mask.max() > 8:
            raise ValueError(f'Invalid mask classes in {name}: {np.unique(mask)}')
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor, name
