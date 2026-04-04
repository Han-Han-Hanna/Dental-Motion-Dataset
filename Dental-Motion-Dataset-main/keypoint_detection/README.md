# 🦷 Dental Keypoint Detection (MMPose-based)

This project focuses on **dental keypoint detection** based on the MMPose framework.
We provide dataset usage instructions, environment setup, and training guidelines to ensure full reproducibility.

---

## 📦 1. Dataset Preparation

The dataset is hosted on Zenodo.

### 🔽 Download

Please download the dataset from Zenodo:

```
https://doi.org/10.5281/zenodo.19201948
```
📁 Organize Dataset

After downloading, please arrange the dataset into the following structure:

```text
data/
├── data/
│   ├── annotations/
│   │   ├── person_keypoints_train.json
│   │   └── person_keypoints_val.json
│   ├── train/
│   │   └── images
│   └── val/
│       └── images
├── train_txtjson/
│   ├── txt
│   └── json
└── val_txtjson/
    ├── txt
    └── json
```

---

## 🛠️ 2. Environment Setup

```bash
pip install -r requirements.txt
```

---

## 📥 3. Install MMPose

Please install MMPose from the official repository:

👉 https://github.com/open-mmlab/mmpose

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

---

## 🔧 4. Prepare Config File

You need to replace the default config with the provided custom config adapted for dental keypoints.

Steps:

1. Copy our config file 
2. Place it under:

```
mmpose/configs/
```

3. Make sure:

   * `data_root = 'data/data'`
   * `num_keypoints = 35`
   * Annotation paths are correct

---

## 🚀 5. Training

Use the following command format:

```bash
python tools/train.py configs/your_config.py
```

### ✅ Example

```bash
python tools/train.py configs/vitpose_small_teeth.py
```

---

## 🧪 6. Testing

```bash
python tools/test.py configs/your_config.py work_dirs/xxx/best.pth
```

---

## 🙌 Acknowledgement

This project is built upon the excellent framework:

* MMPose (OpenMMLab)

---
