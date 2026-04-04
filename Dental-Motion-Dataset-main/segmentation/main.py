import os
import warnings
import cv2
import random
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import functional as F_tv

# 警告过滤配置
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# 解决Windows多进程问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from src.dataloader.dataset import MedicalDataSets
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score

# 模型导入
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


# 1. 预处理类
class TrainTransform:
    def __init__(self, img_size):
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img, label):
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = F_tv.rotate(img, angle)
            label = F_tv.rotate(label, angle)

        if random.random() > 0.5:
            img = F_tv.hflip(img)
            label = F_tv.hflip(label)

        img = F_tv.resize(img, (self.img_size, self.img_size), interpolation=F_tv.InterpolationMode.BILINEAR)
        label = F_tv.resize(label, (self.img_size, self.img_size), interpolation=F_tv.InterpolationMode.NEAREST)

        img = F_tv.to_tensor(img)
        label = torch.tensor(np.array(label), dtype=torch.long)  # 转为张量
        img = self.normalize(img)

        return img, label


class ValTransform:
    def __init__(self, img_size):
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img, label):
        img = F_tv.resize(img, (self.img_size, self.img_size), interpolation=F_tv.InterpolationMode.BILINEAR)
        label = F_tv.resize(label, (self.img_size, self.img_size), interpolation=F_tv.InterpolationMode.NEAREST)

        img = F_tv.to_tensor(img)
        label = torch.tensor(np.array(label), dtype=torch.long)  # 转为张量
        img = self.normalize(img)

        return img, label


# 2. 损失函数类
class MultiClassDiceCELoss(torch.nn.Module):
    def __init__(self, num_classes, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def dice_loss(self, input, target):
        dice_total = 0.0
        for c in range(self.num_classes):
            input_c = input[:, c, :, :]
            target_c = target[:, c, :, :]
            intersection = (input_c * target_c).sum()
            union = input_c.sum() + target_c.sum() + 1e-6
            dice = (2.0 * intersection + 1e-6) / union
            dice_total += dice
        return 1.0 - (dice_total / self.num_classes)

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=1)
        dice_loss = self.dice_loss(input_soft, target)
        target_label = torch.argmax(target, dim=1)
        ce_loss = F.cross_entropy(input, target_label, weight=self.weight)
        return dice_loss + ce_loss


# 3. 随机种子设置
def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 4. 数据集类（核心修复：添加标签值范围检查）
class PreloadMedicalDataSets(MedicalDataSets):
    def __init__(self, base_dir, split, transform, train_file_dir, val_file_dir, num_classes, color_map):
        super().__init__(base_dir, split, transform, train_file_dir, val_file_dir, num_classes, color_map)

        self.base_dir = base_dir
        self.split = split
        self.train_file_dir = self._get_full_path(train_file_dir)
        self.val_file_dir = self._get_full_path(val_file_dir)
        self.num_classes = num_classes  # 类别数（如9），标签值必须 < 该值

        # 图像和标签路径配置
        self.img_dir = os.path.join(base_dir, "images")
        self.label_dir = os.path.join(base_dir, "masks")
        self.img_suffix = ".jpg"
        self.label_suffix = ".png"

        # 获取样本列表
        self.samples = self._get_samples_from_parent()
        print(f"[{split}] 数据集加载完成，共{len(self.samples)}个样本")

    def _get_full_path(self, file_dir):
        full_path = os.path.join(self.base_dir, file_dir)
        if os.path.exists(full_path):
            return full_path
        if os.path.exists(file_dir):
            return file_dir
        return file_dir

    def _get_samples_from_parent(self):
        samples = []
        file_dir = self.train_file_dir if self.split == "train" else self.val_file_dir

        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                f"样本列表文件不存在！\n"
                f"尝试的路径：\n1. {os.path.join(self.base_dir, file_dir)}\n2. {file_dir}"
            )

        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            sample_id = line
            img_path = os.path.join(self.img_dir, f"{sample_id}{self.img_suffix}")
            label_path = os.path.join(self.label_dir, f"{sample_id}{self.label_suffix}")

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在：{img_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"标签文件不存在：{label_path}")

            samples.append((img_path, label_path, sample_id))
        return samples

    def __getitem__(self, idx):
        img_path, label_path, case_name = self.samples[idx]

        # 读取图像并转换为PIL格式
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像文件：{img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # 读取标签并转换为PIL格式
        label = cv2.imread(label_path, 0)  # 灰度图
        if label is None:
            raise ValueError(f"无法读取标签文件：{label_path}")
        label = Image.fromarray(label)

        # 应用变换（转为张量）
        if self.transform is not None:
            img, label = self.transform(img, label)  # label此时为torch.Tensor

        # 核心修复：检查标签值范围
        max_label = torch.max(label)
        if max_label >= self.num_classes:
            raise ValueError(
                f"样本 {case_name} 的标签值超出范围！\n"
                f"标签最大值为 {max_label}，但类别数为 {self.num_classes}（标签值必须 < {self.num_classes}）\n"
                f"标签文件路径：{label_path}"
            )

        # 转换为one-hot编码
        label_onehot = F.one_hot(label, num_classes=self.num_classes).permute(2, 0, 1).float()
        return {
            'image': img,
            'label': label_onehot,
            'case_name': case_name
        }

    def __len__(self):
        return len(self.samples)


# 5. 模型加载
def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    else:
        model = get_transformer_based_model(
            parser=parser, model_name=args.model,
            img_size=args.img_size, num_classes=args.num_classes, in_ch=3
        ).cuda()
    device = next(model.parameters()).device
    assert device.type == 'cuda', f"模型未加载到GPU！当前设备：{device}"
    print(f"模型加载完成，设备：{device}（GPU型号：{torch.cuda.get_device_name(device)}）")
    return model


# 6. 数据加载器
def getDataloader(args):
    img_size = args.img_size if args.model != "SwinUnet" else 224

    train_transform = TrainTransform(img_size)
    val_transform = ValTransform(img_size)

    db_train = PreloadMedicalDataSets(
        base_dir=args.base_dir,
        split="train",
        transform=train_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        num_classes=args.num_classes,
        color_map=args.color_map
    )
    db_val = PreloadMedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=val_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        num_classes=args.num_classes,
        color_map=args.color_map
    )

    print(f"训练集样本数：{len(db_train)}，验证集样本数：{len(db_val)}")

    # Windows单进程模式
    train_workers = 0
    val_workers = 0

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        drop_last=True
    )
    valloader = DataLoader(
        db_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        drop_last=False
    )

    return trainloader, valloader


# 7. 主训练逻辑
def main(args):
    trainloader, valloader = getDataloader(args=args)
    model = get_model(args)

    print(f"训练样本列表：{args.train_file_dir}，验证样本列表：{args.val_file_dir}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )
    criterion = MultiClassDiceCELoss(num_classes=args.num_classes).cuda()
    scaler = GradScaler()

    best_iou = 0.0
    iter_num = 0
    max_epoch = args.epoch
    max_iterations = len(trainloader) * max_epoch
    print(f"总训练迭代次数：{max_iterations}（{max_epoch}轮 × {len(trainloader)}batch/轮）")

    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {
            'loss': AverageMeter(), 'iou': AverageMeter(),
            'val_loss': AverageMeter(), 'val_iou': AverageMeter(),
            'val_SE': AverageMeter(), 'val_PC': AverageMeter(),
            'val_F1': AverageMeter(), 'val_ACC': AverageMeter()
        }

        # 训练轮次
        for i_batch, sampled_batch in enumerate(trainloader):
            img_batch = sampled_batch['image'].cuda(non_blocking=True)
            label_batch = sampled_batch['label'].cuda(non_blocking=True)

            with autocast():
                outputs = model(img_batch)
                loss = criterion(outputs, label_batch)

            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))

            if (i_batch + 1) % 10 == 0:
                print(f"Epoch [{epoch_num}/{max_epoch}] Batch [{i_batch + 1}/{len(trainloader)}] "
                      f"Loss: {avg_meters['loss'].avg:.4f} IOU: {avg_meters['iou'].avg:.4f} "
                      f"LR: {lr_:.6f}")

        # 验证轮次
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                img_batch = sampled_batch['image'].cuda(non_blocking=True)
                label_batch = sampled_batch['label'].cuda(non_blocking=True)
                case_names = sampled_batch['case_name']

                outputs = model(img_batch)
                loss = criterion(outputs, label_batch)
                iou, _, SE, PC, F1, _, ACC = iou_score(outputs, label_batch)

                if args.debug and (i_batch + 1) % 5 == 0:
                    pred_idx = torch.argmax(outputs, dim=1)
                    true_idx = torch.argmax(label_batch, dim=1)
                    print(f"\n===== 验证集 Epoch {epoch_num} Batch {i_batch + 1} =====")
                    print(f"样本名称: {case_names[0]}（仅显示第一个样本）")
                    print(f"  预测类别范围: {pred_idx[0].min().item()} ~ {pred_idx[0].max().item()}")
                    print(f"  真实类别范围: {true_idx[0].min().item()} ~ {true_idx[0].max().item()}")

                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

        # 轮次结果打印
        print(
            f"\n==================== Epoch [{epoch_num}/{max_epoch}] 结果 ===================="
            f"\n训练集：Loss={avg_meters['loss'].avg:.4f}, IOU={avg_meters['iou'].avg:.4f}"
            f"\n验证集：Loss={avg_meters['val_loss'].avg:.4f}, IOU={avg_meters['val_iou'].avg:.4f}"
            f"\n        SE={avg_meters['val_SE'].avg:.4f}, PC={avg_meters['val_PC'].avg:.4f}"
            f"\n        F1={avg_meters['val_F1'].avg:.4f}, ACC={avg_meters['val_ACC'].avg:.4f}"
            f"\n=========================================================================\n"
        )

        # 保存最优模型
        if avg_meters['val_iou'].avg > best_iou:
            os.makedirs("./checkpoint", exist_ok=True)
            save_path = f'checkpoint/{args.model}_epoch{epoch_num}_iou{avg_meters["val_iou"].avg:.4f}.pth'
            torch.save(model.state_dict(), save_path)
            best_iou = avg_meters['val_iou'].avg
            print(f"=> 保存最优模型到：{save_path}（当前最佳IOU：{best_iou:.4f}）")

    return f"Training Finished! 最佳验证IOU：{best_iou:.4f}"


# 入口函数
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="U_Net",
                        choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net",
                                 "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT"], help='模型名称')
    parser.add_argument('--base_dir', type=str, default=r"D:\yuyifenge\pythonProject2\yuyi\data\tee",
                        help='数据根目录')
    parser.add_argument('--train_file_dir', type=str, default="tee_train.txt", help='训练样本列表文件名')
    parser.add_argument('--val_file_dir', type=str, default="tee_val.txt", help='验证样本列表文件名')
    parser.add_argument('--base_lr', type=float, default=0.04, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epoch', type=int, default=300, help='训练轮数')
    parser.add_argument('--img_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--num_classes', type=int, default=9, help='类别数（标签值必须小于该值）')
    parser.add_argument('--seed', type=int, default=41, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='打印调试信息')

    args = parser.parse_args()
    args.color_map = {
        0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 0, 64), 3: (128, 128, 128),
        4: (128, 128, 0), 5: (0, 128, 128), 6: (128, 0, 128),
        7: (0, 128, 0), 8: (0, 0, 128)
    }

    seed_torch(args.seed)
    result = main(args)
    print(result)