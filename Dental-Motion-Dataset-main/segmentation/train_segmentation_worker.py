import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segmentation_dataset import ToothSegmentationDataset
from segmentation_models import build_segmentation_model


def evaluate(model, loader, criterion, device, num_classes, prediction_dir=None, prediction_names=None):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, masks)
            predictions = logits.argmax(1)
            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
            valid = (masks >= 0) & (masks < num_classes)
            encoded = masks[valid] * num_classes + predictions[valid]
            confusion += torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
            if prediction_dir is not None:
                for prediction, name in zip(predictions.cpu().numpy(), names):
                    if prediction_names is None or name in prediction_names:
                        np.save(Path(prediction_dir) / f'{name}.npy', prediction.astype(np.uint8))
    true_positive = confusion.diag()
    false_positive = confusion.sum(0) - true_positive
    false_negative = confusion.sum(1) - true_positive
    foreground = slice(1, num_classes)
    epsilon = 1e-8
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    sensitivity = true_positive / (true_positive + false_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative + epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + epsilon)
    metrics = {
        'val_loss': loss_sum / max(sample_count, 1),
        'val_iou': iou[foreground].mean().item(),
        'DSC': dice[foreground].mean().item(),
        'val_SE': sensitivity[foreground].mean().item(),
        'val_PC': precision[foreground].mean().item(),
        'val_F1': f1[foreground].mean().item(),
        'val_ACC': (true_positive.sum() / confusion.sum().clamp_min(1)).item(),
    }
    return metrics


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
DATA_ROOT = BASE_DIR / 'extracted' / '数据' / 'Core Annotated Data' / 'Tooth Segmentation Data'
SPLIT_ROOT = BASE_DIR / 'segmentation_fivefold'
RESULT_ROOT = BASE_DIR / 'results' / 'segmentation_fivefold'
ASSIGNED_JOBS = json.loads(os.environ['TRAIN_JOBS'])
TARGETS = {'0014', '0225', '0736', '1035'}
NUM_CLASSES = 9
EPOCHS = 210
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 1e-4
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 4

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Each worker must see exactly one CUDA GPU')
device = torch.device('cuda:0')
print('Visible GPU:', torch.cuda.get_device_name(0))
print('Assigned jobs:', ASSIGNED_JOBS)

for model_name, fold in ASSIGNED_JOBS:
    random.seed(4200 + fold)
    np.random.seed(4200 + fold)
    torch.manual_seed(4200 + fold)
    torch.cuda.manual_seed_all(4200 + fold)
    torch.backends.cudnn.benchmark = True

    fold_dir = SPLIT_ROOT / f'fold{fold}'
    work_dir = RESULT_ROOT / model_name / f'fold{fold}'
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / 'best_test_metrics.json'
    if result_path.exists():
        print('[skip]', model_name, 'fold', fold)
        continue

    train_dataset = ToothSegmentationDataset(DATA_ROOT, fold_dir / 'train.txt', training=True)
    val_dataset = ToothSegmentationDataset(DATA_ROOT, fold_dir / 'val.txt', training=False)
    test_dataset = ToothSegmentationDataset(DATA_ROOT, fold_dir / 'test.txt', training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = build_segmentation_model(model_name, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 1
    best_iou = -1.0
    best_epoch = 0
    early_stop_best_iou = -1.0
    epochs_without_improvement = 0
    last_path = work_dir / 'last.pth'
    best_path = work_dir / 'best_val_iou.pth'
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        best_iou = state['best_iou']
        best_epoch = state['best_epoch']
        early_stop_best_iou = state.get('early_stop_best_iou', best_iou)
        epochs_without_improvement = state.get('epochs_without_improvement', 0)

    log_path = work_dir / 'train_log.csv'
    write_header = not log_path.exists() or start_epoch == 1
    with open(log_path, 'a', newline='', buffering=1) as log_file:
        fields = ['epoch', 'train_loss', 'lr', 'val_loss', 'val_iou', 'DSC', 'val_SE', 'val_PC', 'val_F1', 'val_ACC']
        writer = csv.DictWriter(log_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            train_count = 0
            for images, masks, _ in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * images.size(0)
                train_count += images.size(0)
            scheduler.step()
            metrics = evaluate(model, val_loader, criterion, device, NUM_CLASSES)
            row = {
                'epoch': epoch,
                'train_loss': train_loss / max(train_count, 1),
                'lr': optimizer.param_groups[0]['lr'],
                **metrics,
            }
            writer.writerow(row)
            print(model_name, 'fold', fold, 'epoch', epoch, row)
            if metrics['val_iou'] > best_iou:
                best_iou = metrics['val_iou']
                best_epoch = epoch
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics}, best_path)
                with open(work_dir / 'best_val_metrics.json', 'w', encoding='utf-8') as file:
                    json.dump({'epoch': epoch, **metrics}, file, indent=2)
            if metrics['val_iou'] > early_stop_best_iou + EARLY_STOP_MIN_DELTA:
                early_stop_best_iou = metrics['val_iou']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            finite_metrics = all(np.isfinite(value) for value in row.values())
            should_stop = (
                epochs_without_improvement >= EARLY_STOP_PATIENCE
                or not finite_metrics
            )
            if epoch % 10 == 0 or epoch == EPOCHS or should_stop:
                torch.save({
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'best_iou': best_iou, 'best_epoch': best_epoch,
                    'early_stop_best_iou': early_stop_best_iou,
                    'epochs_without_improvement': epochs_without_improvement,
                }, last_path)
            if should_stop:
                reason = 'non-finite metric' if not finite_metrics else 'no validation improvement'
                print(model_name, 'fold', fold, 'early stop at epoch', epoch,
                      'reason', reason, 'best epoch', best_epoch,
                      'best validation IoU', best_iou)
                break

    best_state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model'])
    prediction_dir = work_dir / 'test_predictions'
    prediction_dir.mkdir(exist_ok=True)
    test_metrics = evaluate(model, test_loader, criterion, device, NUM_CLASSES, prediction_dir, TARGETS)
    test_metrics['best_epoch'] = best_state['epoch']
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(test_metrics, file, indent=2)
    print('[done]', model_name, 'fold', fold, test_metrics)
