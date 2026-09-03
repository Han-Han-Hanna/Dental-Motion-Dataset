import glob
import json
import os
import copy
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import torch
from mmengine.config import Config
from mmengine.runner import Runner

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
DATA_ROOT = str(BASE_DIR / 'tooth_keypoint_coco')
RESULT_ROOT = str(BASE_DIR / 'results' / 'fivefold')
MODEL_CONFIGS = {
    'ViTPose-small': str(BASE_DIR / 'models' / 'vitpose_small.py'),
    'HRNet-W32': str(BASE_DIR / 'models' / 'hrnet_w32.py'),
    'LiteHRNet-18': str(BASE_DIR / 'models' / 'litehrnet_18.py'),
    'ResNet-50': str(BASE_DIR / 'models' / 'resnet50.py'),
    'Hourglass-52': str(BASE_DIR / 'models' / 'hourglass52.py'),
    'HRFormer-small': str(BASE_DIR / 'models' / 'hrformer_small.py'),
}
ASSIGNED_JOBS = json.loads(os.environ['TRAIN_JOBS'])

os.makedirs(RESULT_ROOT, exist_ok=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Each worker must see exactly one CUDA GPU')
print('Visible GPU:', torch.cuda.get_device_name(0))
print('Assigned jobs:', ASSIGNED_JOBS)

for model_name, fold in ASSIGNED_JOBS:
    config_path = MODEL_CONFIGS[model_name]
    fold_dir = os.path.join(DATA_ROOT, 'fivefold', f'fold{fold}')
    work_dir = os.path.join(RESULT_ROOT, model_name, f'fold{fold}')
    result_json = os.path.join(work_dir, 'best_test_metrics.json')
    if os.path.exists(result_json):
        print(f'[skip] {model_name} fold{fold} already completed')
        continue

    os.makedirs(work_dir, exist_ok=True)
    cfg = Config.fromfile(config_path)
    if 'PROJECT_ROOT' in cfg:
        cfg.PROJECT_ROOT = str(cfg.PROJECT_ROOT)
    # Imported helper symbols are not valid entries in a dumped Python config.
    if 'Path' in cfg:
        cfg.pop('Path')
    cfg.work_dir = work_dir
    cfg.randomness = dict(seed=42 + fold, deterministic=False)
    cfg.load_from = None
    cfg.resume = bool(glob.glob(os.path.join(work_dir, 'epoch_*.pth')))
    cfg.train_dataloader = copy.deepcopy(cfg.train_dataloader)
    cfg.val_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = copy.deepcopy(cfg.test_dataloader)
    cfg.val_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.test_evaluator = copy.deepcopy(cfg.test_evaluator)
    cfg.custom_hooks = copy.deepcopy(cfg.get('custom_hooks', []))
    cfg.custom_hooks.append(dict(
        type='EarlyStoppingHook',
        monitor='NME',
        rule='less',
        min_delta=1e-5,
        patience=30,
        strict=True,
        check_finite=True,
    ))
    cfg.train_cfg.max_epochs = 210
    cfg.train_cfg.val_interval = 1
    cfg.env_cfg.cudnn_benchmark = True
    cfg.train_dataloader.num_workers = 4
    cfg.val_dataloader.num_workers = 4
    cfg.test_dataloader.num_workers = 4
    cfg.val_dataloader.batch_size = 32
    cfg.test_dataloader.batch_size = 32
    cfg.train_dataloader.persistent_workers = True
    cfg.val_dataloader.persistent_workers = True
    cfg.test_dataloader.persistent_workers = True

    train_ann = os.path.join(fold_dir, 'train.json')
    val_ann = os.path.join(fold_dir, 'val.json')
    test_ann = os.path.join(fold_dir, 'test.json')
    cfg.train_dataloader.dataset.data_root = DATA_ROOT
    cfg.train_dataloader.dataset.ann_file = train_ann
    cfg.train_dataloader.dataset.data_prefix = dict(img='all/')
    cfg.val_dataloader.dataset.data_root = DATA_ROOT
    cfg.val_dataloader.dataset.ann_file = val_ann
    cfg.val_dataloader.dataset.data_prefix = dict(img='all/')
    cfg.test_dataloader.dataset.data_root = DATA_ROOT
    cfg.test_dataloader.dataset.ann_file = test_ann
    cfg.test_dataloader.dataset.data_prefix = dict(img='all/')

    for evaluator in cfg.val_evaluator:
        if evaluator.get('type') == 'CocoMetric':
            evaluator.ann_file = val_ann
            evaluator.outfile_prefix = os.path.join(work_dir, 'validation_predictions')
    for evaluator in cfg.test_evaluator:
        if evaluator.get('type') == 'CocoMetric':
            evaluator.ann_file = test_ann
            evaluator.outfile_prefix = os.path.join(work_dir, 'best_predictions')

    cfg.default_hooks.checkpoint.save_best = 'NME'
    cfg.default_hooks.checkpoint.rule = 'less'
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2

    cfg.dump(os.path.join(work_dir, 'fold_config.py'))
    runner = Runner.from_cfg(cfg)
    runner.train()

    best_files = sorted(glob.glob(os.path.join(work_dir, 'best_NME_epoch_*.pth')))
    if not best_files:
        raise FileNotFoundError(f'best checkpoint missing: {model_name} fold{fold}')

    cfg.load_from = max(best_files, key=os.path.getmtime)
    cfg.resume = False
    test_runner = Runner.from_cfg(cfg)
    metrics = test_runner.test()
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'[done] {model_name} fold{fold}: {metrics}')
