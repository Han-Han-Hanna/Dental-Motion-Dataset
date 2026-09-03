import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS = ['AttU-Net', 'CMU-Net', 'CMUNeXt', 'U-Net', 'U-Net++', 'UNet 3+']
GPU0_JOBS = [[model, fold] for model in MODELS for fold in [1, 3, 5]]
GPU1_JOBS = [[model, fold] for model in MODELS for fold in [2, 4]]
all_jobs = [tuple(item) for item in GPU0_JOBS + GPU1_JOBS]
assert len(all_jobs) == 30 and len(set(all_jobs)) == 30

log_dir = BASE_DIR / 'results' / 'segmentation_fivefold'
log_dir.mkdir(parents=True, exist_ok=True)
environment0 = os.environ.copy()
environment0['CUDA_VISIBLE_DEVICES'] = '0'
environment0['TRAIN_JOBS'] = json.dumps(GPU0_JOBS)
environment0['PYTHONUNBUFFERED'] = '1'
environment1 = os.environ.copy()
environment1['CUDA_VISIBLE_DEVICES'] = '1'
environment1['TRAIN_JOBS'] = json.dumps(GPU1_JOBS)
environment1['PYTHONUNBUFFERED'] = '1'
log0 = open(log_dir / 'gpu0_runner.log', 'a', buffering=1)
log1 = open(log_dir / 'gpu1_runner.log', 'a', buffering=1)
process0 = subprocess.Popen([sys.executable, str(BASE_DIR / 'train_segmentation_worker.py')],
                            cwd=BASE_DIR, env=environment0, stdout=log0, stderr=subprocess.STDOUT)
process1 = subprocess.Popen([sys.executable, str(BASE_DIR / 'train_segmentation_worker.py')],
                            cwd=BASE_DIR, env=environment1, stdout=log1, stderr=subprocess.STDOUT)
print('GPU 0 process:', process0.pid, 'jobs:', GPU0_JOBS)
print('GPU 1 process:', process1.pid, 'jobs:', GPU1_JOBS)
while True:
    code0 = process0.poll()
    code1 = process1.poll()
    if code0 is not None and code0 != 0:
        if code1 is None:
            process1.terminate()
        raise RuntimeError(f'GPU 0 worker failed with exit code {code0}')
    if code1 is not None and code1 != 0:
        if code0 is None:
            process0.terminate()
        raise RuntimeError(f'GPU 1 worker failed with exit code {code1}')
    if code0 == 0 and code1 == 0:
        break
    time.sleep(30)
log0.close()
log1.close()
print('All thirty segmentation model-fold jobs completed.')
