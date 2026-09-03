import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKER = BASE_DIR / 'train_keypoint_worker.py'
RUNNER_LOG_DIR = BASE_DIR / 'results' / 'fivefold'
RUNNER_LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    'ViTPose-small', 'HRNet-W32', 'LiteHRNet-18',
    'ResNet-50', 'Hourglass-52', 'HRFormer-small'
]
# GPU 0 has the larger 80C profile and receives folds 1, 3 and 5.
# GPU 1 has the 40C profile and receives folds 2 and 4.
GPU0_JOBS = [[model, fold] for model in MODELS for fold in [1, 3, 5]]
GPU1_JOBS = [[model, fold] for model in MODELS for fold in [2, 4]]

all_jobs = [tuple(item) for item in GPU0_JOBS + GPU1_JOBS]
assert len(all_jobs) == 30
assert len(set(all_jobs)) == 30
assert set(all_jobs) == {
    (model, fold)
    for model in MODELS
    for fold in range(1, 6)
}

environment0 = os.environ.copy()
environment0['CUDA_VISIBLE_DEVICES'] = '0'
environment0['TRAIN_JOBS'] = json.dumps(GPU0_JOBS)
environment0['PYTHONUNBUFFERED'] = '1'

environment1 = os.environ.copy()
environment1['CUDA_VISIBLE_DEVICES'] = '1'
environment1['TRAIN_JOBS'] = json.dumps(GPU1_JOBS)
environment1['PYTHONUNBUFFERED'] = '1'

log0 = open(RUNNER_LOG_DIR / 'gpu0_runner.log', 'a', buffering=1)
log1 = open(RUNNER_LOG_DIR / 'gpu1_runner.log', 'a', buffering=1)

process0 = subprocess.Popen(
    [sys.executable, str(WORKER)], cwd=BASE_DIR, env=environment0,
    stdout=log0, stderr=subprocess.STDOUT)
process1 = subprocess.Popen(
    [sys.executable, str(WORKER)], cwd=BASE_DIR, env=environment1,
    stdout=log1, stderr=subprocess.STDOUT)

print('GPU 0 process:', process0.pid, 'jobs:', GPU0_JOBS)
print('GPU 1 process:', process1.pid, 'jobs:', GPU1_JOBS)
print('Logs:', RUNNER_LOG_DIR / 'gpu0_runner.log', RUNNER_LOG_DIR / 'gpu1_runner.log')

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
print('All thirty keypoint model-fold jobs completed.')
