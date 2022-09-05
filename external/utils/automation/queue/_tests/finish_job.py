from utils.progress import ProgressBar
from time import sleep

ITERATIONS = 10
pbar = ProgressBar(ITERATIONS)
for i in range(ITERATIONS):
    sleep(2)
    pbar.step(1)
print('[INFO] Success')