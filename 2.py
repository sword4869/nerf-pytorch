from tqdm import tqdm
from time import sleep

with tqdm(range(10, 5)) as pbar:
    for i in pbar:
        pbar.set_description(f'{i}')
        sleep(1)