from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from millify import millify
import numpy as np
import torch 
from ray.util.multiprocessing import Pool
import ray


def func_torch(size):
    cols = 100
    return torch.zeros((size // cols, cols), dtype=torch.float32)

def func_numpy(size):
    cols = 100
    return np.zeros((size // cols, cols), dtype=np.float32)

def func_bytes(size):
    return 'b' * size


def main(pool, func, descr):
    with pool:
        size = 80_000 * 100
        
        count = 1000
        start = time.time()
        for _ in pool.map(func, [size] * count):
            pass
            # print(_.shape)
        elapsed = time.time() - start
        total = size * count
        print(f"Time={elapsed:.2f} {millify(total)=} Speed={millify(total / elapsed)}values/s - {descr} func={func.__name__}")
        
    
if __name__ == "__main__":
    workers = 10
    for func in [func_torch, func_numpy, func_bytes]:
        main(pool=ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")), func=func, descr="ProcessPoolExecutor spawn")
        main(pool=ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")), func=func, descr="ProcessPoolExecutor fork")
        main(pool=mp.get_context("spawn").Pool(processes=workers), func=func, descr="Pool spawn")
        main(pool=mp.get_context("fork").Pool(processes=workers), func=func, descr="Poll fork")
        main(pool=Pool(processes=workers), func=func, descr="ray pool")
        