import os
import fsspec
from functools import partial
import time
import click
import torch
from data_bench import utils
from data_bench.utils import M
from tqdm import tqdm
from loguru import logger
import dotenv 
import numpy as np

import torch
import numpy as np
# from ray.util.multiprocessing import Pool

from torch import multiprocessing as mp

dotenv.load_dotenv()


def load_and_process_npy(path, fs):
    from torch import multiprocessing as mp
    with fs.open(path, "rb") as f:
        return torch.from_numpy(np.load(f, allow_pickle=True))

def dataset_iterator(fs, s3_url, pool):
    s3_paths = fs.ls(s3_url)
    s3_paths = [path for path in s3_paths if path.endswith('.npy')]
    yield from pool.imap_unordered(partial(load_and_process_npy, fs=fs), s3_paths)
    # yield from pool.map(load_and_process_npy, s3_paths)
            
            
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    path = "s3://ab-users/grachev/ray_benchmark/40gb_100000rows_numpy"
    # path = "/opt/dlami/nvme/40gb_100000rows_numpy"
    
    rows = 100_000
    cols = 68
    n_files = 1470
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)

    if create_dataset:
        utils.remove(path)
        file_url = utils.write_numpy_dataset(fs=fs, path=path, rows=rows, cols=cols)
        utils.clone_file(file_url, n_files=n_files)

    if protocol == "file":
        logger.info("Dropping caches")
        os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches')

    total_rows = 0
    
    logger.info("Starting pool")
    with mp.get_context("forkserver").Pool(processes=30) as pool:
        iterator = dataset_iterator(fs=fs, s3_url=path, pool=pool)
        total_start_ts = time.time()
        for tensor in utils.iter_timeit(tqdm(iterator), verbose=False):
            pass
    print(f"Total time: {time.time() - total_start_ts}")
            #print(tensor.shape)
            #total_rows += tensor.shape[0]
            # pbar.update(tensor.shape[0] * cols // M)
    # total_mvalues_per_sec = cols * total_rows / M / (time.time() - total_start_ts)
    # logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")


if __name__ == "__main__":
    main()
    