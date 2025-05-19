import io
import os
import fsspec
from functools import partial
import time
import click
import pyarrow
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

def get_fs(path):
    if path.startswith("s3://"):
        return pyarrow.fs.S3FileSystem(endpoint_override=f"http://s3.ap-northeast-1.amazonaws.com"), path.replace("s3://", "")
    return pyarrow.fs.LocalFileSystem(), path

def load_and_process_npy(path, fs):
    fs, _path = get_fs(path)
    start_ts = time.time()
    with fs.open_input_file(path) as f:
        tensor = torch.from_numpy(np.load(io.BytesIO(f.read()), allow_pickle=True)).to(device="cuda:0")
    # sprint(f"elapsed_time={time.time() - start_ts:.2f} seconds")
    return dict(data=tensor)

def dataset_iterator(fs, s3_url, pool):
    s3_paths = fs.ls(s3_url)
    s3_paths = [path for path in s3_paths if path.endswith('.npy')]
    yield from pool.imap_unordered(partial(load_and_process_npy, fs=fs), s3_paths)
            
            
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    path = "s3://ab-users/grachev/ray_benchmark/40gb_100000rows_numpy"
    # path = "/opt/dlami/nvme/40gb_100000rows_numpy"
    
    rows = 100_000 
    cols = 68
    n_files = 1470 
    workers = 40
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
    total_size = 0
    with mp.get_context("forkserver").Pool(processes=workers) as pool:
        iterator = dataset_iterator(fs=fs, s3_url=path, pool=pool)
        total_start_ts = time.time()
        # for tensor in utils.iter_timeit(tqdm(iterator), verbose=False):
        #     total_size += tensor.numel()
        utils.benchmark(utils.iter_timeit(iterator), total_mvalues=rows * cols * n_files // 1024**2)
    # elapsed_time = time.time() - total_start_ts
    # print(f"Total time: {elapsed_time:.2f} seconds")
    # mvalues_per_sec = rows * cols * n_files / elapsed_time // (1024 ** 2)
    # print(f"{mvalues_per_sec=} ")
            #print(tensor.shape)
            #total_rows += tensor.shape[0]
            # pbar.update(tensor.shape[0] * cols // M)
    # total_mvalues_per_sec = cols * total_rows / M / (time.time() - total_start_ts)
    # logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")


if __name__ == "__main__":
    main()
    