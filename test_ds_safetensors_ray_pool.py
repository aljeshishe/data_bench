import io
from multiprocessing.reduction import ForkingPickler
import pickle
import pyarrow
import ray
from ray.util.multiprocessing import Pool
import os
import fsspec
from functools import partial
import time
import click
import safetensors
import torch
from tqdm import tqdm
from data_bench import utils
from data_bench.utils import M
from loguru import logger
import dotenv 
import numpy as np
import ray

import torch
import numpy as np


dotenv.load_dotenv()

def get_fs(path):
    if path.startswith("s3://"):
        return pyarrow.fs.S3FileSystem(endpoint_override=f"http://s3.ap-northeast-1.amazonaws.com"), path.replace("s3://", "")
    return pyarrow.fs.LocalFileSystem(), path

def read_safetensors(path):
    # start_ts = time.time()
    with safetensors.safe_open(path, framework="pt", device="cuda:0") as fp:
        tensor = fp.get_tensor("X")
    # print(f"elapsed_time={time.time() - start_ts:.2f} seconds")
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(dict(data=tensor))
    return buf.getvalue()

def random_tensors(path):
    tensor = torch.rand(100_000, 68, device="cuda:0")
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(dict(data=tensor))
    return buf.getvalue()

@ray.remote(num_gpus=.01)
def read_safetensors_remote(path):
    return read_safetensors(path=path)

def dataset_iterator(fs, s3_url, pool):
    s3_paths = fs.ls(s3_url)
    s3_paths = [path for path in s3_paths if path.endswith('.safetensors')]
    yield from pool.imap_unordered(read_safetensors, s3_paths)
            
def unpickler(iter):    
    for item in iter:
        if not isinstance(item, bytes):
            logger.error(f"Item is not bytes: {item}")
            continue
        yield pickle.loads(item)
        
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    # path = "s3://ab-users/grachev/ray_benchmark/20gb_100000rows_safetensors"
    path = "/opt/dlami/nvme/20gb_100000rows_numpy"
    # path = "/tmp/20gb_100000rows_numpy"
    
    rows = 100_000
    cols = 68
    n_files = 700 * 8
    workers = 5

    fs, _ = fsspec.url_to_fs(path)

    if create_dataset:
        utils.remove(path)
        file_url = utils.write_safetensors_dataset(fs=fs, path=path, rows=rows, cols=cols)
        utils.clone_file(file_url, n_files=n_files)

    if path.startswith("/"):
        logger.info("Dropping caches")
        os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches')

    ray.init(num_cpus=workers, logging_level="INFO", include_dashboard=True, dashboard_host="0.0.0.0", num_gpus=1)
    with Pool(processes=workers, ray_remote_args=dict(num_gpus=.01)) as pool:
        iterator = dataset_iterator(fs=fs, s3_url=path, pool=pool)
        utils.benchmark(utils.iter_timeit(unpickler(iterator)), total_mvalues=rows * cols * n_files // 1024**2)


if __name__ == "__main__":
    main()

#  Total mvalues/s=486.65 irst=3.65  - 5 workers on g4dn.8xlarge ssd with 2GB/s max speed
# Total mvalues/s=453.45  first=9.18 - create with tensor.rand, 40 workers