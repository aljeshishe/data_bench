import io
from multiprocessing.reduction import ForkingPickler
import pickle
import pyarrow
from ray.util.multiprocessing import Pool
import os
import fsspec
from functools import partial
import time
import click
import torch
from tqdm import tqdm
from data_bench import utils
from data_bench.utils import M
from loguru import logger
import dotenv 
import numpy as np

import torch
import numpy as np


dotenv.load_dotenv()


def s3_npy(path):
    from multiprocessing.reduction import ForkingPickler
    if path.startswith("s3://"):
        fs = pyarrow.fs.S3FileSystem(endpoint_override=f"http://s3.ap-northeast-1.amazonaws.com")
    else:
        fs = pyarrow.fs.LocalFileSystem()
    start_ts = time.time()
    with fs.open_input_file(path) as f:
        tensor = torch.from_numpy(np.load(io.BytesIO(f.read()), allow_pickle=True)).to(device="cuda:0")
    
    # print(f"elapsed_time={time.time() - start_ts:.2f} seconds")
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(dict(data=tensor))
    return buf.getvalue()

def random_tensors(path):
    tensor = torch.rand(100_000, 68, device="cuda:0")
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(dict(data=tensor))
    return buf.getvalue()

def dataset_iterator(fs, s3_url, pool):
    s3_paths = fs.ls(s3_url)
    s3_paths = [path for path in s3_paths if path.endswith('.npy')]
    yield from pool.imap_unordered(partial(s3_npy), s3_paths)
            
def unpickler(iter):
    for item in iter:
        yield pickle.loads(item)
        
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    path = "s3://ab-users/grachev/ray_benchmark/20gb_100000rows_numpy"
    path = "/opt/dlami/nvme/ray_benchmark/20gb_100000rows_numpy"
    
    rows = 100_000
    cols = 68
    n_files = 700
    workers = 10

    fs, _ = fsspec.url_to_fs(path)

    if create_dataset:
        utils.remove(path)
        file_url = utils.write_numpy_dataset(fs=fs, path=path, rows=rows, cols=cols)
        utils.clone_file(file_url, n_files=n_files)

    if path.startswith == "/":
        logger.info("Dropping caches")
        os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches')

    import ray
    ray.init(num_cpus=workers, logging_level="INFO", include_dashboard=True, dashboard_host="0.0.0.0", num_gpus=1)
    with Pool(processes=workers, ray_remote_args=dict(num_gpus=.01)) as pool:
        iterator = dataset_iterator(fs=fs, s3_url=path, pool=pool)
        utils.benchmark(utils.iter_timeit(unpickler(iterator)), total_mvalues=rows * cols * n_files // 1024**2)


if __name__ == "__main__":
    main()

# Total mvalues/s=1295.11 first=2.50 - create with tensor.rand, 2 workers
# Total mvalues/s=453.45  first=9.18 - create with tensor.rand, 40 workers
# CONCLUSIONS:
# many workers slow down transfer data from workers
# first batch delay is not because of data transfer
# np.load(io.BytesIO(f.read()) - increases speed
# larger files increase speed 
# Total mvalues/s=183.58 first=3.70  - s3_npy, 10 workers
# Total mvalues/s=286.41 first=5.27  - s3_npy, 20 workers
# Total mvalues/s=302.14 first=9.11  - s3_npy, 40 workers
# Total mvalues/s=783.02 first=3.01 - s3_npy local files, 10 workers