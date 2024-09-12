import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import boto3
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import urllib
from pathlib import Path
import shutil
import time
from loguru import logger
import human_readable as hr
import attr
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm

GB = 1024 * 1024 * 1024
MB = 1024 * 1024

G = 1024 * 1024 * 1024
M = 1024 * 1024

def remove(path:  str):
    if str(path).startswith("s3"):
        parsed = urllib.parse.urlparse(path)
        boto3.resource('s3').Bucket(parsed.netloc).objects.filter(Prefix=parsed.path.lstrip("/")).delete()
        return 
        
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

class Stopwatch:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

    def __str__(self):
        return f"{self.elapsed:.2f} seconds"
    

@attr.define
class Params:
    rows: int
    cols: int
    cleanup: bool = True
    
    def __attrs_post_init__(self):
        logger.info(f"Params:")
        for k, v in attr.asdict(self).items():
            logger.info(f"{k}={v}")
        logger.info(f"vals={hr.int_word(self.rows * self.cols)}")

def dask_df(rows, cols):
    data = da.random.random((rows, cols)).astype('float32')
    column_names = [f'col_{i}' for i in range(cols)]
    return dd.from_dask_array(data, columns=column_names).persist()

def pandas_df(rows, cols):
    df = pd.DataFrame(np.random.rand(rows, cols).astype('float32'))
    df.columns = [f"col_{i}" for i in range(cols)]
    return df

def write_parquet_dataset(path: str, mvalues: int, cols: int):
    rows = mvalues * M // cols
    logger.info(f"Creating df with {cols=} {rows=}")
    file_path = f"{path}/0.parquet"
    pandas_df(rows=rows, cols=cols).to_parquet(file_path)
    return file_path

def write_tensors_parquet_dataset(path: str, mvalues: int, cols: int):
    rows = mvalues * M // cols
    logger.info(f"Creating df with {cols=} {rows=}")
    file_path = f"{path}/0.parquet"
    array = np.random.rand(rows, cols).astype(np.float32).tolist()
    pd.DataFrame(dict(X=array)).to_parquet(file_path)
    return file_path

def write_tensors_parquet_dataset_new(path: str, mvalues: int, cols: int):
    rows = mvalues * M // cols
    logger.info(f"Creating df with {cols=} {rows=}")
    file_path = f"{path}/0.parquet"
    arr = np.random.rand(rows, cols).astype(np.float32)
    extension_array = pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
    table = pa.table(dict(X=extension_array))
    pq.write_table(table, file_path)
    return file_path

      
def write_numpy_dataset(path: str, mvalues: int, cols: int):
    rows = mvalues * M // cols
    logger.info(f"Creating df with {cols=} {rows=}")
    file_path = f"{path}/0.npy"
    tmp_path = f"/tmp/0.npy"
    array = np.random.rand(rows, cols).astype(np.float32)
    np.save(tmp_path, array)
    upload_file(tmp_path, file_path)
    return file_path
    
def clone_s3_file(file_path:str, n_files:int):
    logger.info(f"Creating {n_files} copies")
    file_name, _, ext = file_path.rpartition(".")
    with ThreadPoolExecutor(32) as pool:
        for i in range(1, n_files):
            pool.submit(copy_file, src=file_path, dst=f"{file_name}_{i}.{ext}")


def upload_file(path, dst):
    dst_parsed = urlparse(dst)
    boto3.client("s3").upload_file(path, dst_parsed.netloc, dst_parsed.path.lstrip("/"))


def copy_file(src, dst):
    src_parsed = urlparse(src)
    dst_parsed = urlparse(dst)
    boto3.client("s3").copy_object(
        CopySource=dict(Bucket=src_parsed.netloc, Key=src_parsed.path.lstrip("/")),
        Bucket=dst_parsed.netloc,
        Key=dst_parsed.path.lstrip("/")
    )
    
def s3_dir_size(s3_uri: str) -> int:
    s3 = s3fs.S3FileSystem()

    file_list = s3.ls(s3_uri)
    total_size = 0
    for file in file_list:
        file_info = s3.info(file)
        if not file_info["type"] == "directory":
            total_size += file_info["size"]
    return total_size


def iter_timeit(iter, verbose=False):
    start_ts = time.time()
    elapsed_values = []
    for batch in iter:
        elapsed = time.time() - start_ts
        elapsed_values.append(elapsed)
        if verbose:
            logger.info(f"Batch creation time {elapsed:.2f}s")
        yield batch
        start_ts = time.time()
    
    logger.info("Percentiles")
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        p_val = np.percentile(elapsed_values, p)
        logger.info(f"p{p}={p_val:.2f}")
        
                
def benchmark(ds, total_mvalues=None):
    counter = 0
    with tqdm(unit="Mvalues", total=total_mvalues) as pbar:
        total_start_ts = time.time()
        logger.info("start")
        for batch in ds:
            n_values = sum(tensor.numel() for tensor in batch.values())
            pbar.update(n_values // 1000**2)
            counter += n_values
        total_mvalues_per_sec = counter // 1000**2  / (time.time() - total_start_ts)
        logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")
        
def show_dataloader_info(dataloader):
    batch = next(iter(dataloader))
    batch_info_str = " ".join(f"{k}:{v.shape}" for k, v in batch.items())
    logger.info(f"batch: {batch_info_str}")
    