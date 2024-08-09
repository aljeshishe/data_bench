import boto3
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import urllib
from pathlib import Path
import shutil
import time
from loguru import logger
import human_readable as hr
from datetime import datetime 
import attr
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
import boto3

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

def write_dummy_dataset(path: str, mvalues: int, part_mvalues: int, cols: int):
    rows = part_mvalues * M // cols
    logger.info(f"Creating df with {cols=} {rows=}")
    file_path = f"{path}/0.npy"
    tmp_path = f"/tmp/0.npy"
    array = np.random.rand(rows, cols).astype(np.float32)
    np.save(tmp_path, array)
    upload_file(tmp_path, file_path)
    
    parts_count = mvalues // part_mvalues
    logger.info(f"Creating {parts_count} copies")
    with ThreadPoolExecutor(32) as pool:
        for i in range(1, parts_count):
            pool.submit(copy_file, src=file_path, dst=f"{path}/{i}.npy")


def upload_file(path, dst):
    dst_parsed = urlparse(dst)
    boto3.client('s3').upload_file(path, dst_parsed.netloc, dst_parsed.path.lstrip('/'))


def copy_file(src, dst):
    src_parsed = urlparse(src)
    dst_parsed = urlparse(dst)
    boto3.client('s3').copy_object(
        CopySource=dict(Bucket=src_parsed.netloc, Key=src_parsed.path.lstrip('/')),
        Bucket=dst_parsed.netloc,
        Key=dst_parsed.path.lstrip('/')
    )
    
