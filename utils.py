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

def remove(path: Path | str):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()
    elif str(path).startswith("s3"):
        parsed = urllib.parse.urlparse(path)
        boto3.client("s3").delete_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))


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
    df = pd.DataFrame(np.random.rand(rows, cols))
    df.columns = [f"col_{i}" for i in range(cols)]
    return df
                