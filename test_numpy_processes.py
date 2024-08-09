from contextlib import contextmanager
from functools import partial
import time
from urllib.parse import urlparse
import click
import ray
import ray.data
import torch
from torch.utils.data import DataLoader, Dataset
from data_bench import utils
from data_bench.utils import M
from tqdm import tqdm
from loguru import logger
import dotenv 
import os 
import io
import numpy as np

import torch
import numpy as np
import s3fs
from multiprocessing import Pool
import boto3


dotenv.load_dotenv()

def load_and_process_npy(path):
    bucket, _, path = path.partition("/")
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=path)
    with io.BytesIO(obj["Body"].read()) as f:
        f.seek(0)
        return torch.from_numpy(np.load(f, allow_pickle=True))

def dataset_iterator(s3_dir, workers: int = os.cpu_count()):
    s3_paths = s3fs.S3FileSystem().ls(s3_dir)
    s3_paths = [path for path in s3_paths if path.endswith('.npy')]

    with Pool(processes=workers) as pool:
        try:
            yield from pool.imap_unordered(load_and_process_npy, s3_paths)
        finally:
            pool.terminate()
            
            
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    s3_uri = "s3://ab-users/grachev/ray_benchmark/20gb.parquet"
    mvalues = 9000 # 20 GB
    part_mvalues = 8
    cols = 100

    if create_dataset:
        utils.remove(s3_uri)
        utils.write_dummy_dataset(s3_uri, mvalues=mvalues, part_mvalues=part_mvalues, cols=cols)


    total_rows = 0
    with tqdm(unit="Mvalues") as pbar:
        iterator = dataset_iterator(s3_uri, workers=30)
        total_start_ts = time.time()
        for tensor in iterator:
            total_rows += tensor.shape[0]
            pbar.update(tensor.shape[0] * cols // M)
    total_mvalues_per_sec = cols * total_rows / M / (time.time() - total_start_ts)
    logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")


if __name__ == "__main__":
    main()
    