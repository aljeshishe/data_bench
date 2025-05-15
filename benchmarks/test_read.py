from functools import partial
import time
import numpy as np
from safetensors import safe_open
import safetensors
import torch
import pandas as pd
import click
import dotenv
from loguru import logger
from data_bench import utils
import smart_open

import numpy as np 
import random
import time
import smart_open
import pytest
from loguru import logger

S3_BASE_PATH = "s3://tmp-grachev"
ROWS = 800_000
COLS = 64

@pytest.fixture(scope="session")
def s3_numpy_file():
    path = f"{S3_BASE_PATH}/data.npy"
    with smart_open.open(path, "wb") as f:
        np.save(f, utils.pandas_df(rows=ROWS, cols=COLS).to_numpy())
    return path
    
    
@pytest.fixture(scope="session")
def s3_parquet_file():
    path = f"{S3_BASE_PATH}/data.parquet"
    logger.info(f"Creating {path}")
    with smart_open.open(path, "wb") as f:
        utils.pandas_df(rows=ROWS, cols=COLS).to_parquet(f)
    return path


@pytest.fixture(scope="session")
def s3_tensor_file():
    path = f"{S3_BASE_PATH}/data.pt"
    logger.info(f"Creating {path}")
    with smart_open.open(path, "wb") as f:
        torch.save(torch.from_numpy(utils.pandas_df(rows=ROWS, cols=COLS).to_numpy()), f)
    return path

    
@pytest.fixture(scope="session")
def local_safetensors_file():
    from safetensors.torch import save_file
    path = f"/tmp/data.safetensors"
    logger.info(f"Creating {path}")
    tensors = dict(tensor=torch.from_numpy(utils.pandas_df(rows=ROWS, cols=COLS).to_numpy()))
    save_file(tensors, path)
    return path

    
@pytest.fixture(scope="session")
def s3_safetensors_file():
    from safetensors.torch import save_file
    path = f"{S3_BASE_PATH}/data.safetensors"
    logger.info(f"Creating {path}")
    tensors = dict(tensor=torch.from_numpy(utils.pandas_df(rows=ROWS, cols=COLS).to_numpy()))
    local_path = f"/tmp/data.safetensors"
    save_file(tensors, local_path)
    with smart_open.open(local_path, 'rb') as fin, smart_open.open(path, 'wb') as fout:
        fout.write(fin.read())
    return path
    
    
        
@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_numpy(benchmark, s3_numpy_file, device):
    
    def func():
        path = s3_numpy_file
        with smart_open.open(path, "rb") as f:
            tensor = torch.from_numpy(np.load(f, allow_pickle=True)).to(device=device)
    
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)
    

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_parquet(benchmark, s3_parquet_file, device):
    def func():
        path = s3_parquet_file
        with smart_open.open(path, "rb") as f:
            tensor = torch.from_numpy(pd.read_parquet(f).to_numpy()).to(device="cuda:0")

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_tensor(benchmark, s3_tensor_file, device):
    def func():
        path = s3_tensor_file
        with smart_open.open(path, "rb") as f:
            tensor = torch.load(f=f, map_location=lambda storage, loc: storage.cuda(0))
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_safetensors_local(benchmark, local_safetensors_file, device):
    def func():
        path = local_safetensors_file
        with safe_open(path, framework="pt", device="cuda:0") as f:
            tensor = f.get_tensor("tensor")

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_safetensors(benchmark, s3_safetensors_file, device):
    def func():
        path = s3_safetensors_file
        with smart_open.open(path, "rb") as f:
            tensor = safetensors.torch.load(f.read())["tensor"].to(device="cpu")

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)
