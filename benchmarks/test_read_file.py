import fsspec
import pyarrow
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
def s3_file():
    path = f"{S3_BASE_PATH}/data"
    with smart_open.open(path, "wb") as f:
        f.write(b"0" * 100_000_000)
    return path
    
    
        
@pytest.mark.benchmark
def test_smart_open_read(benchmark, s3_file):
    
    def func():
        with smart_open.open(s3_file, "rb") as f:
            f.read()
    
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)
    

@pytest.mark.benchmark
def test_pyarrow_s3(benchmark, s3_file):
    def func():
        filesystem, path = pyarrow.fs.FileSystem.from_uri(s3_file)
        with filesystem.open_input_file(path) as f:
            f.read()

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)

@pytest.mark.benchmark
def test_s3(benchmark, s3_file):
    def func():
        fs, urlpath = fsspec.url_to_fs(s3_file)
        with fs.open(urlpath) as f:
            f.read()

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=3)

