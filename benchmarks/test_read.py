from functools import partial
import io
import time
import numpy as np
import pyarrow
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
ROUNDS = 5 
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
    
    
def s3_read(path):
    fs = pyarrow.fs.S3FileSystem(endpoint_override=f"http://s3.ap-northeast-1.amazonaws.com")
    with fs.open_input_file(path.replace("s3://", "")) as f:
        return f.read()
    
@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_numpy(benchmark, s3_numpy_file, device):
    
    def func():
        tensor = torch.from_numpy(np.load(io.BytesIO(s3_read(path=s3_numpy_file)), allow_pickle=True)).to(device=device)
        print(tensor[0])
    
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)
    

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_parquet(benchmark, s3_parquet_file, device):
    def func():
        data = s3_read(path=s3_parquet_file)
        stream = io.BytesIO(data)
        df = pd.read_parquet(stream)
        array = df.to_numpy()
        tensor = torch.from_numpy(array)
        tensor = tensor.to(device=device)
        print(tensor[0])

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_tensor(benchmark, s3_tensor_file, device):
    def func():
        data = s3_read(path=s3_tensor_file)
        stream = io.BytesIO(data)
        tensor = torch.load(stream, map_location=lambda storage, loc: storage.cuda(0))
        print(tensor[0])
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_safetensors_local(benchmark, local_safetensors_file, device):
    def func():
        path = local_safetensors_file
        with safe_open(path, framework="pt", device=device) as f:
            tensor = f.get_tensor("tensor")
            print(tensor[0])

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)

@pytest.mark.benchmark
def test_read_safetensors_local_move2gpu(benchmark, local_safetensors_file):
    def func():
        path = local_safetensors_file
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor("tensor").to("cuda:0")
            print(tensor[0])

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_safetensors(benchmark, s3_safetensors_file, device):
    def func():
        tensor = safetensors.torch.load(s3_read(path=s3_safetensors_file))["tensor"].to(device=device)
        print(tensor[0])

    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)

# --------------------------------------------------------------------------------------------------------------- benchmark: 11 tests ---------------------------------------------------------------------------------------------------------------
# Name (time in us)                                   Min                       Max                      Mean                  StdDev                    Median                     IQR            Outliers         OPS            Rounds  Iterations
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# test_read_safetensors_local[cpu]               665.8290 (1.0)            809.9790 (1.0)            705.1192 (1.0)           59.9933 (1.0)            676.0290 (1.0)           55.7632 (1.0)           1;0  1,418.1999 (1.0)           5           1
# test_read_safetensors_local_move2gpu        30,475.2710 (45.77)       30,768.6400 (37.99)       30,570.6648 (43.36)        117.3969 (1.96)        30,547.1200 (45.19)        132.9305 (2.38)          1;0     32.7111 (0.02)          5           1
# test_read_safetensors_local[cuda:0]         30,598.8000 (45.96)       35,487.0520 (43.81)       31,939.6184 (45.30)      2,036.4821 (33.95)       31,211.9300 (46.17)      2,032.4905 (36.45)         1;0     31.3091 (0.02)          5           1
# test_read_numpy[cpu]                       287,919.7680 (432.42)     409,283.1560 (505.30)     348,325.1366 (493.99)    55,021.8655 (917.13)     332,133.9170 (491.30)   100,380.5360 (>1000.0)       2;0      2.8709 (0.00)          5           1
# test_read_numpy[cuda:0]                    313,325.0010 (470.58)     860,142.6820 (>1000.0)    440,219.1262 (624.32)   235,960.0461 (>1000.0)    332,015.5540 (491.13)   177,541.2107 (>1000.0)       1;1      2.2716 (0.00)          5           1
# test_read_safetensors[cpu]                 380,726.4440 (571.81)     474,153.9570 (585.39)     421,861.9284 (598.28)    36,198.6994 (603.38)     412,901.9930 (610.78)    52,055.6463 (933.51)        2;0      2.3704 (0.00)          5           1
# test_read_tensor[cuda:0]                   405,078.7360 (608§.38)     466,662.1490 (576.14)     437,348.5326 (620.25)    26,437.6158 (440.68)     449,429.7890 (664.81)    43,609.6525 (782.05)        2;0      2.2865 (0.00)          5           1
# test_read_tensor[cpu]                      411,487.3730 (618.01)     477,672.2560 (589.73)     440,217.0268 (624.32)    24,016.6711 (400.32)     434,558.1050 (642.81)    23,435.5483 (420.27)        2;0      2.2716 (0.00)          5           1
# test_read_safetensors[cuda:0]              427,775.3730 (642.47)     516,168.2480 (637.26)     460,507.1048 (653.09)    34,659.6866 (577.73)     459,096.1210 (679.11)    43,367.9367 (777.72)        1;0      2.1715 (0.00)          5           1
# test_read_parquet[cpu]                   1,272,138.3570 (>1000.0)  1,325,498.7220 (>1000.0)  1,300,327.5512 (>1000.0)   22,388.3506 (373.18)   1,298,170.6480 (>1000.0)   38,410.5685 (688.82)        2;0      0.7690 (0.00)          5           1
# test_read_parquet[cuda:0]                1,276,821.4260 (>1000.0)  1,498,115.4510 (>1000.0)  1,331,898.6608 (>1000.0)   93,526.8284 (>1000.0)  1,289,693.1220 (>1000.0)   69,164.6370 (>1000.0)       1;1      0.7508 (0.00)          5           1
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
