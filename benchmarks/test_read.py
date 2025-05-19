import io
import time
import numpy as np
import pyarrow
from safetensors import safe_open
import safetensors
import torch
import pandas as pd
from loguru import logger
from data_bench import utils

import numpy as np 
import time
import pytest
from loguru import logger

S3_BASE_PATH = "s3://tmp-grachev"
ROWS = 800_000
COLS = 64
ROUNDS = 5

def get_fs(path):
    if path.startswith("s3://"):
        return pyarrow.fs.S3FileSystem(endpoint_override=f"http://s3.ap-northeast-1.amazonaws.com"), path.replace("s3://", "")
    return pyarrow.fs.LocalFileSystem(), path

def get_path(path_type):
    match path_type:
        case "local":
            return f"/tmp/data.npy"
        case "local_ssd":
            return f"/opt/dlami/nvme/data.npy"
        case "s3":
            return f"{S3_BASE_PATH}/data.npy"

@pytest.fixture(scope="session", params=["local", "local_ssd", "s3"])
def numpy_file(request):
    path = get_path(path_type=request.param)
    fs, _path = get_fs(path=path)
    with fs.open_output_stream(path=_path) as fp:
        np.save(fp, utils.pandas_df(rows=ROWS, cols=COLS).to_numpy())
    return path
    
    
@pytest.fixture(scope="session", params=["local", "local_ssd", "s3"])
def parquet_file(request):
    path = get_path(path_type=request.param)
    fs, _path = get_fs(path=path)
    with fs.open_output_stream(path=_path) as fp:
        utils.pandas_df(rows=ROWS, cols=COLS).to_parquet(fp)
    return path


@pytest.fixture(scope="session", params=["local", "local_ssd", "s3"])
def tensor_file(request):
    path = get_path(path_type=request.param)
    fs, _path = get_fs(path=path)
    with fs.open_output_stream(path=_path) as fp:
        torch.save(torch.from_numpy(utils.pandas_df(rows=ROWS, cols=COLS).to_numpy()), fp)
    return path

    
@pytest.fixture(scope="session", params=["local", "local_ssd", "s3"])
def safetensors_file(request):
    path = get_path(path_type=request.param)
    fs, _path = get_fs(path=path)
    with fs.open_output_stream(path=_path) as fp:
        tensors = dict(tensor=torch.from_numpy(utils.pandas_df(rows=ROWS, cols=COLS).to_numpy()))
        data = safetensors.torch.save(tensors=tensors)
        fp.write(data)
    return path

    
    
def read_file(path):
    fs, path = get_fs(path=path)
    with fs.open_input_file(path.replace("s3://", "")) as f:
        return f.read()
    
@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_numpy(benchmark, numpy_file, device):
    
    def func():
        tensor = torch.from_numpy(np.load(io.BytesIO(read_file(path=numpy_file)), allow_pickle=True)).to(device=device)
        print(tensor[0])
    
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)
    

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_parquet(benchmark, parquet_file, device):
    def func():
        data = read_file(path=parquet_file)
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
def test_read_tensor(benchmark, tensor_file, device):
    def func():
        data = read_file(path=tensor_file)
        stream = io.BytesIO(data)
        tensor = torch.load(stream, map_location=device)
        print(tensor[0])
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_read_safetensors(benchmark, safetensors_file, device):
    def func():
        if safetensors_file.startswith("s3://"):
            tensor = safetensors.torch.load(read_file(path=safetensors_file))["tensor"].to(device=device)
        else:
            with safe_open(safetensors_file, framework="pt", device=device) as f:
                tensor = f.get_tensor("tensor")
                print(tensor[0])
            
    benchmark._timer = time.process_time
    benchmark.pedantic(func, iterations=1, rounds=ROUNDS)

# ---------------------------------------------------------------------------------------------------- benchmark: 24 tests -----------------------------------------------------------------------------------------------------
# Name (time in ms)                                  Min                   Max                  Mean              StdDev                Median                 IQR            Outliers         OPS            Rounds  Iterations
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# test_read_numpy[local-cpu]                    165.7200 (240.71)     168.7009 (202.89)     167.0351 (225.80)     1.4222 (26.83)      166.3316 (237.07)     2.5912 (50.04)         1;0      5.9868 (0.00)          5           1
# test_read_numpy[local-cuda:0]                 183.0714 (265.91)     601.1675 (722.99)     267.6814 (361.85)   186.4258 (>1000.0)    184.7963 (263.39)   104.8542 (>1000.0)       1;1      3.7358 (0.00)          5           1
# test_read_numpy[local_ssd-cpu]                165.2719 (240.06)     169.5395 (203.90)     166.6110 (225.22)     1.8250 (34.43)      165.6661 (236.12)     2.5110 (48.49)         1;0      6.0020 (0.00)          5           1
# test_read_numpy[local_ssd-cuda:0]             182.8296 (265.56)     184.1595 (221.48)     183.2944 (247.78)     0.6118 (11.54)      182.9107 (260.70)     0.9908 (19.13)         1;0      5.4557 (0.00)          5           1
# test_read_numpy[s3-cpu]                       292.8066 (425.30)     342.1404 (411.47)     312.5909 (422.56)    19.3468 (364.97)     311.0202 (443.29)    27.2040 (525.38)        2;0      3.1991 (0.00)          5           1
# test_read_numpy[s3-cuda:0]                    323.2447 (469.51)     377.0978 (453.52)     343.4016 (464.21)    20.8044 (392.47)     337.1159 (480.48)    25.2617 (487.87)        1;0      2.9120 (0.00)          5           1
# test_read_parquet[local-cpu]                  801.3518 (>1000.0)  1,011.5277 (>1000.0)    911.0008 (>1000.0)   75.4632 (>1000.0)    921.4657 (>1000.0)   77.1878 (>1000.0)       2;0      1.0977 (0.00)          5           1
# test_read_parquet[local-cuda:0]               864.0469 (>1000.0)  1,172.5321 (>1000.0)    978.7363 (>1000.0)  133.1158 (>1000.0)    908.0589 (>1000.0)  207.6212 (>1000.0)       1;0      1.0217 (0.00)          5           1
# test_read_parquet[local_ssd-cpu]              960.6673 (>1000.0)  1,018.5878 (>1000.0)    997.8000 (>1000.0)   23.7151 (447.37)     995.8414 (>1000.0)   31.8214 (614.55)        1;0      1.0022 (0.00)          5           1
# test_read_parquet[local_ssd-cuda:0]           974.3286 (>1000.0)  1,086.8373 (>1000.0)  1,005.3856 (>1000.0)   46.5466 (878.08)     989.0645 (>1000.0)   43.8793 (847.42)        1;1      0.9946 (0.00)          5           1
# test_read_parquet[s3-cpu]                   1,326.2019 (>1000.0)  1,349.8289 (>1000.0)  1,338.6454 (>1000.0)    8.6181 (162.58)   1,338.0361 (>1000.0)   10.0220 (193.55)        2;0      0.7470 (0.00)          5           1
# test_read_parquet[s3-cuda:0]                1,295.3057 (>1000.0)  1,426.1463 (>1000.0)  1,358.9539 (>1000.0)   47.8454 (902.58)   1,360.7663 (>1000.0)   58.1033 (>1000.0)       2;0      0.7359 (0.00)          5           1
# test_read_safetensors[local-cpu]                0.6885 (1.0)          1.0190 (1.23)         0.7625 (1.03)       0.1436 (2.71)         0.7016 (1.0)        0.0894 (1.73)          1;1  1,311.5305 (0.97)          5           1
# test_read_safetensors[local-cuda:0]            30.0179 (43.60)       36.6086 (44.03)       31.6230 (42.75)      2.8225 (53.25)       30.3377 (43.24)      2.4605 (47.52)         1;1     31.6225 (0.02)          5           1
# test_read_safetensors[local_ssd-cpu]            0.7007 (1.02)         0.8315 (1.0)          0.7398 (1.0)        0.0530 (1.0)          0.7279 (1.04)       0.0518 (1.0)           1;0  1,351.8007 (1.0)           5           1
# test_read_safetensors[local_ssd-cuda:0]        29.8993 (43.43)       35.8656 (43.13)       32.4203 (43.83)      2.2000 (41.50)       32.4852 (46.30)      2.3407 (45.20)         2;0     30.8449 (0.02)          5           1
# test_read_safetensors[s3-cpu]                 372.0109 (540.35)     470.6330 (566.00)     405.9504 (548.76)    38.9383 (734.55)     394.9260 (562.88)    46.2129 (892.49)        1;0      2.4634 (0.00)          5           1
# test_read_safetensors[s3-cuda:0]              392.4284 (570.00)     492.0773 (591.79)     419.7663 (567.44)    41.5449 (783.73)     404.3105 (576.25)    41.7885 (807.04)        1;0      2.3823 (0.00)          5           1
# test_read_tensor[local-cpu]                   240.2891 (349.02)     242.7457 (291.94)     241.2315 (326.10)     0.9980 (18.83)      241.0987 (343.63)     1.4946 (28.86)         1;0      4.1454 (0.00)          5           1
# test_read_tensor[local-cuda:0]                258.7728 (375.87)     294.3837 (354.04)     271.5932 (367.14)    13.8253 (260.81)     266.3693 (379.65)    15.7960 (305.06)        1;0      3.6820 (0.00)          5           1
# test_read_tensor[local_ssd-cpu]               239.3703 (347.68)     240.4649 (289.19)     239.7674 (324.12)     0.4641 (8.76)       239.5878 (341.48)     0.7211 (13.93)         1;0      4.1707 (0.00)          5           1
# test_read_tensor[local_ssd-cuda:0]            255.3709 (370.93)     258.7524 (311.19)     257.4392 (348.01)     1.4776 (27.87)      258.0691 (367.82)     2.4630 (47.57)         1;0      3.8844 (0.00)          5           1
# test_read_tensor[s3-cpu]                      372.7774 (541.46)     496.0748 (596.60)     414.1481 (559.85)    50.9883 (961.87)     385.2131 (549.03)    65.5344 (>1000.0)       1;0      2.4146 (0.00)          5           1
# test_read_tensor[s3-cuda:0]                   409.6312 (594.99)     569.5835 (685.01)     483.8551 (654.08)    61.0896 (>1000.0)    488.7357 (696.58)    86.3567 (>1000.0)       2;0      2.0667 (0.00)          5           1
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONCLUSIONS:
# - local and local_ssd are the sam