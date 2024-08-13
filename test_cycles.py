from urllib.parse import urljoin, urlparse
import numpy as np
import torch
import pandas as pd
import click
import dotenv
from loguru import logger
from data_bench import utils
import smart_open
dotenv.load_dotenv()
    
def create_numpy(base_path, rows: int, cols: int):
    path = f"{base_path}/data.npy"
    logger.info(f"Creating {path}")
    with smart_open.open(path, "wb") as f:
        np.save(f, utils.pandas_df(rows=rows, cols=cols).to_numpy())
    
def create_parquet(base_path, rows: int, cols: int):
    path = f"{base_path}/data.parquet"
    logger.info(f"Creating {path}")
    with smart_open.open(path, "wb") as f:
        utils.pandas_df(rows=rows, cols=cols).to_parquet(f)

def create_tensor(base_path, rows: int, cols: int):
    path = f"{base_path}/data.pt"
    logger.info(f"Creating {path}")
    with smart_open.open(path, "wb") as f:
        torch.save(torch.from_numpy(utils.pandas_df(rows=rows, cols=cols).to_numpy()), f)
    
    
    
def read_numpy(base_path):
    path = f"{base_path}/data.npy"
    logger.info(f"Reading {path}")
    with smart_open.open(path, "rb") as f:
        tensor = torch.from_numpy(np.load(f, allow_pickle=True))
        print(f"tensor={tensor.shape}")

# def read_parquet(base_path):
#     path = f"{base_path}/data.parquet"
#     logger.info(f"Reading {path}")
#     tensor = torch.from_numpy(pd.read_parquet(path).to_numpy())
#     print(f"tensor={tensor.shape}")

def read_parquet(base_path):
    path = f"{base_path}/data.parquet"
    logger.info(f"Reading {path}")
    with smart_open.open(path, "rb") as f:
        tensor = torch.from_numpy(pd.read_parquet(f).to_numpy())
        print(f"tensor={tensor.shape}")

def read_tensor(base_path):
    path = f"{base_path}/data.pt"
    logger.info(f"Reading {path}")
    with smart_open.open(path, "rb") as f:
        tensor = torch.load(f=f)
        print(f"tensor={tensor.shape}")

    

@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
@click.option("--numpy", is_flag=True)
@click.option("--parquet", is_flag=True)
@click.option("--tensor", is_flag=True)
@click.option("--base-path", default="s3://tmp-grachev")
@click.option("--rows", default=800_000)
@click.option("--cols", default=64)
def main(create_dataset, numpy, parquet, tensor, base_path, rows:int, cols:int):
    
    if create_dataset:
        if numpy:
            create_numpy(base_path=base_path, rows=rows, cols=cols)
        if parquet:
            create_parquet(base_path=base_path, rows=rows, cols=cols)
        if tensor:
            create_tensor(base_path=base_path, rows=rows, cols=cols)
    if numpy:
        read_numpy(base_path=base_path)
    if parquet:
        read_parquet(base_path=base_path)
    if tensor:
        read_tensor(base_path=base_path)
        
if __name__ == "__main__":
    main()
    
# results
# parquet with smart_open real 3.7
# parquet without smart_open real 6.3
# tensor real 6.3
# numpy real 5.141