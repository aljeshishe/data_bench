from collections import defaultdict
import time
from typing import Any, Dict
import click
import numpy as np
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

def collate_fn(batch: Dict[str, np.ndarray]) -> Any:
    groups = defaultdict(list)
    for col, values in batch.items():
        if col == "__index_level_0__":
            continue
        coll0, _, col1 = col[2:-2].partition("', '")
        if col1 == "COIN":
            continue
        groups[coll0].append(torch.as_tensor(values))
    
    result = {k: torch.stack(v, axis=1) for k,v in groups.items()}
    return result
    
        

dotenv.load_dotenv()
@click.command()
def main():
    s3_uri = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
    ray.data.DataContext.get_current().enable_progress_bars = False
    ray.init(logging_level="INFO")
    ds = ray.data.read_parquet(s3_uri)
    num_cols = len(ds.columns())
    num_rows = ds.count()
    
    total_start_ts = time.time()
    with tqdm(total=num_rows * num_cols //  M, unit="Mvalues") as pbar:
        train_dataloader = ds.iter_torch_batches(batch_size=800_000, collate_fn=collate_fn)
        for batch in train_dataloader:
            pbar.update(len(batch["X"]) * num_cols // M)

    total_mvalues_per_sec = num_cols * num_rows / M / (time.time() - total_start_ts)
    logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")
    print(ds.stats())

if __name__ == "__main__":
    main()

# mvalues/s=87.63