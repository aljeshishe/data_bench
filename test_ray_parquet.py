import time
import click
import ray
import ray.data
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_bench import utils
from data_bench.utils import M
from tqdm import tqdm
from loguru import logger
import dotenv 
import os 
import human_readable as hr

def collate_fn(batch):
    tensor = torch.stack([torch.as_tensor(array) for array in batch.values()], axis=1)
    return dict(X=tensor)
    
dotenv.load_dotenv()
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
@click.option("-p", "--prefetch_batches", type=int, default=3)
@click.option("-o", "--collate", is_flag=True)
@click.option("-m", "--map_batches", is_flag=True)
def main(create_dataset, prefetch_batches, collate, map_batches):
    # params
    s3_uri = "s3://ab-users/grachev/ray_benchmark/20gb.parquet"
    mvalues = 12
    cols = 68
    n_files = 500
    batch_size = 800_000

    if create_dataset:
        utils.remove(s3_uri)
        file_name = utils.write_parquet_dataset(s3_uri, mvalues=mvalues, cols=cols)
        utils.clone_s3_file(file_name, n_files=n_files)

    size_str = hr.file_size(utils.s3_dir_size(s3_uri))
    logger.info(f"Dataset: mvalues={mvalues} rows={mvalues // cols} cols={cols} size={size_str}")

    ray.data.DataContext.get_current().enable_progress_bars = False
    ray.init(logging_level="INFO", include_dashboard=True, dashboard_host="0.0.0.0")
    ds = ray.data.read_parquet(s3_uri)

    if collate:
        logger.info(f"Merging columns in collate_fn")

    if map_batches:
        logger.info(f"Merging columns in map_batches")
        ds = ds.map_batches(collate_fn)
        
    train_dataloader = ds.iter_torch_batches(batch_size=batch_size, prefetch_batches=prefetch_batches, collate_fn=collate_fn if collate else None)
    # utils.show_dataloader_info(train_dataloader)
    it = iter(train_dataloader)
    val1 = next(it)
    val2 = next(it)
    
    # utils.benchmark(utils.iter_timeit(train_dataloader), total_mvalues=mvalues * n_files)

    print(ds.stats())

if __name__ == "__main__":
    main()

# mvalues/s=123.55 with preprocessing
# mvalues/s=208.62 with preprocessing and materialize
# mvalues/s=569.62 no preprocessing

# r5.4xlarge
# p25=0.22
# p50=0.27
# p75=0.42
# p90=0.50
# Total mvalues/s=143.67

# r5.4xlarge with -o
# p25=0.11
# p50=0.33
# p75=0.60
# p90=0.91
# Total mvalues/s=119.05

# r5.4xlarge with -m
# p25=0.37
# p50=0.49
# p75=0.71
# p90=1.04
# Total mvalues/s=82.70

# g4dn.12xlarge prefetch_batches=3
# p25=0.04
# p50=0.07
# p75=0.13
# p90=0.22
# Total mvalues/s=405.92

# g4dn.12xlarge prefetch_batches=3 -0
# p25=0.24
# p50=0.34
# p75=0.44
# p90=0.63
# Total mvalues/s=134.71

# g4dn.12xlarge prefetch_batches=3 -m
# p25=0.00
# p50=0.06
# p75=0.41
# p90=0.60
# Total mvalues/s=212.12