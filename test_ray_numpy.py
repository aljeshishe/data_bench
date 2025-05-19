import time
import click
import fsspec
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
        


@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
@click.option("-p", "--prefetch_batches", type=int, default=3)
def main(create_dataset, prefetch_batches):
    # params
    s3_uri = "s3://ab-users/grachev/ray_benchmark/1mb_per_file.numpy"
    cols = 70
    rows = 357_000 // (9*9)
    n_files = 50
    batch_size = 800_000

    dotenv.load_dotenv()

    protocol = fsspec.utils.get_protocol(s3_uri)
    fs = fsspec.filesystem(protocol)

    if create_dataset:
        utils.remove(s3_uri, fs=fs)
        file_url = utils.write_numpy_dataset(fs=fs, path=s3_uri, rows=rows, cols=cols)
        utils.clone_file(file_url, n_files=n_files)


    size_str = hr.file_size(utils.s3_dir_size(s3_uri))
    logger.info(f"Dataset:  rows={rows} cols={cols} size={size_str}")

    ray.data.DataContext.get_current().enable_progress_bars = False
    ray.init(logging_level="INFO", include_dashboard=True, dashboard_host="0.0.0.0", 
             num_cpus=32, 
             object_store_memory=2e9, 
             _system_config={'automatic_object_spilling_enabled': False})
    data_ctx = ray.data.DataContext.get_current()
    data_ctx.execution_options.preserve_order = True
    # data_ctx.execution_options.resource_limits.num_cpus = num_cpus
    # data_ctx.execution_options.resource_limits.object_store_memory = 10e9
    #data_ctx.enable_progress_bars = False
    #data_ctx.print_on_execution_start = False

    results = []
    ds = ray.data.read_numpy(s3_uri, shuffle="files")
    train_dataloader = ds.iter_torch_batches(batch_size=batch_size, prefetch_batches=prefetch_batches)
    utils.benchmark(utils.iter_timeit(train_dataloader), total_mvalues=rows * cols * n_files // 1024**2)
    # batch_str = " ".join({f"{k}:{v.shape}" for k, v in batch.items()})
    # logger.info(f"Batch: {batch_str}")
    # utils.benchmark(utils.iter_timeit(train_dataloader), total_mvalues=mvalues * n_files)

    print(ds.stats())

if __name__ == "__main__":
    main()

# mvalues/s=123.55 with preprocessing
# mvalues/s=208.62 with preprocessing and materialize
# mvalues/s=569.62 no preprocessing

# r5.4xlarge
# p25=0.10
# p50=0.14
# p75=0.28
# p90=0.45
# Total mvalues/s=212.65

# g4dn.12xlarge prefetch_batches=3
# p25=0.00
# p50=0.05
# p75=0.12
# p90=0.23
# Total mvalues/s=514.14