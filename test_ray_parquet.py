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


dotenv.load_dotenv()
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
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
    ray.init(logging_level="INFO")
    ds = ray.data.read_parquet(s3_uri)

    train_dataloader = ds.iter_torch_batches(batch_size=batch_size)
    utils.benchmark(utils.iter_timeit(train_dataloader), total_mvalues=mvalues * n_files)

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