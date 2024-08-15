import time
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

def preprocess(batch):
    return batch

dotenv.load_dotenv()
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
def main(create_dataset):
    # params
    s3_uri = "s3://ab-users/grachev/ray_benchmark/20gb.parquet"
    mvalues = 3000 # 20 GB
    part_mvalues = 8
    cols = 100

    if create_dataset:
        utils.remove(s3_uri)
        utils.write_dummy_dataset(s3_uri, mvalues=mvalues, part_mvalues=part_mvalues, cols=cols)

    ray.data.DataContext.get_current().enable_progress_bars = False
    ray.init(logging_level="INFO")
    ds = ray.data.read_numpy(s3_uri, shuffle="files")
    total_rows = ds.count()
    logger.info(f"Dataset: rows={total_rows} cols={cols}")

    # ds = ds.map_batches(preprocess)
    # ds = ds.materialize()
    # Define batch size and shuffle the dataset
    batch_size = 800_000
    # dataset = dataset.random_shuffle()  # Shuffle the entire dataset
    # row_count = dataset.count()
    # Use iter_batches to iterate over batches directly
    total_start_ts = time.time()
    with tqdm(total=total_rows * cols //  M, unit="Mvalues") as pbar:
        train_dataloader = ds.iter_torch_batches(batch_size=batch_size)
        for batch in train_dataloader:
            pbar.update(len(batch["data"]) * cols // M)

    total_mvalues_per_sec = cols * total_rows / M / (time.time() - total_start_ts)
    logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")
    print(ds.stats())

if __name__ == "__main__":
    main()

# mvalues/s=123.55 with preprocessing
# mvalues/s=208.62 with preprocessing and materialize
# mvalues/s=569.62 no preprocessing