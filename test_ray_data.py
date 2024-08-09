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

    ray.data.set_progress_bars(False)
    ray.init(logging_level="INFO")
    ds = ray.data.read_parquet(s3_uri)
    logger.info(f"Dataset: rows={ds.count()} cols={len(ds.columns())}")

    # Define batch size and shuffle the dataset
    batch_size = 1024 * 1024
    # dataset = dataset.random_shuffle()  # Shuffle the entire dataset
    # row_count = dataset.count()
    # Use iter_batches to iterate over batches directly
    total_start_ts = time.time()
    with tqdm(total=ds.count() // batch_size) as pbar:
        start_ts = time.time()
        train_dataloader = ds.iter_torch_batches(batch_size=batch_size, device="cpu")
        for batch in train_dataloader:
            mvalues_per_sec = cols * batch_size / (time.time() - start_ts) / M
            start_ts = time.time()
            pbar.set_postfix_str(f"mvalues/s={mvalues_per_sec:.2f}")
            pbar.update()

    total_mvalues_per_sec = cols * ds.count() / M / (time.time() - total_start_ts)
    logger.info(f"Total mvalues/s={total_mvalues_per_sec:.2f}")

if __name__ == "__main__":
    main()