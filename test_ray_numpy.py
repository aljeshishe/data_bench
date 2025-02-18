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
        

def read_numpy(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    parallelism: int = -1,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider: Optional[BaseFileMetadataProvider] = None,
    partition_filter: Optional[PathPartitionFilter] = None,
    partitioning: Partitioning = None,
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal["files"], None] = None,
    file_extensions: Optional[List[str]] = NumpyDatasource._FILE_EXTENSIONS,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **numpy_load_args,
) -> Dataset:

    if meta_provider is None:
        meta_provider = get_generic_metadata_provider(NumpyDatasource._FILE_EXTENSIONS)

    datasource = NumpyDatasource(
        paths,
        numpy_load_args=numpy_load_args,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return read_datasource(
        datasource,
        parallelism=parallelism,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )


dotenv.load_dotenv()
@click.command()
@click.option("-c", "--create_dataset", is_flag=True)
@click.option("-p", "--prefetch_batches", type=int, default=3)
def main(create_dataset, prefetch_batches):
    # params
    s3_uri = "s3://ab-users/grachev/ray_benchmark/20gb.numpy"
    mvalues = 12 # 20 GB
    cols = 68
    n_files = 500
    batch_size = 800_000

    if create_dataset:
        utils.remove(s3_uri)
        file_name = utils.write_numpy_dataset(s3_uri, mvalues=mvalues, cols=cols)
        utils.clone_s3_file(file_name, n_files=n_files)

    size_str = hr.file_size(utils.s3_dir_size(s3_uri))
    logger.info(f"Dataset: mvalues={mvalues} rows={mvalues // cols} cols={cols} size={size_str}")

    ray.data.DataContext.get_current().enable_progress_bars = False
    ray.init(logging_level="INFO", include_dashboard=True, dashboard_host="0.0.0.0")
    data_ctx = ray.data.DataContext.get_current()
    data_ctx.execution_options.preserve_order = True
    #data_ctx.execution_options.resource_limits.num_cpus = num_cpus
    data_ctx.execution_options.resource_limits.object_store_memory = 7e10
    #data_ctx.enable_progress_bars = False
    #data_ctx.print_on_execution_start = False

    ds = ray.data.read_numpy(s3_uri, shuffle="files")

    train_dataloader = ds.iter_torch_batches(batch_size=batch_size, prefetch_batches=prefetch_batches)
    while True:
        logger.info(f"new Epoch")
        for batch in train_dataloader:
            pass
    batch_str = " ".join({f"{k}:{v.shape}" for k, v in batch.items()})
    logger.info(f"Batch: {batch_str}")
    utils.benchmark(utils.iter_timeit(train_dataloader), total_mvalues=mvalues * n_files)

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