from urllib.parse import urlparse
import boto3
import time
import ray
import logging

import pyarrow.compute as pc
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

class Stopwatch:
    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def elapsed(self):
        return time.time() - self.start

    def __exit__(self, *args):
        self.end = time.time()

    def __str__(self):
        return f"{self.elapsed:.2f} seconds"
    
class request_resources:
    """Request resources from ray cluster.
    Usage 1: request resouces, with or without manual release
    ```
    request_resources(num_cpus=10)
    ...
    request_resources(num_cpus=0)
    ```

    Usage 2: rely on context manager to release resources
    ```
    with request_resources(num_cpus=10):
        ...
    ```

    Args:
        num_cpus: Number of CPUs to request.
    """

    def __init__(self, num_cpus: int = 0):
        self.num_cpus = num_cpus
        logger.debug(f"Allocating cluster resources: {num_cpus} CPUs")
        ray.autoscaler.sdk.request_resources(num_cpus=num_cpus)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"Releasing cluster resources: 0 CPUs")
        ray.autoscaler.sdk.request_resources(num_cpus=0)
    
    def wait(self):
        while (cpus:=ray.available_resources().get("CPU", 0)) < self.num_cpus:
            logger.info(f"Waiting for resources {cpus}/{self.num_cpus}")
            time.sleep(1)
        
        logger.info(f"Resources allocated {cpus}/{self.num_cpus}")


def remove(path:  str):
    parsed = urlparse(path)
    boto3.resource('s3').Bucket(parsed.netloc).objects.filter(Prefix=parsed.path.lstrip("/")).delete()

def create_expr(start: datetime, end: datetime, span: timedelta):
    expr = (pc.field("__index_level_0__") < datetime(2000, 1, 1, 0))
    while start <  end:
        logger.info(f"period {start:%Y-%m-%d %H:%M} -> {start + span:%Y-%m-%d %H:%M}")
        expr = expr | ((pc.field("__index_level_0__") > start) & (pc.field("__index_level_0__") < start + span))
        start += span * 2
    
def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Starting")

    # PATH = "s3://alblml/kaggle/preprocessing/240910_142947_83PD" # test dataset
    PATH = "s3://alblml/kaggle/preprocessing/240911_141548_1IT4" # full dataset
    # PATH = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
    # PATH = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
    
    with request_resources(num_cpus=1) as resources:
        resources.wait()

        expr = create_expr(start=datetime.fromisoformat('2024-05-01'), end=datetime.fromisoformat('2024-06-30'), span=timedelta(hours=8))
        with Stopwatch() as sw:
            ds = ray.data.read_parquet(PATH, filter=expr)
            logger.info(f"read_parquet Elapsed={sw.elapsed:.2f}")
            logger.info(ds.stats())
        
        with Stopwatch() as sw:
            ds = ds.random_shuffle().materialize()
            logger.info(f"random_shuffle Elapsed={sw.elapsed:.2f}")
        
        with Stopwatch() as sw:
            out_path = f"{PATH}_test"
            logger.info(f"Writing to {out_path}")
            remove(out_path)
            ds.write_parquet(out_path) 
            logger.info(f"write_parquet Elapsed={sw.elapsed:.2f}")
            logger.info(ds.stats())

    
    
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level='info')
    print(ray.get(ray.remote(num_cpus=0)(main).remote()))
    # main()
    
# local shuffle  of numpy files: 682.41s
# local shuffle  of numpy files: 916.57 use_push_based_shuffle = True
# local shuffle  of numpy files: 779.78  use_push_based_shuffle = False
# local shuffle  of pqrquet files: 855.41s
# 700 cpus
# read_parquet Elapsed=18.67
# random_shuffle Elapsed=190.42
# write_parquet Elapsed=55.36