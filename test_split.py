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
    return 

def create_expr(start: datetime, end: datetime, span: timedelta):
    expr = (pc.field("__index_level_0__") < datetime(2000, 1, 1, 0))
    dt = datetime.fromisoformat('2024-05-01')
    while dt <  datetime.fromisoformat('2024-06-30'):
        logger.info(f"period {dt:%Y-%m-%d %H:%M} -> {dt + span:%Y-%m-%d %H:%M}")
        expr = expr | ((pc.field("__index_level_0__") > dt) & (pc.field("__index_level_0__") < dt + span))
        dt += span * 2
    
def main():
    PATH = "s3://alblml/kaggle/preprocessing/240909_101936_VSD0"
    # PATH = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
    ray.init()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    with request_resources(num_cpus=700) as resources:
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
            out_path = f"{PATH}_train"
            logger.info(f"Writing to {out_path}")
            remove(out_path)
            ds.write_parquet(out_path) 
            logger.info(f"write_parquet Elapsed={sw.elapsed:.2f}")
            logger.info(ds.stats())

    
# def main():
#     ray.init(address='ray://18.177.232.240:10001', ignore_reinit_error=True, include_dashboard=False, logging_level='info')
#     print(ray.get(func.remote()))
    
if __name__ == "__main__":
    main()
    
# local shuffle  of numpy files: 682.41s
# local shuffle  of numpy files: 916.57 use_push_based_shuffle = True
# local shuffle  of numpy files: 779.78  use_push_based_shuffle = False
# local shuffle  of pqrquet files: 855.41s
# 700 cpus
# read_parquet Elapsed=18.67
# random_shuffle Elapsed=190.42
# write_parquet Elapsed=55.36