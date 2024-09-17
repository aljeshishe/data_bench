import time
import ray
from loguru import logger

def main():
    #  ray.data.DataContext.get_current().use_push_based_shuffle = False
    # ray.init(address='local', ignore_reinit_error=True, include_dashboard=False, logging_level='warning')
    # ds = ray.data.read_numpy("s3://alblml/kaggle/preprocessing/240904_171741_EDIM/X")
    ds = ray.data.read_parquet("s3://alblml/kaggle/preprocessing/240911_141548_1IT4_val")
 
    start_ts = time.time()
    ds.random_shuffle().materialize()
    logger.info(f"Elapsed={time.time() - start_ts:.2f}")
    print(ds.stats())
    
if __name__ == "__main__":
    main()
    
# local shuffle  of numpy files: 682.41s
# local shuffle  of numpy files: 916.57 use_push_based_shuffle = True
# local shuffle  of numpy files: 779.78  use_push_based_shuffle = False
# local shuffle  of pqrquet files: 855.41s
