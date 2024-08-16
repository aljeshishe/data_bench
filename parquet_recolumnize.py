from collections import defaultdict
import numpy as np
import ray

ray.init()

def preprocess(batch):
    groups = defaultdict(list)
    for col, values in batch.items():
        if col == "__index_level_0__":
            continue
        coll0, _, col1 = col[2:-2].partition("', '")
        if col1 == "COIN":
            continue
        groups[coll0].append(values)
    
    result = {k: np.stack(v, axis=1) for k,v in groups.items()}
    return result
    
path = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
ds = ray.data.read_parquet(path)
ds = ds.map_batches(preprocess)
ds.write_parquet("s3://alblml/kaggle/preprocessing/240813_132219_XVEN_recolumnize")