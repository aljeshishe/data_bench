from collections import defaultdict
import numpy as np
import ray

ray.data.DataContext.get_current().enable_progress_bars = True
ray.init(logging_level="error", log_to_driver=True)

def preprocess(df):
    print(df)
    return dict(X=df["X"], spread_spot=df["spread_spot"])
    # return df
#     print(df)
    # pandas df filter by datetime
    # df = df.set_index("time")
    # df = df.between_time("06:00", "22:00")
    #(MapBatches(preprocess) pid=1634926) 2024-06-24 06:29:35.798242                       -4.338771e-05  ...    1.759195                                                                                                     
# (MapBatches(preprocess) pid=1634926) 2024-06-24 06:29:35.999966                       -6.656569e-06  ...    1.759195  
    # groups = defaultdict(list)
    # for col, values in batch.items():
    #     if col == "__index_level_0__":
    #         continue
    #     coll0, _, col1 = col[2:-2].partition("', '")
    #     if col1 == "COIN":
    #         continue
    #     groups[coll0].append(values)
    
    # result = {k: np.stack(v, axis=1) for k,v in groups.items()}
    # return result
    
path = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN/BNB_2024-06-24_06-00-00_21600s.parquet"
ds = ray.data.read_parquet(path)
ds = ds.map_batches(preprocess, batch_format="pandas")
print("done")
print(ds.to_pandas())