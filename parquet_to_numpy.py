from collections import defaultdict
import tempfile
from dotenv import load_dotenv
import numpy as np
import ray
import s3fs 
from ray.util.multiprocessing import Pool
import pandas as pd

from data_bench import utils

ray.data.DataContext.get_current().enable_progress_bars = True
ray.init(logging_level="debug", log_to_driver=True)

def preprocess(file, new_path):
    df = pd.read_parquet(file)
    level0_cols = list(df.columns.levels[0])
    dfs = {col: df[col] for col in level0_cols}
    
    dfs["X"] = dfs["X"].drop(columns=["COIN"])
    
    for name, df in dfs.items():
        prefix, _, file_name = file.rpartition("/")
        new_file = f"{new_path}/{name}/{file_name}.npy"
        array = df.to_numpy()
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = f"{tmp.name}.npy"
            np.save(file=tmp_path, arr=array)
            utils.upload_file(tmp_path, new_file)
        
load_dotenv()
path = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN"
new_path = "s3://alblml/kaggle/preprocessing/240813_132219_XVEN_numpy1"
files = s3fs.S3FileSystem().ls(path)
print(files)
args = [(f"s3://{file}", new_path) for file in files]
with Pool(processes=16) as pool:
    print(pool.starmap(preprocess, args))

