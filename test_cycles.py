from urllib.parse import urlparse
import boto3
import io
import numpy as np
import torch
import pandas as pd
import dotenv
dotenv.load_dotenv()

def read_numpy():
    path = "s3://tmp-grachev/data.npy"
    _, bucket, path, *_ = urlparse(path)
    obj = boto3.client('s3').get_object(Bucket=bucket, Key=path.lstrip("/"))
    with io.BytesIO(obj["Body"].read()) as f:
        f.seek(0)
        tensor = torch.from_numpy(np.load(f, allow_pickle=True))
        print(f"tensor={tensor.shape}")

def read_parquet():
    path = "s3://tmp-grachev/data.parquet"
    df = pd.read_parquet(path)
    tensor =  torch.from_numpy(df.to_numpy())
    print(f"tensor={tensor.shape}")

for i in range(5):
    read_numpy()