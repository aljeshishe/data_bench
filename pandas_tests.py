
import attr
import pandas as pd
from test import Test, StorageTest
import utils

import dask.array as da
import dask.dataframe as dd
import attr
import numpy as np
import pandas as pd
from loguru import logger
import human_readable as hr
import utils



@attr.define(slots=False, kw_only=True)
class PandasReadTest(StorageTest):
    engine: str = "auto"
    
    @property
    def param_names(self):
        return ["engine", "storage"]
    
    def _setup(self):
        utils.dask_df(rows=self.params.rows, cols=self.params.cols).to_parquet(self.path)
    
    def _run(self):
        pd.read_parquet(self.path, engine=self.engine)
    

@attr.define(slots=False)
class PandasWriteTest(StorageTest):
    engine: str = "auto"
    compression: str = None
    
    @property
    def param_names(self):
        return ["engine", "compression", "storage"]
        
    def _setup(self):
        self.df = utils.pandas_df(rows=self.params.rows, cols=self.params.cols)
        
    def _run(self):
        self.df.to_parquet(self.path)
    
