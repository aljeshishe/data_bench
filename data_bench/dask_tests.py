
import attr
import pandas as pd
from .test import Test, StorageTest
from . import utils

import dask.array as da
import dask.dataframe as dd
import attr
import numpy as np
import pandas as pd
from loguru import logger
import human_readable as hr
from . import utils



@attr.define(slots=False, kw_only=True)
class DaskReadTest(StorageTest):
    engine: str = "auto"
    
    @property
    def param_names(self):
        return ["engine", "storage"]
    
    def _setup(self):
        utils.dask_df(rows=self.params.rows, cols=self.params.cols).to_parquet(self.path)
    
    def _run(self):
        dd.read_parquet(self.path, engine=self.engine).persist()
    


@attr.define(slots=False)
class DaskWriteTest(StorageTest):
    engine: str = "auto"
    compression: str = None
    
    @property
    def param_names(self):
        return ["engine", "compression", "storage"]
        
    def _setup(self):
        self.df = utils.dask_df(rows=self.params.rows, cols=self.params.cols)
        
    def _run(self):
        self.df.to_parquet(self.path, engine=self.engine, compression=self.compression)
    
