from pathlib import Path
import shutil
import time
from loguru import logger
import human_readable as hr
from datetime import datetime 
import attr

@attr.define
class Result:
    id: str
    elapsed: float
    values: int = 0
    
    def __str__(self):
        values_per_sec = self.values / self.elapsed
        return f"id:{self.id} elapsed:{self.elapsed:.1f}s throughput:{hr.int_word(values_per_sec)}vals/sec"
    
@attr.define
class ExceptionResult:
    name: str
    exception: str
    
    def __str__(self):
        return f"name:{self.name}: exception:{self.exception}"