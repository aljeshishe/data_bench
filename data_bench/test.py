
from datetime import datetime
import itertools
from multiprocessing import Process, Queue
from pathlib import Path
import attr
from loguru import logger
from .resutls import Result, ExceptionResult
from . import utils

@attr.define(slots=False)
class Test:
    name: str = ""
    params: utils.Params = None

    def from_product(self, **kwargs):
        keys = list(kwargs.keys())
        values = list(kwargs.values())
        for combination in itertools.product(*values):
            yield attr.evolve(self, **dict(zip(keys, combination)))
        return

    def run(self):
        results = Queue()
        p = Process(target=self.process_run, kwargs=dict(results=results))
        p.start()
        p.join()
        return results.get()
        
    def process_run(self, results):
        try:
            self._setup()
            
            with utils.Stopwatch() as sw:
                self._run()
            
            results.put(Result(id=self.id, elapsed=sw.elapsed, values=self.params.rows * self.params.cols))
            
            self._teardown()
        except Exception as exception:
            logger.exception(exception)
            results.put(ExceptionResult(name=self.name, exception=exception))        
    
    @property
    def id(self):
        params_str = " ".join(f"{param}={getattr(self, param)}" for param in self.param_names)
        return f"{self.__class__.__name__} {params_str}"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def _setup(self):
        pass    
    
    def _run(self):
        pass
    
    def _teardown(self):
        pass

    @property
    def param_names(self):
        return []

NOW = datetime.now().isoformat()

@attr.define(slots=False)
class StorageTest(Test):
    local_path: str = str(Path(__file__).parent / "tmp" / NOW)
    s3_path: str = f"s3://tmp-grachev/bench/{NOW}"
    storage: str = "local"

    @property
    def path(self):
        match self.storage:
            case "local":
                return f"{self.local_path}/{self.id}"
            case "s3":
                return f"{self.s3_path}/{self.id}"
        
    def _teardown(self):
        if self.params.cleanup:
            utils.remove(self.path)
        