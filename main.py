import warnings
import os
import click
from loguru import logger
from data_bench import utils
from data_bench.pandas_tests import PandasReadTest, PandasWriteTest
from data_bench.dask_tests import DaskReadTest, DaskWriteTest
from human_readable import numbers 

warnings.filterwarnings("ignore", category=FutureWarning)

N_ = numbers.N_
numbers.POWERS = [10**x for x in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]

numbers.HUMAN_POWERS = (
    N_("K"),
    N_("M"),
    N_("B"),
    N_("T"),
    N_("QD"),
    N_("QT"),
    N_("SX"),
    N_("SP"),
    N_("O"),
    N_("N"),
    N_("D"),
    N_("G"),
)


@click.command()
@click.option("-f", "--fast", is_flag=True, show_default=True, default=False, help="fast mode")
def main(fast):
    os.environ["AWS_PROFILE"] = "abml"
    tests = [
        # , compression=["snappy", "gzip", "brotli", "lz4", "zstd"]
        *PandasReadTest().from_product(engine=["pyarrow", "fastparquet"], storage=["s3", "local"]),
        *PandasWriteTest().from_product(engine=["pyarrow", "fastparquet"], storage=["s3", "local"]),
        *DaskReadTest().from_product(storage=["s3", "local"]),
        *DaskWriteTest().from_product(storage=["s3", "local"]),
        ]

    params = utils.Params(rows=1024 if fast else 1024**2 , cols=100, cleanup=True)
    for test in tests:
        test.params = params
        logger.info(test.run())
    

if __name__ == "__main__":
    main()
    
