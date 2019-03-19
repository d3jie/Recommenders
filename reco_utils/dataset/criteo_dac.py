# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import atexit
import tarfile
from tempfile import TemporaryDirectory

try:
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
except ImportError:
    pass  # so the environment without spark doesn't break

from reco_utils.dataset.url_utils import maybe_download
from reco_utils.common.notebook_utils import is_databricks


URL = dict(
    full="https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz",
    sample="http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"
)

LABEL = ['label']
INT_FIELDS = ["int{0:02d}".format(i) for i in range(13)]
CAT_FIELDS = ["cat{0:02d}".format(i) for i in range(26)]
HEADER = LABEL + INT_FIELDS + CAT_FIELDS


def load_pandas_df(
        size="sample",
        local_cache_path=None,
        header=None
):
    """Loads the Criteo DAC dataset as pandas.DataFrame. This function download, untar, and load the dataset.
    The schema is:
    <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
    More details: http://labs.criteo.com/2013/12/download-terabyte-click-logs/
    Args:
        size (str): Dataset size. It can be "sample" or "full".
        local_cache_path (str): Path where to cache the tar.gz file locally
        header (list): Dataset header names.
    Returns:
        pd.DataFrame: Criteo DAC sample dataset.
    """
    filepath, filename = handle_cache(size=size, cache_path=local_cache_path)
    download_criteo(size=size, filename=filename, work_directory=filepath)
    data_path = extract_criteo(size, filename=filename, work_directory=filepath)
    return pd.read_csv(data_path, sep="\t", header=None, names=header or HEADER)


def load_spark_df(
        spark,
        size="sample",
        header=None,
        local_cache_path=None,
        dbfs_datapath="dbfs:/FileStore/dac",
        dbutils=None,
):
    """Loads the Criteo DAC dataset as pySpark.DataFrame.
    The schema is:
    <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
    More details: http://labs.criteo.com/2013/12/download-terabyte-click-logs/
    Args:
        spark (pySpark.SparkSession): Spark session
        size (str): Dataset size. It can be "sample" or "full"
        local_cache_path (str or None): Path where to cache the tar.gz file locally, optional
        header (list): Dataset header names, optional
        dbfs_datapath (str): Where to store the extracted files on Databricks
        dbutils (Databricks.dbutils): Databricks utility object

    Returns:
        pySpark.DataFrame: Criteo DAC training dataset.
    """

    filepath, filename = handle_cache(size=size, cache_path=local_cache_path)
    download_criteo(size=size, filename=filename, work_directory=filepath)
    data_path = extract_criteo(size=size, filename=filename, work_directory=filepath)

    if is_databricks():
        if dbutils is None:
            raise ValueError(
                "To use on a Databricks notebook, dbutils object should be passed as an argument"
            )
        # needs to be on dbfs to load
        dbutils.fs.cp("file:{}".format(data_path), dbfs_datapath, recurse=True)
        data_path = dbfs_datapath

    # create schema
    header = header or HEADER
    fields = [StructField(header[i], IntegerType()) for i in range(len(INT_FIELDS))]
    fields += [StructField(header[len(INT_FIELDS) + i], StringType()) for i in range(len(CAT_FIELDS))]
    schema = StructType(fields)

    return spark.read.csv(data_path, schema=schema, sep="\t", header=False)


def download_criteo(size="sample", filename="dac.tgz", work_directory="."):
    """Download criteo dataset as a compressed file.
    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample"
        filename (str): Filename
        work_directory (str): Working directory
    Returns:
        str: Path of the downloaded file
    """
    return maybe_download(URL[size.lower()], filename, work_directory)


def extract_criteo(size, filename, work_directory):
    """Extract Criteo dataset tar
    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample"
        filename (str): Filename
        work_directory (str): Working directory
    """

    with tarfile.open(os.path.join(work_directory, filename)) as tar:
        tar.extractall(work_directory)

    if size == 'full':
        data_path = os.path.join(work_directory, 'train.txt')
    elif size == 'sample':
        data_path = os.path.join(work_directory, 'dac_sample.txt')
    else:
        raise ValueError('Invalid size option, can be one of ["sample", "full"]')

    return data_path


def handle_cache(size, cache_path=None):
    """Creates a cache directory and registers cleanup
    Args:
        size (str): data size [full|sample]
        cache_path (str): path to cache file, optional
    """
    if cache_path is None:
        tmp_dir = TemporaryDirectory()
        atexit.register(tmp_dir.cleanup)
        cache_path = os.path.join(tmp_dir.name, os.path.basename(URL[size].lower()))
    return os.path.split(os.path.realpath(cache_path))

