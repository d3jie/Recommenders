# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import atexit
import os
import tarfile
from tempfile import TemporaryDirectory

import pandas as pd

from reco_utils.dataset.url_utils import maybe_download


def load_pandas_df(
    local_cache_path=None,
    label_col="Label",
    numeric_cols=None,
    categorical_cols=None,
):
    """Loads the Criteo DAC dataset as pandas.DataFrame.

      Download the dataset from http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz, untar, and load

      For the data, the schema is:

      <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

      Args:
          local_cache_path (str): Path where to cache the tar.gz file locally
          label_col (str): The column of Label.
          numeric_cols (list): The names of numerical features.
          categorical_cols (list): The names of categorical features.
      Returns:
          pandas.DataFrame: Criteo DAC sample dataset.
      """

    if numeric_cols is None:
        numeric_cols = ["I{}".format(i) for i in range(1, 14)]
    if categorical_cols is None:
        categorical_cols = ["C{}".format(i) for i in range(1, 27)]

    column_names = [label_col] + numeric_cols + categorical_cols

    # download and untar the data file
    data_path = _load_datafile(local_cache_path=local_cache_path)

    return pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=column_names,
    )


def _load_datafile(local_cache_path=None):
    """ Download and extract file """

    if local_cache_path is None:
        tmp_dir = TemporaryDirectory()
        local_cache_path = os.path.join(tmp_dir.name, 'dac_sample.tar.gz')
        atexit.register(tmp_dir.cleanup)

    path, filename = os.path.split(os.path.realpath(local_cache_path))

    # download if it doesn't already exist locally
    maybe_download(
        "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz",
        filename,
        work_directory=path,
    )

    with tarfile.open(local_cache_path) as tar:
        tar.extract("dac_sample.txt", path)

    return os.path.join(path, "dac_sample.txt")
