# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None  # skip this import if we are in pure python environment


def start_or_get_spark(app_name="Sample", url="local[*]", memory="10G", env_vars=None):
    """Start Spark if not started

    Args:
        app_name (str): Set name of the application
        url (str): URL for spark master
        memory (str): Size of memory for spark driver
        env_vars (dict): optional kwargs to use for setting environment variables
    Returns:
        obj: Spark context.
    """

    if env_vars is not None:
        os.environ.update(env_vars)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )

    return spark
